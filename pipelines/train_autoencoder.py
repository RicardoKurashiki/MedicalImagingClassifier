import os
import time

import json
import numpy as np
import psutil

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import TemporaryDirectory

import pickle as pk

from models import AutoEncoder, ClassificationModel
from utils import plot_pca, FeatureDataset

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

EARLY_STOPPING_PATIENCE = 100


def get_memory_usage():
    metrics = {}

    # CPU
    process = psutil.Process(os.getpid())
    metrics["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

    # GPU
    if torch.cuda.is_available():
        metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        metrics["gpu_memory_max_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
        )
        metrics["gpu_utilization"] = (
            torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else None
        )
    else:
        metrics["gpu_memory_allocated_mb"] = None
        metrics["gpu_memory_reserved_mb"] = None
        metrics["gpu_memory_max_allocated_mb"] = None
        metrics["gpu_utilization"] = None

    return metrics


def get_model_size(model):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb


def validate_autoencoder(
    model,
    source: DataLoader,
    target: DataLoader,
    criterion=nn.MSELoss(),
    verbose=False,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    batch_times = []

    with torch.no_grad():
        pbar = tqdm(
            zip(source, target),
            desc="Val     ",
            unit="batch",
            leave=False,
            disable=not verbose,
        )
        for (source_inputs, _), (target_inputs, _) in pbar:
            batch_start = time.time()
            source_inputs = source_inputs.to(device, non_blocking=True)
            target_inputs = target_inputs.to(device, non_blocking=True)

            x_recon, _ = model(source_inputs)
            loss = criterion(x_recon, target_inputs)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            total_loss += loss.item() * source_inputs.size(0)
            total_samples += source_inputs.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss, batch_times, total_samples


def train_autoencoder(
    model,
    source_train: DataLoader,
    target_train: DataLoader,
    source_val: DataLoader = None,
    target_val: DataLoader = None,
    epochs=500,
    output_path="./results/",
    verbose=True,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    since = time.time()
    metrics = {
        "total_time_seconds": 0,
        "epoch_times": [],
        "batch_times": {"train": [], "val": []},
        "throughput": {"train": [], "val": []},  # samples/segundo
        "memory_usage": [],
        "model_info": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "model_size_mb": get_model_size(model),
        },
    }

    initial_memory = get_memory_usage()
    metrics["initial_memory"] = initial_memory

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    early_stop = False

    with TemporaryDirectory() as tempdir:
        best_model_params_path = tempdir
        if verbose:
            print("Melhores pesos serão salvos temporariamente")

        for epoch in range(epochs):
            if early_stop:
                if verbose:
                    print(f"Early stopping acionado na época {epoch}")
                break

            epoch_start = time.time()
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                print("-" * 10)

            model.train()
            phase_start = time.time()
            epoch_train_loss = 0.0
            train_samples = 0
            train_batch_times = []

            pbar = tqdm(
                zip(source_train, target_train),
                desc="Train   ",
                unit="batch",
                leave=False,
                disable=not verbose,
            )

            for (source_inputs, _), (target_inputs, _) in pbar:
                batch_start = time.time()
                source_inputs = source_inputs.to(device, non_blocking=True)
                target_inputs = target_inputs.to(device, non_blocking=True)

                x_recon, _ = model(source_inputs)
                loss = criterion(x_recon, target_inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start
                train_batch_times.append(batch_time)

                epoch_train_loss += loss.item() * source_inputs.size(0)
                train_samples += source_inputs.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            phase_time = time.time() - phase_start
            avg_train_loss = (
                epoch_train_loss / train_samples if train_samples > 0 else 0.0
            )
            history["train_loss"].append(avg_train_loss)

            avg_batch_time = (
                sum(train_batch_times) / len(train_batch_times)
                if train_batch_times
                else 0
            )

            metrics["batch_times"]["train"].append(
                {
                    "epoch": epoch,
                    "avg_batch_time_seconds": avg_batch_time,
                    "total_batches": len(train_batch_times),
                    "total_time_seconds": phase_time,
                }
            )

            metrics["throughput"]["train"].append(
                {
                    "epoch": epoch,
                    "total_samples": train_samples,
                    "total_time_seconds": phase_time,
                }
            )

            if verbose:
                print(f"Train Loss: {avg_train_loss:.6f}")
                print(f"Train Time: {phase_time:.2f}s")

            if source_val is not None and target_val is not None:
                phase_start = time.time()
                avg_val_loss, val_batch_times, val_samples = validate_autoencoder(
                    model, source_val, target_val, criterion, verbose
                )
                phase_time = time.time() - phase_start

                history["val_loss"].append(avg_val_loss)
                avg_batch_time = (
                    sum(val_batch_times) / len(val_batch_times)
                    if val_batch_times
                    else 0
                )

                metrics["batch_times"]["val"].append(
                    {
                        "epoch": epoch,
                        "avg_batch_time_seconds": avg_batch_time,
                        "total_batches": len(val_batch_times),
                        "total_time_seconds": phase_time,
                    }
                )

                metrics["throughput"]["val"].append(
                    {
                        "epoch": epoch,
                        "total_samples": val_samples,
                        "total_time_seconds": phase_time,
                    }
                )

                if verbose:
                    print(f"Val Loss: {avg_val_loss:.6f}")
                    print(f"Val Time: {phase_time:.2f}s")

                if avg_val_loss < best_val_loss:
                    past_loss = best_val_loss
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    model.save_weights(best_model_params_path)
                    best_model_state = {
                        "encoder": model.encoder.state_dict().copy(),
                        "decoder": model.decoder.state_dict().copy(),
                    }

                    if verbose:
                        print(
                            f"Loss val melhorou de {past_loss:.6f} para {best_val_loss:.6f} [MELHOR]"
                        )
                else:
                    patience_counter += 1
                    if verbose:
                        print(
                            f"Loss val não melhorou de {best_val_loss:.6f} - Paciência: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                        )

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    early_stop = True
                    if verbose:
                        print("Early stopping acionado")
                    break
            else:
                if verbose:
                    print(f"Train Loss: {avg_train_loss:.6f}")

            epoch_time = time.time() - epoch_start
            metrics["epoch_times"].append({"epoch": epoch, "time_seconds": epoch_time})

            epoch_memory = get_memory_usage()
            metrics["memory_usage"].append({"epoch": epoch, **epoch_memory})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                print()

        if best_model_state is not None:
            model.encoder.load_state_dict(best_model_state["encoder"])
            model.decoder.load_state_dict(best_model_state["decoder"])
            if verbose:
                print(f"\nMelhor modelo carregado (val_loss: {best_val_loss:.6f})")
        elif os.path.exists(best_model_params_path):
            model.load_weights(best_model_params_path)
            if verbose:
                print(
                    f"\nMelhor modelo carregado do disco (val_loss: {best_val_loss:.6f})"
                )

    time_elapsed = time.time() - since
    metrics["total_time_seconds"] = time_elapsed

    final_memory = get_memory_usage()
    metrics["final_memory"] = final_memory

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if verbose:
        print(
            f"Treinamento completo em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        if source_val is not None and target_val is not None:
            print(f"Melhor val_loss: {best_val_loss:.6f}")

    model.save_weights(os.path.join(output_path, "autoencoder_weights.pt"))

    return model, history, metrics


def map_features_to_centroids(features, labels, centroids):
    mapped_features = np.zeros_like(features)
    features_by_labels = get_separated_labels(features, labels)

    for label in features_by_labels:
        if label not in centroids:
            mapped_features[np.array(labels) == label] = features_by_labels[label]
            continue

        label_indices = np.where(np.array(labels) == label)[0]
        label_features = features_by_labels[label]

        for idx, feature in enumerate(label_features):
            distances = np.linalg.norm(feature - centroids[label], axis=1)
            closest_centroid_idx = np.argmin(distances)
            mapped_features[label_indices[idx]] = centroids[label][closest_centroid_idx]

    return mapped_features


def get_separated_labels(features, labels):
    result = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        result[int(label)] = features[np.array(labels) == label]
    return result


def load_centroids(centroids_path, dataset_name, phase):
    result = {}
    for file in os.listdir(centroids_path):
        if file.startswith(f"{dataset_name}_{phase}_centroids_"):
            label = file.split("_")[-1].replace(".npy", "")
            result[int(label)] = np.load(os.path.join(centroids_path, file))
    return result


def extract_features(model, dataloader):
    model.eval()

    all_features = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            x_recon, _ = model(inputs)
        all_features.append(x_recon.cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def confusion_matrix(labels, predictions, num_classes=2):
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    if isinstance(predictions, list):
        predictions = torch.tensor(predictions)

    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for t, p in zip(labels, predictions):
        conf_matrix[t, p] += 1

    return conf_matrix


def get_class_metrics(conf_matrix, class_idx):
    num_classes = conf_matrix.shape[0]

    TP = conf_matrix[class_idx, class_idx].item()

    idx = torch.ones(num_classes, dtype=torch.bool)
    idx[class_idx] = False

    TN = (
        conf_matrix.sum()
        - conf_matrix[class_idx, :].sum()
        - conf_matrix[:, class_idx].sum()
        + TP
    ).item()

    FP = conf_matrix[idx, class_idx].sum().item()
    FN = conf_matrix[class_idx, idx].sum().item()

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


def evaluate_model(
    model,
    dataloader,
    num_classes=2,
):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    conf_matrix = confusion_matrix(all_labels, all_preds, num_classes)

    class_metrics = {}
    for c in range(num_classes):
        class_metrics[c] = get_class_metrics(conf_matrix, c)

    correct_predictions = conf_matrix.diag().sum().item()
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    avg_loss = (
        running_loss / total_samples
        if criterion is not None and total_samples > 0
        else None
    )

    results = {
        "accuracy": accuracy,
        "total_samples": int(total_samples),
        "correct_predictions": int(correct_predictions),
        "confusion_matrix": [
            [int(x) for x in row] for row in conf_matrix.numpy().tolist()
        ],
        "class_metrics": class_metrics,
        "predictions": [int(x) for x in all_preds],
        "labels": [int(x) for x in all_labels],
    }

    if avg_loss is not None:
        results["loss"] = float(avg_loss)

    return results


def classification_report(results, class_names=None):
    if class_names is None:
        num_classes = len(results["class_metrics"])
        class_names = [f"Class {i}" for i in range(num_classes)]

    num_classes = len(class_names)
    report = {}

    # Calcular métricas por classe
    for c, class_name in enumerate(class_names):
        metrics = results["class_metrics"][c]
        TP = metrics["TP"]
        FP = metrics["FP"]
        FN = metrics["FN"]
        TN = metrics["TN"]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "specificity": specificity,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        }

    macro_precision = (
        sum([report[name]["precision"] for name in class_names]) / num_classes
    )
    macro_recall = sum([report[name]["recall"] for name in class_names]) / num_classes
    macro_f1 = sum([report[name]["f1_score"] for name in class_names]) / num_classes

    report["macro_avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
    }

    report["accuracy"] = results["accuracy"]

    return report


def run(
    output_path,
    source_name,
    target_name,
    pretrained_model="densenet",
    epochs=500,
    batch_size=32,
    verbose=True,
):
    features_path = os.path.join(output_path, "features/")
    centroids_path = os.path.join(output_path, "centroids/")
    autoencoder_path = os.path.join(output_path, source_name, "autoencoder/")

    os.makedirs(autoencoder_path, exist_ok=True)

    source_centroids = load_centroids(
        centroids_path,
        source_name,
        "train",
    )

    target_train_features = np.load(
        os.path.join(features_path, f"{target_name}_train_features.npy")
    )
    target_train_labels = np.load(
        os.path.join(features_path, f"{target_name}_train_labels.npy")
    )

    target_val_features = np.load(
        os.path.join(features_path, f"{target_name}_val_features.npy")
    )
    target_val_labels = np.load(
        os.path.join(features_path, f"{target_name}_val_labels.npy")
    )

    target_test_features = np.load(
        os.path.join(features_path, f"{target_name}_test_features.npy")
    )
    target_test_labels = np.load(
        os.path.join(features_path, f"{target_name}_test_labels.npy")
    )

    pca_path = os.path.join(output_path, "pca", "pca_components.pkl")
    pca = pk.load(open(pca_path, "rb"))

    plot_pca(
        target_test_features,
        target_test_labels,
        output_path=os.path.join(
            output_path, "plots", f"{target_name}_with_{source_name}_centroids.png"
        ),
        centroids=source_centroids,
        pca=pca,
        title=f"{target_name} with {source_name} Centroids",
    )

    mapped_target_train_features = map_features_to_centroids(
        target_train_features, target_train_labels, source_centroids
    )

    mapped_target_val_features = map_features_to_centroids(
        target_val_features, target_val_labels, source_centroids
    )

    source_train_dataloader = DataLoader(
        FeatureDataset(target_train_features, target_train_labels),
        batch_size=batch_size,
    )
    source_val_dataloader = DataLoader(
        FeatureDataset(target_val_features, target_val_labels),
        batch_size=batch_size,
    )
    target_train_dataloader = DataLoader(
        FeatureDataset(mapped_target_train_features, target_train_labels),
        batch_size=batch_size,
    )
    target_val_dataloader = DataLoader(
        FeatureDataset(mapped_target_val_features, target_val_labels),
        batch_size=batch_size,
    )

    model = AutoEncoder()
    model, training_history, training_metrics = train_autoencoder(
        model,
        source_train_dataloader,
        target_train_dataloader,
        source_val_dataloader,
        target_val_dataloader,
        epochs=epochs,
        output_path=autoencoder_path,
        verbose=verbose,
    )

    history_path = os.path.join(
        output_path, f"{source_name}_autoencoder_{target_name}_training_history.json"
    )
    metrics_path = os.path.join(
        output_path, f"{source_name}_autoencoder_{target_name}_training_metrics.json"
    )

    os.makedirs(output_path, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    serializable_metrics = {
        "total_time_seconds": training_metrics["total_time_seconds"],
        "epoch_times": training_metrics["epoch_times"],
        "batch_times": {
            "train": training_metrics["batch_times"]["train"],
            "val": training_metrics["batch_times"]["val"],
        },
        "throughput": {
            "train": training_metrics["throughput"]["train"],
            "val": training_metrics["throughput"]["val"],
        },
        "memory_usage": training_metrics["memory_usage"],
        "model_info": training_metrics["model_info"],
        "initial_memory": training_metrics["initial_memory"],
        "final_memory": training_metrics["final_memory"],
    }

    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    if verbose:
        print(f"\nHistórico de treinamento salvo em: {history_path}")
        print(f"Métricas computacionais salvas em: {metrics_path}")

    # Usar todos os dados de treino para extração de features (após treinamento)
    source_dataloader = DataLoader(
        FeatureDataset(target_train_features, target_train_labels),
        batch_size=batch_size,
    )

    train_x_recon, train_labels = extract_features(model, source_dataloader)

    plot_pca(
        train_x_recon,
        train_labels,
        output_path=os.path.join(
            output_path, "plots", f"{target_name}_train_reconstructed.png"
        ),
        centroids=source_centroids,
        pca=pca,
        title=f"{target_name} Train Reconstructed",
    )

    test_dataloader = DataLoader(
        FeatureDataset(target_test_features, target_test_labels),
        batch_size=batch_size,
    )

    test_x_recon, test_labels = extract_features(model, test_dataloader)

    plot_pca(
        test_x_recon,
        test_labels,
        output_path=os.path.join(
            output_path, "plots", f"{target_name}_test_reconstructed.png"
        ),
        centroids=source_centroids,
        pca=pca,
        title=f"{target_name} Test Reconstructed",
    )

    mapped_dataloader = DataLoader(
        FeatureDataset(test_x_recon, test_labels),
        batch_size=batch_size,
    )

    cm = ClassificationModel(num_classes=2, backbone=pretrained_model)
    cm.load_weights(output_path)
    classification_module = cm.model.classifier
    classification_module.to(device)

    results = evaluate_model(classification_module, mapped_dataloader, num_classes=2)
    report = classification_report(results, class_names=["NORMAL", "PNEUMONIA"])

    confusion_matrix = results["confusion_matrix"]
    for row in confusion_matrix:
        print(" ".join([str(cell) for cell in row]))

    json_filename = f"{source_name}_autoencoder_{target_name}_test_mapped_results.json"
    json_path = os.path.join(output_path, json_filename)

    all_results = {
        "confusion_matrix": results["confusion_matrix"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "accuracy": results["accuracy"],
        "loss": results["loss"],
        "classification_report": report,
        "training_history": training_history,
        "computational_metrics": serializable_metrics,
        "training_config": {
            "source_name": source_name,
            "target_name": target_name,
            "pretrained_model": pretrained_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "device": str(device),
        },
    }

    os.makedirs(output_path, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nResultados completos salvos em: {json_path}")
