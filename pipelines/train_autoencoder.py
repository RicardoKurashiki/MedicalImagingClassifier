import os

import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from models import AutoEncoder, ClassificationModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class AEDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def plot_pca(features, labels, output_path, file_name, title, centroids=None):
    plt.figure(figsize=(10, 5))
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.6, s=20
    )

    if centroids is not None:
        all_centroids = []
        centroid_labels = []
        for label in centroids:
            centroids_array = centroids[label]
            all_centroids.append(centroids_array)
            centroid_labels.extend([label] * len(centroids_array))

        if len(all_centroids) > 0:
            all_centroids = np.concatenate(all_centroids, axis=0)
            centroids_2d = pca.transform(all_centroids)
            plt.scatter(
                centroids_2d[:, 0],
                centroids_2d[:, 1],
                c="red",
                marker="X",
                s=200,
                edgecolors="black",
                linewidths=1.5,
                label="Centroids",
                zorder=10,
            )
            plt.legend()

    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(output_path, f"{file_name}.png"))
    plt.close()


def train_autoencoder(model, source: DataLoader, target: DataLoader, epochs=100):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        for (source_inputs, _), (target_inputs, _) in zip(source, target):
            source_inputs = source_inputs.to(device)
            target_inputs = target_inputs.to(device)

            x_recon, _ = model(source_inputs)
            loss = mse_loss(x_recon, target_inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
        print("-" * 10)
        print()

    return model


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


def get_centroids(features, labels, k=1):
    result = {}
    features_by_labels = get_separated_labels(features, labels)
    for label in features_by_labels:
        if len(features_by_labels[label]) >= k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(
                features_by_labels[label]
            )
            result[label] = kmeans.cluster_centers_
    return result


def extract_features(model, dataloader):
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


def run(output_path, k=1, epochs=50, pretrained_model="densenet"):
    features_path = os.path.join(output_path, "features/")
    centroids_path = os.path.join(output_path, "centroids/")
    os.makedirs(centroids_path, exist_ok=True)

    source_features = np.load(
        os.path.join(features_path, "same_domain_train_features.npy")
    )
    source_labels = np.load(os.path.join(features_path, "same_domain_train_labels.npy"))

    target_features = np.load(
        os.path.join(features_path, "cross_domain_train_features.npy")
    )
    target_labels = np.load(
        os.path.join(features_path, "cross_domain_train_labels.npy")
    )

    source_centroids = get_centroids(source_features, source_labels, k)
    for label in source_centroids:
        np.save(
            os.path.join(centroids_path, f"source_centroids_{label}.npy"),
            source_centroids[label],
        )

    mapped_target_features = map_features_to_centroids(
        target_features, target_labels, source_centroids
    )

    source_dataloader = DataLoader(
        AEDataset(target_features, target_labels),
        batch_size=32,
    )
    target_dataloader = DataLoader(
        AEDataset(mapped_target_features, target_labels),
        batch_size=32,
    )

    model = AutoEncoder()
    model = train_autoencoder(model, source_dataloader, target_dataloader, epochs)

    features, labels = extract_features(model, source_dataloader)
    plot_pca(
        features,
        labels,
        output_path,
        "source_features",
        "Mapped Target Train Features",
        centroids=source_centroids,
    )

    target_features = np.load(
        os.path.join(features_path, "cross_domain_test_features.npy")
    )
    target_labels = np.load(os.path.join(features_path, "cross_domain_test_labels.npy"))

    source_dataloader = DataLoader(
        AEDataset(target_features, target_labels),
        batch_size=32,
    )

    features, labels = extract_features(model, source_dataloader)

    plot_pca(
        features,
        labels,
        output_path,
        "target_features",
        "Mapped Target Test Features",
        centroids=source_centroids,
    )

    mapped_dataloader = DataLoader(
        AEDataset(features, labels),
        batch_size=32,
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

    json_filename = "train_autoencoder_results.json"
    json_path = os.path.join(output_path, json_filename)

    all_results = {
        "confusion_matrix": results["confusion_matrix"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "accuracy": results["accuracy"],
        "loss": results["loss"],
        "classification_report": report,
    }

    os.makedirs(output_path, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
