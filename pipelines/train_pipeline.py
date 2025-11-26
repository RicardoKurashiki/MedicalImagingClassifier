#!/usr/bin/env python3

import os
import numpy as np
import torch
import time
import json
import psutil
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from tempfile import TemporaryDirectory
from utils import BatchSampler, load_data

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

EARLY_STOPPING_PATIENCE = 10


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


def unfreeze_layers(model, n_layers):
    for p in model.parameters():
        p.requires_grad = False

    if n_layers is None or n_layers <= 0:
        return

    indexed = [
        (idx, name, module) for idx, (name, module) in enumerate(model.named_modules())
    ]
    convs = [
        (idx, name, module)
        for idx, name, module in indexed
        if isinstance(module, nn.Conv2d)
    ]
    if len(convs) < n_layers:
        raise ValueError("O modelo não contém camadas Conv2d suficientes.")

    conv_idx = convs[-n_layers][0]
    for idx, _, module in indexed:
        if idx >= conv_idx:
            for p in module.parameters(recurse=False):
                p.requires_grad = True


def summary(model):
    print(f"{'Layer Name':50} {'Type':20} {'Trainable'}")
    print("-" * 90)

    for name, module in model.named_modules():
        if name == "":
            continue
        has_params = any(True for _ in module.parameters(recurse=False))
        if has_params:
            trainable = all(p.requires_grad for p in module.parameters(recurse=False))
            print(f"{name:50} {module.__class__.__name__:20} {trainable}")


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=100,
):
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

    # Memória inicial
    initial_memory = get_memory_usage()
    metrics["initial_memory"] = initial_memory

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        print(f"Saving best model to {best_model_params_path}")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        early_stop = False

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            epoch_start = time.time()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            for phase in ["train", "val"]:
                phase_start = time.time()
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples_processed = 0
                batch_times = []
                samples_per_second = []

                pbar = tqdm(
                    dataloaders[phase],
                    desc=f"{phase.capitalize():5s}",
                    unit="batch",
                    leave=False,
                )

                for inputs, labels in pbar:
                    batch_start = time.time()
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                    total_samples_processed += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data).double() / inputs.size(
                        0
                    )
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                phase_time = time.time() - phase_start
                epoch_loss = running_loss / total_samples_processed
                epoch_acc = running_corrects.double() / total_samples_processed

                # Salvar métricas computacionais da fase
                avg_batch_time = (
                    sum(batch_times) / len(batch_times) if batch_times else 0
                )
                avg_throughput = (
                    sum(samples_per_second) / len(samples_per_second)
                    if samples_per_second
                    else 0
                )

                metrics["batch_times"][phase].append(
                    {
                        "epoch": epoch,
                        "avg_batch_time_seconds": avg_batch_time,
                        "total_batches": len(batch_times),
                        "total_time_seconds": phase_time,
                    }
                )

                metrics["throughput"][phase].append(
                    {
                        "epoch": epoch,
                        "avg_samples_per_second": avg_throughput,
                        "total_samples": total_samples_processed,
                        "total_time_seconds": phase_time,
                    }
                )

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                print(
                    f"{phase} Time: {phase_time:.2f}s | Throughput: {avg_throughput:.2f} samples/s"
                )

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc.item())
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc.item())

                print(
                    f"{phase.capitalize():5s} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
                )

                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        print(f"Best {phase} acc: {best_acc:.4f}")

                    if epoch_loss < best_val_loss:
                        past_loss = best_val_loss
                        best_val_loss = epoch_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_params_path)
                        print(
                            f"Val loss improved from {past_loss:.4f} to {best_val_loss:.4f} - Saving best model"
                        )
                    else:
                        patience_counter += 1
                        print(
                            f"Val loss did not improve from {best_val_loss:.4f} - Patience: {patience_counter}"
                        )
                        if patience_counter >= EARLY_STOPPING_PATIENCE:
                            early_stop = True
                            print(
                                f"Early stopping triggered! Patience counter reached {patience_counter}"
                            )
                            break

            # Verificar early stopping após o loop de fases
            if early_stop:
                break

            epoch_time = time.time() - epoch_start
            metrics["epoch_times"].append({"epoch": epoch, "time_seconds": epoch_time})

            # Coletar memória após cada época
            epoch_memory = get_memory_usage()
            metrics["memory_usage"].append({"epoch": epoch, **epoch_memory})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print()

        time_elapsed = time.time() - since
        metrics["total_time_seconds"] = time_elapsed

        final_memory = get_memory_usage()
        metrics["final_memory"] = final_memory

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val acc: {best_acc:4f}")

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, history, metrics


def train_densenet(
    dataset_path,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
):
    weights = DenseNet121_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    val_transform = weights.transforms()

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    n_classes = len(np.unique(train_dataset.labels))

    model = densenet121(weights=weights)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes),
    )

    unfreeze_layers(model, layers)
    summary(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_sampler = BatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    model, history, metrics = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        epochs,
    )

    print("Salvando modelo...")
    os.makedirs(output_path, exist_ok=True)
    torch.save(model, os.path.join(output_path, "model.pt"))

    print("Salvando métricas...")
    metrics_file = os.path.join(output_path, "model_metrics.json")
    all_metrics = {
        "training_history": history,
        "computational_metrics": metrics,
        "training_config": {
            "model": "densenet",
            "layers": layers,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
        },
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Métricas salvas em {metrics_file}")


def run(
    dataset_path,
    pretrained_model,
    trainable_layers,
    batch_size,
    dataset,
    epochs,
):
    output_path = os.path.join(
        "results",
        pretrained_model,
        dataset,
        f"layers_{trainable_layers}",
        f"batch_size_{batch_size}",
        f"epochs_{epochs}/",
    )
    if pretrained_model == "densenet":
        train_densenet(
            dataset_path,
            trainable_layers,
            batch_size,
            epochs,
            output_path=output_path,
        )
    return output_path
