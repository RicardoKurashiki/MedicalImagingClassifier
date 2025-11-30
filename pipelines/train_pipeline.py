#!/usr/bin/env python3

import os
import torch
import json
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from loss import FocalLoss
from utils import BatchSampler, load_data, train_model
from models import ClassificationModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

EARLY_STOPPING_PATIENCE = 10


def train_pipeline(
    dataset_path,
    backbone,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
    loss="cross_entropy",
    sampler="balanced",
    verbose=False,
):
    n_classes = 2

    classification_model = ClassificationModel(
        num_classes=n_classes,
        backbone=backbone,
        trainable_layers=layers,
    )

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=classification_model.transform,
        val_transform=classification_model.val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    if verbose:
        classification_model.summary()

    model = classification_model.model

    if loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss == "focal_loss":
        criterion = FocalLoss(alpha=None, gamma=2.0)
    else:
        raise ValueError(f"Loss function {loss} not supported")

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )

    if sampler == "weighted":
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.class_weight,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )
    elif sampler == "balanced":
        train_sampler = BatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        raise ValueError(f"Sampler {sampler} not supported")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    model, history, metrics = train_model(
        model,
        backbone,
        dataloaders,
        criterion,
        optimizer,
        epochs,
        verbose=verbose,
    )

    if verbose:
        print("Salvando pesos do modelo...")

    os.makedirs(output_path, exist_ok=True)
    classification_model.save_weights(output_path)

    if verbose:
        print("Salvando métricas...")

    metrics_file = os.path.join(output_path, "model_metrics.json")
    all_metrics = {
        "training_history": history,
        "computational_metrics": metrics,
        "training_config": {
            "model": backbone,
            "layers": layers,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
            "n_classes": n_classes,
        },
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

        if verbose:
            print(f"Métricas salvas em {metrics_file}")


def run(
    dataset_path,
    pretrained_model,
    trainable_layers,
    batch_size,
    dataset,
    epochs,
    loss,
    sampler,
    verbose,
):
    output_path = os.path.join(
        "results",
        pretrained_model,
        dataset,
        f"layers_{trainable_layers}",
        f"batch_size_{batch_size}",
        f"loss_{loss}",
        f"sampler_{sampler}",
        f"epochs_{epochs}/",
    )

    train_pipeline(
        dataset_path,
        pretrained_model,
        trainable_layers,
        batch_size,
        epochs,
        output_path,
        loss,
        sampler,
        verbose,
    )

    return output_path
