#!/usr/bin/env python3

import os
import torch
import json
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import BatchSampler, load_data, train_model, check_augmentation
from models import ClassificationModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

EARLY_STOPPING_PATIENCE = 25


def train_pipeline(
    dataset_path,
    backbone,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
    verbose=False,
):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=(-10, 10),
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                shear=2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    n_classes = train_dataset.n_classes

    classification_model = ClassificationModel(
        num_classes=n_classes,
        backbone=backbone,
        trainable_layers=layers,
    )

    if verbose:
        classification_model.summary()

    model = classification_model.model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5,
    )

    for batch in BatchSampler(train_dataset, batch_size):
        check_augmentation(
            train_dataset.dataframe.iloc[batch]["path"].values,
            transform,
            os.path.join(output_path, "augmented_samples/"),
            num_samples=batch_size,
        )
        break

    train_sampler = BatchSampler(train_dataset, batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

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
        scheduler,
        num_epochs=epochs,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
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
    verbose,
):
    output_path = os.path.join(
        "results",
        pretrained_model,
        dataset,
        f"layers_{trainable_layers}",
        f"batch_size_{batch_size}",
        f"epochs_{epochs}/",
    )

    train_pipeline(
        dataset_path,
        pretrained_model,
        trainable_layers,
        batch_size,
        epochs,
        output_path,
        verbose,
    )

    return output_path
