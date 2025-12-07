#!/usr/bin/env python3

import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from models import ClassificationModel
from utils import load_data

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def load_model(model_path, pretrained_model, n_classes):
    classification_model = ClassificationModel(
        num_classes=n_classes,
        backbone=pretrained_model,
    )
    classification_model.load_weights(model_path)

    model = classification_model.model.extractor
    model.to(device)
    model.eval()

    return model


def run(
    model_path,
    dataset_path,
    pretrained_model,
    batch_size=32,
    prefix="",
):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=transform,
    )
    test_data = load_data(
        os.path.join(dataset_path, "test/"),
        transform=transform,
        val_transform=transform,
        training=False,
    )

    train_dataset = train_data["train"]
    val_dataset = train_data["val"]
    test_dataset = test_data["test"]

    n_classes = train_dataset.n_classes

    extraction_model = load_model(model_path, pretrained_model, n_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    output_path = os.path.join(model_path, "features/")
    os.makedirs(output_path, exist_ok=True)

    for phase in dataloaders:
        all_features = []
        all_labels = []

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = extraction_model(inputs)
            all_features.append(features.cpu().detach().numpy())
            all_labels.append(labels.cpu().numpy())

        # Concatena todos os batches e salva uma única vez
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        np.save(
            os.path.join(output_path, f"{prefix}{phase}_features.npy"),
            all_features,
        )
        np.save(
            os.path.join(output_path, f"{prefix}{phase}_labels.npy"),
            all_labels,
        )

    return output_path
