#!/usr/bin/env python3

import os
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from utils import BatchSampler, load_data, train_model

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

EARLY_STOPPING_PATIENCE = 10


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


def train_densenet(
    dataset_path,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
):
    weights = DenseNet121_Weights.IMAGENET1K_V1
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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

    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weight)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.0001,
    )

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
        "densenet",
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


def train_resnet(
    dataset_path,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
):
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    n_classes = len(np.unique(train_dataset.labels))

    model = resnet50(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes),
    )

    unfreeze_layers(model, layers)
    summary(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weight)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.0001,
    )

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
        "resnet",
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
            "model": "resnet",
            "layers": layers,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
        },
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Métricas salvas em {metrics_file}")


def train_mobilenet(
    dataset_path,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
):
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    n_classes = len(np.unique(train_dataset.labels))

    model = mobilenet_v3_large(weights=weights)

    print(model)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(0.2),
    #     nn.Linear(512, n_classes),
    # )

    # unfreeze_layers(model, layers)
    # summary(model)

    # model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weight)
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=0.001,
    #     weight_decay=0.0001,
    # )

    # train_sampler = BatchSampler(train_dataset, batch_size)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=train_sampler,
    #     pin_memory=True,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )

    # dataloaders = {
    #     "train": train_loader,
    #     "val": val_loader,
    # }

    # model, history, metrics = train_model(
    #     model,
    #     "resnet",
    #     dataloaders,
    #     criterion,
    #     optimizer,
    #     epochs,
    # )

    # print("Salvando modelo...")
    # os.makedirs(output_path, exist_ok=True)
    # torch.save(model, os.path.join(output_path, "model.pt"))

    # print("Salvando métricas...")
    # metrics_file = os.path.join(output_path, "model_metrics.json")
    # all_metrics = {
    #     "training_history": history,
    #     "computational_metrics": metrics,
    #     "training_config": {
    #         "model": "resnet",
    #         "layers": layers,
    #         "batch_size": batch_size,
    #         "epochs": epochs,
    #         "device": str(device),
    #     },
    # }

    # with open(metrics_file, "w") as f:
    #     json.dump(all_metrics, f, indent=2)

    # print(f"Métricas salvas em {metrics_file}")


def train_efficientnet(
    dataset_path,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    n_classes = len(np.unique(train_dataset.labels))

    model = efficientnet_v2_s(weights=weights)

    print(model)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(0.2),
    #     nn.Linear(512, n_classes),
    # )

    # unfreeze_layers(model, layers)
    # summary(model)

    # model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weight)
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=0.001,
    #     weight_decay=0.0001,
    # )

    # train_sampler = BatchSampler(train_dataset, batch_size)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=train_sampler,
    #     pin_memory=True,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )

    # dataloaders = {
    #     "train": train_loader,
    #     "val": val_loader,
    # }

    # model, history, metrics = train_model(
    #     model,
    #     "resnet",
    #     dataloaders,
    #     criterion,
    #     optimizer,
    #     epochs,
    # )

    # print("Salvando modelo...")
    # os.makedirs(output_path, exist_ok=True)
    # torch.save(model, os.path.join(output_path, "model.pt"))

    # print("Salvando métricas...")
    # metrics_file = os.path.join(output_path, "model_metrics.json")
    # all_metrics = {
    #     "training_history": history,
    #     "computational_metrics": metrics,
    #     "training_config": {
    #         "model": "resnet",
    #         "layers": layers,
    #         "batch_size": batch_size,
    #         "epochs": epochs,
    #         "device": str(device),
    #     },
    # }

    # with open(metrics_file, "w") as f:
    #     json.dump(all_metrics, f, indent=2)

    # print(f"Métricas salvas em {metrics_file}")


def train_vit(
    dataset_path,
    layers,
    batch_size,
    epochs,
    output_path="./results/",
):
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = load_data(
        os.path.join(dataset_path, "train/"),
        transform=transform,
        val_transform=val_transform,
    )

    train_dataset = data["train"]
    val_dataset = data["val"]

    n_classes = len(np.unique(train_dataset.labels))

    model = vit_b_16(weights=weights)

    print(model)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(0.2),
    #     nn.Linear(512, n_classes),
    # )

    # unfreeze_layers(model, layers)
    # summary(model)

    # model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weight)
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=0.001,
    #     weight_decay=0.0001,
    # )

    # train_sampler = BatchSampler(train_dataset, batch_size)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=train_sampler,
    #     pin_memory=True,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     num_workers=2,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )

    # dataloaders = {
    #     "train": train_loader,
    #     "val": val_loader,
    # }

    # model, history, metrics = train_model(
    #     model,
    #     "resnet",
    #     dataloaders,
    #     criterion,
    #     optimizer,
    #     epochs,
    # )

    # print("Salvando modelo...")
    # os.makedirs(output_path, exist_ok=True)
    # torch.save(model, os.path.join(output_path, "model.pt"))

    # print("Salvando métricas...")
    # metrics_file = os.path.join(output_path, "model_metrics.json")
    # all_metrics = {
    #     "training_history": history,
    #     "computational_metrics": metrics,
    #     "training_config": {
    #         "model": "resnet",
    #         "layers": layers,
    #         "batch_size": batch_size,
    #         "epochs": epochs,
    #         "device": str(device),
    #     },
    # }

    # with open(metrics_file, "w") as f:
    #     json.dump(all_metrics, f, indent=2)

    # print(f"Métricas salvas em {metrics_file}")


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
    elif pretrained_model == "resnet":
        train_resnet(
            dataset_path,
            trainable_layers,
            batch_size,
            epochs,
            output_path=output_path,
        )
    elif pretrained_model == "mobilenet":
        train_mobilenet(
            dataset_path,
            trainable_layers,
            batch_size,
            epochs,
            output_path=output_path,
        )
    elif pretrained_model == "efficientnet":
        train_efficientnet(
            dataset_path,
            trainable_layers,
            batch_size,
            epochs,
            output_path=output_path,
        )
    elif pretrained_model == "vit":
        train_vit(
            dataset_path,
            trainable_layers,
            batch_size,
            epochs,
            output_path=output_path,
        )
    return output_path
