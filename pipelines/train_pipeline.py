#!/usr/bin/env python3

import os
import torch
import time
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils import BatchSampler, load_data
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from tempfile import TemporaryDirectory

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=100
):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        print(f"Saving best model to {best_model_params_path}")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val acc: {best_acc:4f}")

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model


def unfreeze_layers(model, n_layers):
    modules = [m for m in model.modules() if len(list(m.parameters())) > 0]
    modules_to_train = modules[-n_layers:]
    for module in modules_to_train:
        for param in module.parameters():
            param.requires_grad = True


def train_resnet(dataset_path):
    print("Train ResNet")


def train_densenet(dataset_path, layers, kfolds, batch_size, epochs):
    print("Train DenseNet")

    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    unfreeze_layers(model, layers)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    transform = weights.transforms

    data = load_data(
        os.path.join(dataset_path, "train/"),
        n_splits=kfolds,
        transform=transform,
    )

    for fold in data.keys():
        print(f"Training fold {fold + 1} of {kfolds}")

        fold_data = data[fold]

        X_train = fold_data["X_train"]
        y_train = fold_data["y_train"]
        X_val = fold_data["X_val"]
        y_val = fold_data["y_val"]

        train_sampler = BatchSampler(y_train, batch_size)
        train_loader = DataLoader(
            X_train,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            X_val,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
        )

        dataloaders = {
            "train": train_loader,
            "val": val_loader,
        }

        dataset_sizes = {
            "train": len(y_train),
            "val": len(y_val),
        }

        model = train_model(
            model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer,
            epochs,
        )


def run(dataset_path, pretrained_model, trainable_layers, kfolds, batch_size, epochs):
    if pretrained_model == "densenet":
        train_densenet(dataset_path, trainable_layers, kfolds, batch_size, epochs)
        return

    if pretrained_model == "resnet":
        train_resnet(dataset_path)
        return
