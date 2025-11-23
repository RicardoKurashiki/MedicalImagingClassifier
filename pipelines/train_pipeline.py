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
from tqdm import tqdm
from torchinfo import summary

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

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples_processed = 0

                pbar = tqdm(
                    dataloaders[phase],
                    desc=f"{phase.capitalize():5s}",
                    unit="batch",
                    leave=False,
                )

                for inputs, labels in pbar:
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

                    total_samples_processed += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data).double() / inputs.size(
                        0
                    )
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                epoch_loss = running_loss / total_samples_processed
                epoch_acc = running_corrects.double() / total_samples_processed

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc.item())
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc.item())

                print(
                    f"{phase.capitalize():5s} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
                )

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    print(f"Best {phase} acc: {best_acc:.4f}")

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val acc: {best_acc:4f}")

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, history


def get_model_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable: {100 * trainable_params / total_params:.2f}%")

    print("\nTrainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")


def train_densenet(
    dataset_path, layers, kfolds, batch_size, epochs, output_path="./results/"
):
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    if layers > 0:
        print(f"Unfreezing {layers} layers")
        features = list(model.features.children())
        layers_to_unfreeze = features[-layers:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    transform = weights.transforms()
    val_transform = weights.transforms()

    data = load_data(
        os.path.join(dataset_path, "train/"),
        n_splits=kfolds,
        transform=transform,
        val_transform=val_transform,
    )

    for fold in data.keys():
        if kfolds > 0:
            print(f"Training fold {fold + 1} of {kfolds}")
        else:
            print("Training on all data")

        fold_data = data[fold]

        train_dataset = fold_data["X_train"]
        train_labels = fold_data["y_train"]
        val_dataset = fold_data["X_val"]
        val_labels = fold_data["y_val"]

        train_sampler = BatchSampler(train_labels, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
        )

        dataloaders = {
            "train": train_loader,
            "val": val_loader,
        }

        dataset_sizes = {
            "train": len(train_labels),
            "val": len(val_labels),
        }

        summary(model, input_size=(3, 224, 224))
        get_model_params(model)

        model, history = train_model(
            model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer,
            epochs,
        )

        print("Salvando modelo...")
        os.makedirs(output_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_path, f"fold_{fold}.pt"))


def run(
    dataset_path,
    pretrained_model,
    trainable_layers,
    kfolds,
    batch_size,
    epochs,
):
    output_path = os.path.join(
        "results",
        pretrained_model,
        f"layers_{trainable_layers}",
        f"kfolds_{kfolds}",
        f"batch_size_{batch_size}",
        f"epochs_{epochs}/",
    )
    if pretrained_model == "densenet":
        train_densenet(
            dataset_path,
            trainable_layers,
            kfolds,
            batch_size,
            epochs,
            output_path=output_path,
        )
        return
