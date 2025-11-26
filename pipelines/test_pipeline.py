#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights


from utils import load_data

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


def load_model(model_path):
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device)
    model.eval()
    return model


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

    TN = conf_matrix[idx][:, idx].sum().item()
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
    criterion=None,
    num_classes=2,
):
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

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

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


def run(model_path, cross_dataset_path, pretrained_model, batch_size=32, prefix=""):
    model = load_model(model_path)

    print(f"Carregando dataset de teste de {cross_dataset_path}")

    if pretrained_model == "densenet":
        weights = DenseNet121_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    else:
        weights = DenseNet121_Weights.IMAGENET1K_V1
        transform = weights.transforms()

    data = load_data(
        os.path.join(cross_dataset_path, "test/"),
        transform=transform,
        training=False,
    )

    test_dataset = data["test"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Total de amostras de teste: {len(test_dataset)}")

    criterion = nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, criterion=criterion, num_classes=2)

    report = classification_report(results, class_names=["NORMAL", "PNEUMONIA"])

    print(report)
    print(results["confusion_matrix"])

    model_dir = os.path.dirname(model_path)
    cross_dataset_name = os.path.basename(os.path.normpath(cross_dataset_path))
    json_filename = f"{prefix}test_{cross_dataset_name}_results.json"
    json_path = os.path.join(model_dir, json_filename)

    all_results = {
        "confusion_matrix": results["confusion_matrix"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "accuracy": results["accuracy"],
        "loss": results["loss"],
        "classification_report": report,
        "test_config": {
            "model_path": model_path,
            "cross_dataset_path": cross_dataset_path,
            "pretrained_model": pretrained_model,
            "batch_size": batch_size,
            "device": str(device),
        },
    }

    os.makedirs(model_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados de teste salvos em {json_path}")
