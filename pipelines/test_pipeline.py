#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models import ResNet50_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.models import ViT_B_16_Weights
from torchvision.models import resnet50
from torchvision.models import densenet121
from torchvision.models import mobilenet_v3_small
from torchvision.models import efficientnet_v2_s
from torchvision.models import vit_b_16

from utils import load_data

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def create_densenet_model(n_classes):
    """Cria um modelo DenseNet121 com a arquitetura customizada."""
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes),
    )
    return model


def create_resnet_model(n_classes):
    """Cria um modelo ResNet50 com a arquitetura customizada."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes),
    )
    return model


def create_mobilenet_model(n_classes):
    """Cria um modelo MobileNet V3 Small com a arquitetura customizada."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)

    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(512, n_classes),
    )
    return model


def create_efficientnet_model(n_classes):
    """Cria um modelo EfficientNet V2 S com a arquitetura customizada."""
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=num_ftrs, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes),
    )
    return model


def create_vit_model(n_classes):
    """Cria um modelo Vision Transformer B/16 com a arquitetura customizada."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    num_ftrs = model.heads[0].in_features
    model.heads = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, n_classes),
    )
    return model


def load_model(model_path, pretrained_model, n_classes):
    """Carrega os pesos de um modelo treinado e reconstrói a arquitetura."""
    # Criar o modelo com a arquitetura correta
    if pretrained_model == "densenet":
        model = create_densenet_model(n_classes)
    elif pretrained_model == "resnet":
        model = create_resnet_model(n_classes)
    elif pretrained_model == "mobilenet":
        model = create_mobilenet_model(n_classes)
    elif pretrained_model == "efficientnet":
        model = create_efficientnet_model(n_classes)
    elif pretrained_model == "vit":
        model = create_vit_model(n_classes)
    else:
        raise ValueError(f"Pretrained model {pretrained_model} not supported")

    # Carregar os pesos treinados
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

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


def run(
    model_path,
    cross_dataset_path,
    pretrained_model,
    batch_size=32,
    prefix="",
    verbose=False,
):
    # Carregar número de classes do JSON de métricas
    model_dir = os.path.dirname(model_path)
    metrics_file = os.path.join(model_dir, "model_metrics.json")

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(
            f"Arquivo de métricas não encontrado: {metrics_file}. "
            "Certifique-se de que o modelo foi treinado corretamente."
        )

    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    n_classes = metrics_data.get("training_config", {}).get("n_classes", 2)

    if n_classes is None:
        raise ValueError(
            "Número de classes não encontrado no arquivo de métricas. "
            "Certifique-se de que o modelo foi treinado com a versão atualizada do código."
        )

    print(f"Carregando modelo com {n_classes} classes...")
    model = load_model(model_path, pretrained_model, n_classes)

    print(f"Carregando dataset de teste de {cross_dataset_path}")

    if pretrained_model == "densenet":
        weights = DenseNet121_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    elif pretrained_model == "resnet":
        weights = ResNet50_Weights.IMAGENET1K_V2
        transform = weights.transforms()
    elif pretrained_model == "mobilenet":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    elif pretrained_model == "efficientnet":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    elif pretrained_model == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    else:
        raise ValueError(f"Pretrained model {pretrained_model} not supported")

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

    # Determinar nomes das classes do dataset
    if hasattr(test_dataset, "unique_labels"):
        class_names = list(test_dataset.unique_labels)
    else:
        # Fallback para nomes padrão baseado no número de classes
        if n_classes == 2:
            class_names = ["NORMAL", "PNEUMONIA"]
        else:
            class_names = [f"Class {i}" for i in range(n_classes)]

    criterion = nn.CrossEntropyLoss()
    results = evaluate_model(
        model, test_loader, criterion=criterion, num_classes=n_classes
    )

    report = classification_report(results, class_names=class_names)

    print(results["confusion_matrix"])
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
