#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk

from sklearn.decomposition import PCA
from .plot_pca import run as plot_pca


def plot_training_history(training_history, training_config, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(training_history["train_loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Val Loss")

    best_epoch = np.argmin(training_history["val_loss"])
    plt.axvline(x=best_epoch, color="red", linestyle="--")
    plt.text(
        best_epoch, training_history["val_loss"][best_epoch], "Best Epoch", color="red"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.savefig(os.path.join(output_path, "training_loss_history.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(training_history["train_acc"], label="Train Accuracy")
    plt.plot(training_history["val_acc"], label="Val Accuracy")

    best_epoch = np.argmin(training_history["val_loss"])
    plt.axvline(x=best_epoch, color="red", linestyle="--")
    plt.text(
        best_epoch, training_history["val_loss"][best_epoch], "Best Epoch", color="red"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy History")
    plt.legend()
    plt.savefig(os.path.join(output_path, "training_accuracy_history.png"))
    plt.close()


def plot_computational_metrics(computational_metrics, training_config, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(
        [metric["epoch"] for metric in computational_metrics["memory_usage"]],
        [metric["cpu_memory_mb"] for metric in computational_metrics["memory_usage"]],
        label="CPU Memory Usage",
    )

    # Média de cada métrica
    cpu_memory_mb_mean = np.mean(
        [
            memory_usage["cpu_memory_mb"]
            for memory_usage in computational_metrics["memory_usage"]
        ]
    )

    plt.axhline(y=cpu_memory_mb_mean, color="red", linestyle="--")
    plt.text(
        0,
        cpu_memory_mb_mean,
        f"CPU Memory Usage Mean: {cpu_memory_mb_mean:.2f} MB",
        color="red",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage History")
    plt.legend()
    plt.savefig(os.path.join(output_path, "computational_metrics.png"))
    plt.close()


def plot_confusion_matrix(confusion_matrix, title, output_path):
    plt.figure(figsize=(10, 5))
    sns.heatmap(np.array(confusion_matrix), cmap="Blues", annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path)
    plt.close()


def run(results_path, dataset_name, generate_pca=False):
    output_path = os.path.join(results_path, "plots/")
    os.makedirs(output_path, exist_ok=True)

    metrics_path = os.path.join(results_path, "model_metrics.json")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    plot_training_history(
        metrics["training_history"],
        metrics["training_config"],
        output_path,
    )

    plot_computational_metrics(
        metrics["computational_metrics"],
        metrics["training_config"],
        output_path,
    )

    dataset_results_path = os.path.join(
        results_path,
        f"{dataset_name}_test_results.json",
    )
    with open(dataset_results_path, "r") as f:
        dataset_results = json.load(f)

    plot_confusion_matrix(
        dataset_results["confusion_matrix"],
        title=f"{dataset_name} Test Results",
        output_path=os.path.join(output_path, f"{dataset_name}_test_results.png"),
    )

    for phase in ["train", "val", "test"]:
        features_path = os.path.join(
            results_path, "features", f"{dataset_name}_{phase}_features.npy"
        )
        labels_path = os.path.join(
            results_path, "features", f"{dataset_name}_{phase}_labels.npy"
        )
        features = np.load(features_path)
        labels = np.load(labels_path)

        if generate_pca:
            pca = plot_pca(
                features,
                labels,
                output_path=os.path.join(
                    output_path, f"{dataset_name}_{phase}_pca.png"
                ),
                centroids=None,
                pca=None,
                title=f"{dataset_name} - {phase} Features",
            )

            pca_path = os.path.join(results_path, "pca/")
            os.makedirs(pca_path, exist_ok=True)
            pk.dump(
                pca,
                open(
                    os.path.join(pca_path, "pca_components.pkl"),
                    "wb",
                ),
            )
        else:
            pca_path = os.path.join(results_path, "pca", "pca_components.pkl")
            pca = pk.load(open(pca_path, "rb"))
            plot_pca(
                features,
                labels,
                output_path=os.path.join(
                    output_path, f"{dataset_name}_{phase}_pca.png"
                ),
                centroids=None,
                pca=pca,
                title=f"{dataset_name} - {phase} Features",
            )
