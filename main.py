#!/usr/bin/env python3

import os
import argparse


from pipelines import (
    train_pipeline,
    test_pipeline,
    feature_extraction_pipeline,
    ae_training_pipeline,
    cluster_pipeline,
)

from utils import plot_charts
from datetime import datetime

parser = argparse.ArgumentParser(prog="Medical Imaging Analysis Classifier")

parser.add_argument(
    "--model",
    choices=("resnet", "mobilenet", "densenet", "efficientnet"),
    help="Pretrained Model",
    default="densenet",
)

parser.add_argument(
    "--layers",
    type=int,
    choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    help="Trainable Layers",
    nargs="?",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch Size",
)

parser.add_argument(
    "--dataset",
    choices=("chest_xray", "rsna", "CXR8"),
    help="Training Dataset",
    default="chest_xray",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=500,
    help="Training Epochs",
)

parser.add_argument(
    "--k",
    type=int,
    default=1,
    help="Number of Centroids",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Verbose Output",
)

parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot Results",
)

parser.add_argument(
    "--extract",
    action="store_true",
    help="Extract Features",
)

parser.add_argument(
    "--align",
    action="store_true",
    help="Align Features",
)

args = parser.parse_args()


def main():
    dataset_path = os.path.join("../../datasets", args.dataset)
    current_time = datetime.now().strftime("%d%m%Y_%H%M%S")

    cross_datasets = [
        "chest_xray",
        "rsna",
        "CXR8",
    ]

    output_path = os.path.join(
        "results",
        args.model,
        args.dataset,
        f"layers_{args.layers}",
        f"batch_size_{args.batch_size}",
        f"epochs_{args.epochs}/",
        f"timestamp_{current_time}/",
    )

    os.makedirs(output_path, exist_ok=True)

    train_pipeline(
        dataset_path,
        args.model,
        args.layers,
        args.batch_size,
        args.epochs,
        output_path,
        args.verbose,
    )

    for cross_dataset in cross_datasets:
        cross_dataset_path = os.path.join("../../datasets", cross_dataset)

        test_pipeline(
            output_path,
            cross_dataset_path,
            args.model,
            args.batch_size,
            prefix=cross_dataset,
            verbose=args.verbose,
        )

        feature_extraction_pipeline(
            output_path,
            cross_dataset_path,
            args.model,
            args.batch_size,
            prefix=cross_dataset,
        )

        plot_charts(output_path, cross_dataset)

        cluster_pipeline(output_path, cross_dataset, k=args.k)


if __name__ == "__main__":
    main()
