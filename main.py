#!/usr/bin/env python3

import os
import argparse

import torch
import numpy as np
import random

from pipelines import (
    train_pipeline,
    test_pipeline,
    feature_extraction_pipeline,
    ae_training_pipeline,
    cluster_pipeline,
)

from utils import plot_charts
from datetime import datetime

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

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
    "--ae-epochs",
    type=int,
    default=500,
    help="Autoencoder Training Epochs",
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
    "--no-data-aug",
    action="store_true",
    help="Disable Data Augmentation",
)

args = parser.parse_args()


def main():
    dataset_path = os.path.join("../../datasets", args.dataset)
    current_time = datetime.now().strftime("%d%m%Y_%H%M%S")

    all_datasets = ["chest_xray", "rsna", "CXR8"]
    cross_datasets = [args.dataset] + [d for d in all_datasets if d != args.dataset]

    output_path = os.path.join(
        "results",
        args.model,
        args.dataset,
        f"layers_{args.layers}",
        f"batch_size_{args.batch_size}",
        f"epochs_{args.epochs}/",
        f"data_aug_{not args.no_data_aug}/",
        f"seed_{SEED}/",
        f"timestamp_{current_time}/",
    )

    print(f"Output path: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    train_pipeline(
        dataset_path,
        args.model,
        args.layers,
        args.batch_size,
        args.epochs,
        output_path,
        not args.no_data_aug,
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

        plot_charts(
            output_path,
            cross_dataset,
            generate_pca=(args.dataset == cross_dataset),
        )

        cluster_pipeline(output_path, cross_dataset, k=args.k)

        ae_training_pipeline(
            output_path,
            source_name=args.dataset,
            target_name=cross_dataset,
            pretrained_model=args.model,
            epochs=args.ae_epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
