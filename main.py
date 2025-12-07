#!/usr/bin/env python3

import os
import argparse

from pipelines import train_pipeline, test_pipeline, feature_extraction_pipeline
from utils import plot_charts

parser = argparse.ArgumentParser(prog="Medical Imaging Analysis Classifier")

parser.add_argument(
    "--model",
    choices=("resnet", "mobilenet", "densenet", "efficientnet", "vit"),
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
    choices=("chest_xray", "CXR8"),
    help="Training Dataset",
    default="CXR8",
)

parser.add_argument(
    "--cross",
    choices=("chest_xray", "CXR8"),
    help="Training Dataset",
    default="chest_xray",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Training Epochs",
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

args = parser.parse_args()


def plot(output_path, dataset, cross):
    plot_charts(output_path, dataset, cross)


def extract(
    output_path, dataset_path, cross_dataset_path, pretrained_model, batch_size
):
    feature_extraction_pipeline(
        output_path,
        dataset_path,
        pretrained_model,
        batch_size,
        "same_domain_",
    )
    feature_extraction_pipeline(
        output_path,
        cross_dataset_path,
        pretrained_model,
        batch_size,
        "cross_domain_",
    )


def main():
    dataset_path = os.path.join("../../datasets", args.dataset)
    cross_dataset_path = os.path.join("../../datasets", args.cross)
    output_path = os.path.join(
        "results",
        args.model,
        args.dataset,
        f"layers_{args.layers}",
        f"batch_size_{args.batch_size}",
        f"epochs_{args.epochs}/",
    )

    if args.plot:
        plot(output_path, args.dataset, args.cross)
        return 0

    if args.extract:
        extract(
            output_path,
            dataset_path,
            cross_dataset_path,
            args.model,
            args.batch_size,
        )
        return 0

    train_pipeline(
        dataset_path,
        args.model,
        args.layers,
        args.batch_size,
        args.dataset,
        args.epochs,
        args.verbose,
        output_path,
    )

    test_pipeline(
        output_path,
        dataset_path,
        args.model,
        args.batch_size,
        "same_domain_",
        args.verbose,
    )
    test_pipeline(
        output_path,
        cross_dataset_path,
        args.model,
        args.batch_size,
        "cross_domain_",
        args.verbose,
    )

    plot(output_path, args.dataset, args.cross)
    extract(
        output_path,
        dataset_path,
        cross_dataset_path,
        args.model,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
