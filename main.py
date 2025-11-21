#!/usr/bin/env python3

import os
import argparse

from pipelines import train_pipeline

parser = argparse.ArgumentParser(prog="Medical Imaging Analysis Classifier")

parser.add_argument(
        '--model',
        choices=('resnet', 'mobilenet', 'densenet', 'efficientnet', 'vit'),
        help='Pretrained Model',
        default='densenet')

parser.add_argument(
        '--layers',
        type=int,
        choices=('5', '10', '15', '20', '25'),
        help='Trainable Layers',
        default='10')

parser.add_argument(
        '--folds',
        type=int,
        choices=('3', '5', '7', '10'),
        help="KFolds Qtd",
        default='5')

parser.add_argument(
        '--batch-size',
        type=int,
        default='32',
        help='Batch Size')

parser.add_argument(
        '--dataset',
        choices=('chest_xray', 'CXR8'),
        help='Training Dataset',
        default='CXR8')

args = parser.parse_args()

def main():
    dataset_path = os.path.join("../../datasets", args.dataset)
    train_pipeline(dataset_path, args.model, args.layers, args.folds, args.batch_size)

if __name__ == "__main__":
    main()
