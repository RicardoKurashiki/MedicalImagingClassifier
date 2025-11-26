#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

from custom_dataset import CustomDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from custom_sampler import CustomSampler
from torchvision import transforms


def gen_dataframe(root_dir):
    if not os.path.isdir(root_dir):
        return
    map_result = {"path": [], "label": []}
    class_names = os.listdir(root_dir)
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path) or class_name.startswith("."):
            continue
        for img in os.listdir(class_path):
            map_result["path"].append(os.path.join(class_path, img))
            map_result["label"].append(class_name)
    return pd.DataFrame(map_result)


def load_data(path, transform=None, val_transform=None, training=True):
    df = gen_dataframe(path)
    if not training:
        test_dataset = CustomDataset(df, transform=transform)
        print(f"Found {len(test_dataset)} test samples")
        return {
            "test": test_dataset,
        }
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])
        train_dataset = CustomDataset(train_df, transform=transform)
        val_dataset = CustomDataset(val_df, transform=val_transform)
        print(f"Found {len(train_dataset)} training samples")
        print(f"Found {len(val_dataset)} validation samples")
        return {
            "train": train_dataset,
            "val": val_dataset,
        }


if __name__ == "__main__":
    path = "../../../datasets/CXR8/train/"
    data = load_data(
        path,
        transform=transforms.ToTensor(),
        val_transform=transforms.ToTensor(),
    )
    train_dataset = data["train"]
    val_dataset = data["val"]
