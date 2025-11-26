#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

from custom_dataset import CustomDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def load_data(path, n_splits=None, transform=None, val_transform=None, training=True):
    df = gen_dataframe(path)
    labels = df["label"].values

    if not training:
        test_dataset = CustomDataset(df, transform=transform)
        test_labels = df["label"].values
        return {
            0: {
                "test_df": df,
                "X_test": test_dataset,
                "y_test": test_labels,
            }
        }
    else:
        if n_splits is not None and n_splits > 0:
            kf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
            map_result = {}
            for fold, (train_idx, val_idx) in enumerate(kf.split(df, y=labels)):
                df_train = df.iloc[train_idx]
                df_val = df.iloc[val_idx]
                train_dataset = CustomDataset(df_train, transform=transform)
                val_dataset = CustomDataset(df_val, transform=val_transform)
                train_labels = df_train["label"].values
                val_labels = df_val["label"].values

                map_result[fold] = {
                    "train_df": df_train,
                    "val_df": df_val,
                    "X_train": train_dataset,
                    "y_train": train_labels,
                    "X_val": val_dataset,
                    "y_val": val_labels,
                }
            return map_result
        else:
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=labels)
            train_dataset = CustomDataset(train_df, transform=transform)
            val_dataset = CustomDataset(val_df, transform=val_transform)
            train_labels = train_df["label"].values
            val_labels = val_df["label"].values
            return {
                0: {
                    "train_df": train_df,
                    "val_df": val_df,
                    "X_train": train_dataset,
                    "y_train": train_labels,
                    "X_val": val_dataset,
                    "y_val": val_labels,
                }
            }


if __name__ == "__main__":
    path = "../../../datasets/CXR8/train/"
    data = load_data(
        path,
        n_splits=5,
        transform=transforms.ToTensor(),
        val_transform=transforms.ToTensor(),
    )
    for fold in data.keys():
        print(f"Fold {fold+1}")
        fold_data=data[fold]
        dt = fold_data['X_train']
        print(f"{len(dt)} dados")
        sampler = CustomSampler(fold_data['y_train'], batch_size=32)
        dl = DataLoader(dt, batch_sampler=sampler)
        print(f"Batches: {len(dl)}")
        for batch in dl:
            print(pd.DataFrame(data=batch[1].numpy()).value_counts())

