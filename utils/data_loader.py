#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

from custom_dataset import CustomDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from custom_sampler import CustomSampler


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


def visualise_dataloader(dl, id_to_label=None):
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []

    for i, batch in enumerate(dl):
        idxs = batch[0][:, 0].tolist()
        classes = batch[0][:, 1]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        else:
            raise ValueError("More than two classes detected")
    return class_0_batch_counts, class_1_batch_counts, idxs_seen, total_num_images


if __name__ == "__main__":
    path = "../../../datasets/CXR8/train/"
    data = load_data(path, n_splits=5)
    for fold in data.keys():
        print(f"Fold {fold + 1}")
        print("Train:")
        print(pd.DataFrame(data=data[fold]["y_train"]).value_counts(normalize=True))
        print("Val:")
        print(pd.DataFrame(data=data[fold]["y_val"]).value_counts(normalize=True))
        print("-" * 10)
        batch_sampler = CustomSampler(data[fold]["y_train"])
        dataloader = DataLoader(
            data[fold]["X_train"],
            batch_sampler=batch_sampler,
            pin_memory=True,
            num_workers=4,
        )
        class_0, class_1, idxs_seen, total_num_images = (
            visualise_dataloader(dataloader, {0: "NORMAL", 1: "PNEUMONIA"}),
        )
        print(f"Class 0: {class_0}")
        print(f"Class 1: {class_1}")
        print(f"Total number of images: {total_num_images}")
        print(f"Idxs seen: {idxs_seen}")

    print("-" * 10)
    data = load_data(path, n_splits=None)
    print("All data:")
    print("Train:")
    print(pd.DataFrame(data=data[0]["y_train"]).value_counts(normalize=True))
    print("Val:")
    print(pd.DataFrame(data=data[0]["y_val"]).value_counts(normalize=True))
    print("-" * 10)
    batch_sampler = CustomSampler(data[0]["y_train"])
    dataloader = DataLoader(
        data[0]["X_train"],
        batch_sampler=batch_sampler,
        pin_memory=True,
        num_workers=4,
    )
    class_0, class_1, idxs_seen, total_num_images = (
        visualise_dataloader(dataloader, {0: "NORMAL", 1: "PNEUMONIA"}),
    )
    print(f"Class 0: {class_0}")
    print(f"Class 1: {class_1}")
    print(f"Total number of images: {total_num_images}")
    print(f"Idxs seen: {idxs_seen}")
