#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.custom_dataset import CustomDataset
from utils.custom_sampler import CustomSampler


def gen_dataframe(root_dir):
    if not os.path.isdir(root_dir):
        return
    map_result = {"path": [], "label": []}

    # Extensões de imagem válidas
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
    }

    class_names = os.listdir(root_dir)
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path) or class_name.startswith("."):
            continue
        for img in os.listdir(class_path):
            # Ignorar arquivos que começam com ponto (como .DS_Store)
            if img.startswith("."):
                continue

            img_path = os.path.join(class_path, img)

            # Verificar se é um arquivo (não um diretório)
            if not os.path.isfile(img_path):
                continue

            # Verificar extensão válida (case-insensitive)
            _, ext = os.path.splitext(img)
            if ext.lower() not in valid_extensions:
                continue

            map_result["path"].append(img_path)
            map_result["label"].append(class_name)
    return pd.DataFrame(map_result)


def load_data(path, transform=None, val_transform=None, training=True):
    df = gen_dataframe(path)
    if not training:
        test_dataset = CustomDataset(df, transform=transform)
        print(f"Encontrados {len(df)} amostras no dataset de teste")
        return {
            "test": test_dataset,
        }
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])
        train_dataset = CustomDataset(train_df, transform=transform)
        val_dataset = CustomDataset(val_df, transform=val_transform)
        print(f"Encontrados {len(train_df)} amostras no dataset de treinamento")
        print(f"Encontrados {len(val_df)} amostras no dataset de validação")
        return {
            "train": train_dataset,
            "val": val_dataset,
        }


if __name__ == "__main__":
    path = "../../datasets/CXR8/train/"
    data = load_data(
        path,
        transform=transforms.ToTensor(),
        val_transform=transforms.ToTensor(),
    )
    train_dataset = data["train"]
    val_dataset = data["val"]
    sampler = CustomSampler(train_dataset, batch_size=32)

    results = []

    for batch in sampler:
        result = {"batch": batch, "idx": []}
        for index in batch:
            c = train_dataset.dataframe.iloc[index]
            result["idx"].append({"idx": index, "label": c["label"], "path": c["path"]})
        results.append(result)

    df = pd.DataFrame(data=results)
    df.to_csv(
        f"batches_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False
    )
