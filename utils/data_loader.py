#!/usr/bin/env python3

import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision.io import decode_image
from torch.utils.data import DataLoader, Dataset, Subset, Sampler

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.df = self.gen_dataframe(img_dir)
        self.img_labels = self.df['idx']
        self.img_dir = self.df['path']
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir.iloc[idx]
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def gen_dataframe(self, path):
        map_result = { "path": [], "name": [], "class": [], "idx": [] }
        class_names = os.listdir(path)
        for name in class_names:
            class_path = os.path.join(path, f"{name}/")
            for img in os.listdir(class_path):
                if not img.endswith(".png"):
                    continue
                map_result['path'].append(os.path.join(class_path, img))
                map_result['name'].append(img)
                map_result['class'].append(name)
                map_result['idx'].append(0 if name == "NORMAL" else 1)
        dataframe = pd.DataFrame(data=map_result)
        return dataframe
            
class CustomSampler(Sampler):
    def __init__(self, labels, batch_size=32):
        self.labels = labels
        self.classes = np.unique(self.labels)
        self.N = len(self.classes)
        self.m_per_class = batch_size // self.N
        
        self.S = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        self.C = {c: len(self.S[c]) for c in self.classes}
        self.c_max = max(self.C.values())
        self.K = self.c_max // self.m_per_class
        
    def __len__(self):
        return self.K

    def __iter__(self):
        S_work = {c: list(self.S[c]) for c in self.classes}

        for c in self.classes:
            np.random.shuffle(S_work[c])

        for _ in range(self.K):
            batch = []
            for c in self.classes:
                if len(S_work[c]) < self.m_per_class:
                    S_work[c] = list(self.S[c])
                    np.random.shuffle(S_work[c])
                chosen = S_work[c][:self.m_per_class]
                S_work[c] = S_work[c][self.m_per_class:]
                batch.extend(chosen)
            yield batch

def load_data(path, n_splits=5):
    full_dataset = CustomDataset(path)
    labels = full_dataset.img_labels
    
    kf = StratifiedKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset.df, y=labels)):
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        return train_dataset, train_labels, val_dataset, val_labels

if __name__ == "__main__":
    path = "../../../datasets/CXR8/"
    load_data(path, n_splits=5)







    
