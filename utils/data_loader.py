#!/usr/bin/env python3

import os
import torch
import pandas as pd
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
    def __init__(self):
       pass 

def main():
    path = "../../../datasets/CXR8/"
    
    full_dataset = CustomDataset(path)
    labels = full_dataset.img_labels
    
    kf = StratifiedKFold(n_splits=5)
    print(labels.value_counts())

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset.df, y=labels)):
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_labels = labels[train_idx]

        print(train_labels.value_counts())

if __name__ == "__main__":
    main()
