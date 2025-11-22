#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import BatchSampler, load_data
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms


def train_resnet(data):
    print("Train ResNet")

def train_densenet(data, batch_size, layers):
    print("Train DenseNet")
    
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
            param.requires_grad = False

    for param in model.parameters()[-layers:]:
        print(layers)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for fold in data.keys():
        fold_data = data[fold]
        sampler = BatchSampler(fold_data['y_train'], batch_size=batch_size)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        

def run(dataset_path, pretrained_model, trainable_layers, kfolds, batch_size):
    print(dataset_path, pretrained_model, trainable_layers, kfolds, batch_size)
    data = load_data(os.path.join(dataset_path, "train/"), n_splits=kfolds)
    
    if (pretrained_model == "densenet"):
        train_densenet(data, batch_size, trainable_layers)
        return
    
    if (pretrained_model == "resnet"):
        train_resnet(data)
        return
