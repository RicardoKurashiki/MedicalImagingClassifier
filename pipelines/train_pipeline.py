#!/usr/bin/env python3

import os
from utils import BatchSampler, load_data

def train_resnet(data):
    print("Train ResNet")

def train_densenet(data, batch_size, layers):
    print("Train DenseNet")
    for fold in data.keys():
        fold_data = data[fold]
        print("Training Fold:", fold)
        sampler = BatchSampler(fold_data['y_train'], batch_size=batch_size)

def run(dataset_path, pretrained_model, trainable_layers, kfolds, batch_size):
    data = load_data(dataset_path, n_splits=kfolds)
    
    if (pretrained_model == "densenet"):
        train_densenet(data, batch_size, trainable_layers)
        return
    
    if (pretrained_model == "resnet"):
        train_resnet(data)
        return
