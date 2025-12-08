import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans

from models import AutoEncoder

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class AEDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_autoencoder(model, source: DataLoader, target: DataLoader):
    phases = {
        "train": source,
        "test": target,
    }
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for phase in phases:
        for inputs, labels in phases[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def map_features_to_centroids(features, labels, centroids):
    mapped_features = np.zeros_like(features)
    features_by_labels = get_separated_labels(features, labels)

    for label in features_by_labels:
        if label not in centroids:
            mapped_features[np.array(labels) == label] = features_by_labels[label]
            continue

        label_indices = np.where(np.array(labels) == label)[0]
        label_features = features_by_labels[label]

        for idx, feature in enumerate(label_features):
            distances = np.linalg.norm(feature - centroids[label], axis=1)
            closest_centroid_idx = np.argmin(distances)
            mapped_features[label_indices[idx]] = centroids[label][closest_centroid_idx]

    return mapped_features


def get_separated_labels(features, labels):
    result = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        result[int(label)] = features[np.array(labels) == label]
    return result


def get_centroids(features, labels, k=1):
    result = {}
    features_by_labels = get_separated_labels(features, labels)
    for label in features_by_labels:
        if len(features_by_labels[label]) >= k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(
                features_by_labels[label]
            )
            result[label] = kmeans.cluster_centers_
    return result


def run(output_path, k=1):
    features_path = os.path.join(output_path, "features/")
    centroids_path = os.path.join(output_path, "centroids/")
    os.makedirs(centroids_path, exist_ok=True)

    source_features = np.load(
        os.path.join(features_path, "same_domain_train_features.npy")
    )
    source_labels = np.load(os.path.join(features_path, "same_domain_train_labels.npy"))

    target_features = np.load(
        os.path.join(features_path, "cross_domain_train_features.npy")
    )
    target_labels = np.load(
        os.path.join(features_path, "cross_domain_train_labels.npy")
    )

    source_centroids = get_centroids(source_features, source_labels, k)
    for label in source_centroids:
        np.save(
            os.path.join(centroids_path, f"source_centroids_{label}.npy"),
            source_centroids[label],
        )

    mapped_target_features = map_features_to_centroids(
        target_features, target_labels, source_centroids
    )

    source_dataloader = DataLoader(
        AEDataset(target_features, target_labels),
        batch_size=32,
    )
    target_dataloader = DataLoader(
        AEDataset(mapped_target_features, target_labels),
        batch_size=32,
    )

    model = AutoEncoder()
    train_autoencoder(model, source_dataloader, target_dataloader)
