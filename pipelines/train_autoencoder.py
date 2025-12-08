import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
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


def plot_pca(features, labels, output_path, file_name, title):
    plt.figure(figsize=(10, 5))
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(output_path, f"{file_name}.png"))
    plt.close()


def train_autoencoder(model, source: DataLoader, target: DataLoader, epochs=100):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for (source_inputs, _), (target_inputs, _) in zip(source, target):
            source_inputs = source_inputs.to(device)
            target_inputs = target_inputs.to(device)

            x_recon, _ = model(source_inputs)
            loss = mse_loss(x_recon, target_inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


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


def extract_features(model, dataloader):
    all_features = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            x_recon, _ = model(inputs)
        all_features.append(x_recon.cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def run(output_path, k=1, epochs=100):
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
    model = train_autoencoder(model, source_dataloader, target_dataloader, epochs)

    features, labels = extract_features(model, source_dataloader)
    plot_pca(
        features,
        labels,
        output_path,
        "source_features",
        "Mapped Target Train Features",
    )

    target_features = np.load(
        os.path.join(features_path, "cross_domain_test_features.npy")
    )
    target_labels = np.load(os.path.join(features_path, "cross_domain_test_labels.npy"))
    source_dataloader = DataLoader(
        AEDataset(target_features, target_labels),
        batch_size=32,
    )

    features, labels = extract_features(model, source_dataloader)
    plot_pca(
        features,
        labels,
        output_path,
        "target_features",
        "Mapped Target Test Features",
    )
