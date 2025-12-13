import os
import numpy as np
import pickle as pk

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import plot_pca


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
            kmeans = KMeans(n_clusters=k, random_state=42).fit(
                features_by_labels[label]
            )
            result[label] = kmeans.cluster_centers_
    return result


def run(model_path, dataset_name, k=1):
    centroids_path = os.path.join(model_path, "centroids/")
    os.makedirs(centroids_path, exist_ok=True)

    for phase in ["train", "val", "test"]:
        features = np.load(
            os.path.join(
                model_path,
                "features",
                f"{dataset_name}_{phase}_features.npy",
            )
        )
        labels = np.load(
            os.path.join(
                model_path,
                "features",
                f"{dataset_name}_{phase}_labels.npy",
            )
        )
        centroids = get_centroids(features, labels, k)
        for label in centroids:
            np.save(
                os.path.join(
                    centroids_path,
                    f"{dataset_name}_{phase}_centroids_{label}.npy",
                ),
                centroids[label],
            )
        pca_path = os.path.join(model_path, "pca", "pca_components.pkl")
        plot_path = os.path.join(
            model_path, "plots", f"{dataset_name}_{phase}_centroids.png"
        )
        pca = pk.load(open(pca_path, "rb"))
        plot_pca(
            features,
            labels,
            output_path=plot_path,
            centroids=centroids,
            pca=pca,
            title=f"{dataset_name} - {phase} Centroids",
        )
