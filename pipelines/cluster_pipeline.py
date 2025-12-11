import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


def plot_pca(pca, features, labels, centroids, output_path):
    plt.figure(figsize=(10, 5))
    features_2d = pca.transform(features)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis")
    if centroids is not None:
        for label in centroids:
            centroids_2d = pca.transform(centroids[label])
            plt.scatter(
                centroids_2d[:, 0],
                centroids_2d[:, 1],
                c="red",
                marker="X",
                s=200,
                label=f"Centroid {label}",
            )
    plt.legend()
    plt.savefig(output_path)
    plt.close()


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
        pca_path = os.path.join(model_path, "pca", f"{dataset_name}_{phase}_pca.npy")
        plot_path = os.path.join(
            model_path, "plots", f"{dataset_name}_{phase}_centroids.png"
        )
        pca_components = np.load(pca_path)
        pca = PCA(n_components=2)
        pca.components_ = pca_components
        plot_pca(pca, features, labels, centroids, plot_path)
