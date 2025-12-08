import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def plot_pca(features, labels, output_path, file_name):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels)
    plt.savefig(os.path.join(output_path, f"{file_name}.png"))
    plt.close()


def map_features_to_centroids(features, labels, centroids):
    features_by_labels = get_separated_labels(features, labels)
    for label in features_by_labels:
        for feature in features_by_labels[label]:
            closest_centroid = np.argmin(
                np.linalg.norm(feature - centroids[label], axis=1)
            )
            features_by_labels[label][feature] = closest_centroid
    return features_by_labels


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

    plot_pca(
        mapped_target_features, target_labels, output_path, "mapped_target_features"
    )
