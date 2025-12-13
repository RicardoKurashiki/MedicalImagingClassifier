import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def run(
    features,
    labels,
    output_path="./results/",
    centroids=None,
    pca=None,
    title="PCA",
):
    plt.figure(figsize=(10, 5))
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(features)
    features_2d = pca.transform(features)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis")
    if centroids is not None:
        for i, label in enumerate(centroids):
            centroids_2d = pca.transform(centroids[label])
            plt.scatter(
                centroids_2d[:, 0],
                centroids_2d[:, 1],
                c="red",
                marker="X",
                s=200,
                label=f"Centroid {label}",
            )
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

    return pca
