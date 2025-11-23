import subprocess


def main():
    n_jobs = 2
    configs = []

    models = ["densenet"]
    layers = [None, 1, 2, 3, 4, 5]
    folds = [None, 3, 5, 7, 10]
    batch_sizes = [16, 32, 64]
    epochs = [50, 100, 150, 200]
    dataset = ["CXR8", "chest_xray"]
    cross = ["chest_xray", "CXR8"]


if __name__ == "__main__":
    main()
