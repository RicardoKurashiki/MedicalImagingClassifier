from torch.utils.data import Dataset

import torch

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
