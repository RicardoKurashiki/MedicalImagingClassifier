import torch

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from sklearn.utils.class_weight import compute_class_weight

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.img_paths = self.dataframe["path"].values
        self.labels = self.dataframe["label"].values
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()

        self.unique_labels = sorted(self.dataframe["label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

        class_weight = compute_class_weight(
            "balanced",
            classes=self.unique_labels,
            y=self.labels,
        )

        self.class_weight = torch.tensor(
            class_weight,
            dtype=torch.float32,
            device=device,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        label = self.label_to_idx[label]
        image = self.transform(image)
        return image, label
