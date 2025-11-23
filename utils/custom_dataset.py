from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.img_paths = self.dataframe["path"].values
        self.labels = self.dataframe["label"].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    # def __getitem__(self, idx):
    #     img_path = str(self.img_paths[idx])
    #     image = decode_image(img_path)
    #     label = self.labels[idx]
    #     if self.transform:
    #         image = self.transform(image)
    #     return image, label

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
