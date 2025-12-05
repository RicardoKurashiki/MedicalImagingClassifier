import os

import torch
import torchvision

from torchvision.utils import save_image
from PIL import Image


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def run(
    image_paths,
    transform,
    output_folder,
    num_samples=10,
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    os.makedirs(output_folder, exist_ok=True)


    for i, img_path in enumerate(image_paths[:num_samples]):
        img = Image.open(img_path).convert("RGB")
        augmented = transform(img)

        vis_tensor = denormalize(augmented.clone(), mean, std)

        class_name = img_path.split("/")[-2]

        save_path = os.path.join(output_folder, f"{class_name}_{i}.png")
        save_image(vis_tensor, save_path)
