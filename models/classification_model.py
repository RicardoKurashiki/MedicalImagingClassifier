import os

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ClassificationModel:
    def __init__(self, num_classes, backbone="densenet", trainable_layers=None):
        self.num_classes = num_classes
        self.trainable_layers = trainable_layers
        self.model = self._build_model(backbone)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.val_transform = self.weights.transforms()

    def summary(self):
        print(f"{'Layer Name':50} {'Type':20} {'Trainable'}")
        print("-" * 90)

        for name, module in self.model.named_modules():
            if name == "":
                continue
            has_params = any(True for _ in module.parameters(recurse=False))
            if has_params:
                trainable = all(
                    p.requires_grad for p in module.parameters(recurse=False)
                )
                print(f"{name:50} {module.__class__.__name__:20} {trainable}")

    def save_weights(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(
                output_path,
                "model_weights.pt",
            ),
        )
        print(f"Weights saved to {output_path}")
        return output_path

    def load_weights(self, input_path):
        self.model.load_state_dict(
            torch.load(
                os.path.join(input_path, "model_weights.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        print(f"Weights loaded from {input_path}")
        return input_path

    def _build_model(self, backbone):
        if backbone == "densenet":
            self.weights = DenseNet121_Weights.IMAGENET1K_V1
            model = densenet121(weights=self.weights)
            self._unfreeze_layers(model)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes),
            )
            return model.to(device)
        elif backbone == "resnet":
            self.weights = ResNet50_Weights.IMAGENET1K_V2
            model = resnet50(weights=self.weights)
            self._unfreeze_layers(model)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes),
            )
            return model.to(device)
        elif backbone == "mobilenet":
            self.weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = mobilenet_v3_small(weights=self.weights)
            self._unfreeze_layers(model)
            num_ftrs = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes),
            )
            return model.to(device)
        elif backbone == "efficientnet":
            self.weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model = efficientnet_v2_s(weights=self.weights)
            self._unfreeze_layers(model)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes),
            )
            return model.to(device)

    def _unfreeze_layers(self, model):
        for p in model.parameters():
            p.requires_grad = False

        if self.trainable_layers is None or self.trainable_layers <= 0:
            return

        indexed = [
            (idx, name, module)
            for idx, (name, module) in enumerate(model.named_modules())
        ]
        convs = [
            (idx, name, module)
            for idx, name, module in indexed
            if isinstance(module, nn.Conv2d)
        ]
        if len(convs) < self.trainable_layers:
            raise ValueError("O modelo não contém camadas Conv2d suficientes.")

        conv_idx = convs[-self.trainable_layers][0]
        for idx, _, module in indexed:
            if idx >= conv_idx:
                for p in module.parameters(recurse=False):
                    p.requires_grad = True
