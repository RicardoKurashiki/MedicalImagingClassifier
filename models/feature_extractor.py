import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone="densenet", trainable_layers=None):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.trainable_layers = trainable_layers
        self._build_extractor()
        self._unfreeze_layers()

    def _build_extractor(self):
        if self.backbone == "densenet":
            self.weights = DenseNet121_Weights.IMAGENET1K_V1
            base_model = densenet121(weights=self.weights)
            num_ftrs = base_model.classifier.in_features

            self.backbone_model = base_model.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.feature_layers = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
            )

        elif self.backbone == "resnet":
            self.weights = ResNet18_Weights.IMAGENET1K_V1
            base_model = resnet18(weights=self.weights)
            num_ftrs = base_model.fc.in_features

            self.backbone_model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_layers = nn.Linear(num_ftrs, 256)

        elif self.backbone == "mobilenet":
            self.weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            base_model = mobilenet_v3_small(weights=self.weights)
            num_ftrs = base_model.classifier[0].in_features

            self.backbone_model = base_model.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.feature_layers = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
            )

        elif self.backbone == "efficientnet":
            self.weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            base_model = efficientnet_v2_s(weights=self.weights)
            num_ftrs = base_model.classifier[1].in_features

            self.backbone_model = base_model.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.feature_layers = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
            )

    def forward(self, x):
        x = self.backbone_model(x)

        if hasattr(self, "global_pool"):
            x = self.global_pool(x)

        x = x.view(x.size(0), -1)
        x = self.feature_layers(x)

        return x

    def _unfreeze_layers(self):
        for p in self.backbone_model.parameters():
            p.requires_grad = False

        if self.trainable_layers is None or self.trainable_layers <= 0:
            return

        indexed = [
            (idx, name, module)
            for idx, (name, module) in enumerate(self.named_modules())
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
