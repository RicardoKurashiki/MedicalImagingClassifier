import torch

from .complete_model import CompleteModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ClassificationModel:
    def __init__(self, num_classes, backbone="densenet", trainable_layers=None):
        self.num_classes = num_classes
        self.trainable_layers = trainable_layers
        self.model = CompleteModel(
            num_classes=num_classes,
            backbone=backbone,
            trainable_layers=trainable_layers,
        )

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
        self.model.save_weights(output_path)
        return output_path

    def load_weights(self, input_path):
        self.model.load_weights(input_path)
        print(f"Weights loaded from {input_path}")
        return input_path
