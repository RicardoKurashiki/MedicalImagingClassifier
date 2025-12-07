import torch.nn as nn


class FeatureClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super(FeatureClassifier, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        return self.classifier(x)
