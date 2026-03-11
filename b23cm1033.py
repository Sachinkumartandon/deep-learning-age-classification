import torch.nn as nn
from torchvision import models

class _Net(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone   = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 2),
        )
    def extract_features(self, x):
        return self.backbone(x).flatten(1)
    def forward(self, x):
        return self.classifier(self.extract_features(x))

def build_model(num_classes=2, dropout=0.3):
    return _Net(dropout=dropout)