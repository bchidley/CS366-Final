import torch
import torch.nn as nn
from torchvision import models

class ResNetCounter(nn.Module):
    def __init__(self):
        super(ResNetCounter, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.backbone(x)

