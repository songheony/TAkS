import torch.nn as nn
from torchvision import models


def resnet50(grayscale=False, num_classes=100, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
