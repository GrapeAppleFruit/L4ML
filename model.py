import torch.nn as nn
from torchvision import models

def create_model(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model