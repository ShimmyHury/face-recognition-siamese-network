import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet34


class ResNetEncoder(nn.Module):
    def __init__(self, normalize=True, feature_extractor='resnet50'):
        super().__init__()
        if feature_extractor == 'resnet18':
            self.Feature_Extractor = resnet18(weights='DEFAULT')
        elif feature_extractor == 'resnet34':
            self.Feature_Extractor = resnet34(weights='DEFAULT')
        else:
            self.Feature_Extractor = resnet50(weights='DEFAULT')
        num_filters = self.Feature_Extractor.fc.in_features
        self.Feature_Extractor.fc = nn.Sequential(
                  nn.Linear(num_filters, 512),
                  nn.LeakyReLU(),
                  nn.Linear(512, 256))
        self.Output = nn.Sequential(
                  nn.Linear(256, 128))
        self.Normalize = normalize

    def forward(self, x):
        x = self.Feature_Extractor(x)
        x = self.Output(x)
        if self.Normalize:
            x = x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))
        return x
