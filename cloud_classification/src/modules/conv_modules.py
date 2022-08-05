import torch
import torch.nn as nn
from torchvision import models

class CNNExtractor(nn.Module):
    def __init__(self, feature_extraction=False):
        super().__init__()
        
        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )
        
        if feature_extraction:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        return self.cnn(x).view(-1,2048)