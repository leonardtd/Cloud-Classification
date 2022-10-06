import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class CNNExtractor(nn.Module):
    def __init__(self, feature_extraction=False):
        """
        parametro de cuantas capas descongelar (desde el final)
        """
        super().__init__()
        
        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )
        
        if feature_extraction:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        return self.cnn(x).view(-1,2048)
    
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, feature_extraction=False):
        
        super().__init__()
        
        self.cnn = CNNExtractor(feature_extraction)
        self.head = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.head(x)
        return x
    
class CloudNet(nn.Module):
    def __init__(self, out_dims, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        self.pool = nn.MaxPool2d(3, stride=2)
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, bias=False)
        self.b1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, bias=False)
        self.b2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 6 * 6, 9216)
        self.fc2 = nn.Linear(9216, 4096)
        self.fc3 = nn.Linear(4096, out_dims)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if  isinstance(m, nn.Linear):
                ### general init rule
                n = m.in_features
                y = 1.0/np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.pool(F.relu(self.b1(self.conv1(x))))
        x = self.pool(F.relu(self.b2(self.conv2(x))))
        x = F.relu(self.b3(self.conv3(x)))
        x = self.pool(F.relu(self.b4(self.conv4(x))))
        x = x.flatten(1)
        x = F.dropout(F.relu(self.fc1(x)), self.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), self.dropout, training=self.training)
        x = self.fc3(x)
        return x
    