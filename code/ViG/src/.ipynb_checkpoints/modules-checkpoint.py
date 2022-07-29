import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import *
from .utils import *


class Patchifier(nn.Module):
    def __init__(self, hw, patch_size=4, hidden_channels=768):
        super().__init__()

        self.hw = hw  # img height/width

        self.patch_size = patch_size
        self.proyection = nn.Conv2d(  # No overlap
            in_channels=3,
            out_channels=hidden_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )


        self.pos_embedding = nn.Parameter(  # Init at 0
            torch.zeros(
                1,
                hidden_channels,
                hw // patch_size,
                hw // patch_size,
            )
        )

    def forward(self, x):
        x = self.proyection(x)
        x = x + self.pos_embedding
        return x


class FFN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, 1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(2*in_channels),
            nn.Conv2d(2*in_channels, in_channels, 1, bias=False),
            
        )

    def forward(self, x):
        return x + self.feed_forward(x)


class Grapher(nn.Module):
    def __init__(self, in_channels, k=16):
        super().__init__()

        self.k = k

        self.fc1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.graph_conv = MRConv2d(
            in_channels, in_channels, act="gelu", norm="batch", bias=True
        )

        self.norms = nn.ModuleList()
        for _ in range(2):
            self.norms.append(nn.BatchNorm2d(in_channels))

    def forward(self, x):

        input = x
        # 1. fc1
        x = self.fc1(x)
        x = self.norms[0](x)
        # 2. Get index based on knns
        edge_index = dense_knn_matrix(x.detach(), self.k)
        # 3. Max Relative Graph Convolution
        x = self.graph_conv(x, edge_index)
        x = F.gelu(x)
        x = self.norms[1](x)
        # 4. fc2
        x = self.fc2(x)
        return x + input


class Block(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()

        self.block = nn.Sequential(
            Grapher(in_channels, kernel_size), 
            nn.GELU(),
            nn.BatchNorm2d(in_channels), 
            FFN(in_channels)
        )

    def forward(self, x):
        return self.block(x)


def test_patch():
    patchifier = Patchifier(patch_size=4, hidden_channels=768, norm=True)

    x = torch.randn(1, 3, 224, 224)
    print(patchifier(x).shape)  # torch.Size([1, 768, 56, 56])


def test_ffn():
    ffn = FFN(in_channels=768)

    x = torch.randn(1, 768, 56, 56)
    print(ffn(x).shape)


def test_grapher():
    grapher = Grapher(in_channels=768, k=16)

    x = torch.randn(1, 768, 56, 56)
    print(grapher(x).shape)


if __name__ == "__main__":
    test_patch()
    test_ffn()
    test_grapher()
