import torch
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    """We need to go deeper"""

    def __init__(self, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 1. Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # 2. Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # 2. Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # 2. Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, X):
        bn1 = F.relu(self.b1_1(X))
        bn2 = F.relu(self.b2_1(F.relu(self.b2_2(X))))
        bn3 = F.relu(self.b3_1(F.relu(self.b3_2(X))))
        bn4 = F.relu(self.b4_1(F.relu(self.b4_2(X))))
        return torch.cat((bn1, bn2, bn3, bn4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.net = nn.Sequential(self.block_1(),
                                 self.block_2(),
                                 self.block_3(),
                                 self.block_4())

    def block_1(self):
        """
        7x7 Conv2d ->> 3x3 MaxPool2d -> 1x1 Conv2d -> 3x3 Conv2d -> 3x3 MaxPool2d
        """
        b1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(92, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return b1

    def block_2(self):
        """
        2 x Inception Block -> MaxPool2d
        """
        b2 = nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        return b2


    def block_3(self):
        """
        5 x Inception Block -> MaxPool2d
        """
        b3 = nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return b3

    def block_4(self):
        """
        2 x Inception Block -> AvgPool -> FC
        """
        b4 = nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10),
        )
        return b4

    def forward(self, X):
        X = self.net(X)
        return X

    def layer_summary(self, X_shape):
        X = torch.rand((1, 1, 96, 96))
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "Output shape:", X.shape)




