import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .flatten import Flatten


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(64,32,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32,1,3,padding=1),
            Flatten(),
            nn.Linear(49, 10),
        )

    def forward(self, x):
        #print(x.shape)
        N = x.shape[0]
        x = x.view(N,1,28,28)
        return self.model(x)

class DeeperConvModel(nn.Module):
    def __init__(self):
        super(DeeperConvModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64,32,3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32,16,3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(16,1,3,padding=1),
            Flatten(),
            nn.Linear(49, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(x.shape)
        N = x.shape[0]
        x = x.view(N,1,28,28)
        return self.model(x)

class ShallowerConvModel(nn.Module):
    def __init__(self):
        super(ShallowerConvModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(64,1,3,padding=1),
            nn.LeakyReLU(0.01),
            Flatten(),
            nn.Linear(14 * 14, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(x.shape)
        N = x.shape[0]
        x = x.view(N,1,28,28)
        return self.model(x)

class CifarConvModel(nn.Module):
    def __init__(self):
        super(CifarConvModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(8,1,3,padding=1),
            Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        #print(x.shape)
        N = x.shape[0]
        x = x.view(N,3,32,32)
        return self.model(x)

class LeNet5Model(nn.Module):
    def __init__(self):
        super(LeNet5Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(6,16,5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.LeakyReLU(0.01),
            nn.Linear(120, 84),
            nn.LeakyReLU(0.01),
            nn.Linear(84, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(x.shape)
        N = x.shape[0]
        x = x.view(N,3,32,32)
        return self.model(x)


