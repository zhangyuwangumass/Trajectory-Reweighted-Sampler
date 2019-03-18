import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.01),
            nn.Linear(512,out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

