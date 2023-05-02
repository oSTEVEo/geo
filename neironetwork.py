import numpy as np
from matplotlib import pylab as plt
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch import nn

device = torch.device("cuda")

class MyModel(nn.Module):
    # Epoch: 25, Train acc: 0.99422, Test acc: 0.74345
    def __init__(self):
        super().__init__()
        keep_prob = 0.25
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(keep_prob)
        ).to(device)

        self.layer3 = nn.Sequential(
            nn.Linear(87616, 1000, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(keep_prob)
        ).to(device)

        self.layer4 = nn.Sequential(
            nn.Linear(1000, 100, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(keep_prob)
        ).to(device)
        
        self.layer5 = nn.Sequential(
            nn.Linear(100, 5, bias=True),
        ).to(device)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
