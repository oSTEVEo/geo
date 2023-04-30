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
    # Epoch: 6, Train acc: 0.46349, Test acc: 0.46277
    def __init__(self):
        super().__init__()
        keep_prob = 0.5
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(keep_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(keep_prob)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(keep_prob),
        ) 

        self.layer4 = nn.Sequential(
            nn.Linear(18496, 100, bias=True),
            nn.Dropout(keep_prob)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(100, 5, bias=True),
            nn.Dropout(keep_prob)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
