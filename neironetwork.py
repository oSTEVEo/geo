import numpy as np
from matplotlib import pylab as plt
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch import nn

device = torch.device("cuda")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # L1 ImgIn shape=(?, 128 , 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(device)

        # L2 ImgIn shape=(?, 14, 14, 32)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))

        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob))

        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)  # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class AwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (32, 32, 3)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1).to(device)
        self.pool1 = nn.MaxPool2d((2, 2)).to(device)
        self.relu1 = nn.ReLU().to(device)

        # (16, 16, 64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1).to(device)
        self.pool2 = nn.MaxPool2d((2, 2)).to(device)
        self.relu2 = nn.ReLU().to(device)

        # (8, 8, 128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1).to(device)
        self.pool3 = nn.MaxPool2d((2, 2)).to(device)
        self.relu3 = nn.ReLU().to(device)

        # (4, 4, 256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1).to(device)
        self.pool4 = nn.MaxPool2d((2, 2)).to(device)
        self.relu4 = nn.ReLU().to(device)

        # (2, 2, 512)
        self.linear = nn.Linear(2048, 10).to(device)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = self.relu4(self.pool4(self.conv4(x)))
        print(x.shape)
        return self.linear(x.reshape(batch_size, -1))


class MyModel(nn.Module):
    # stride - размер шага
    #
    def __init__(self):
        super().__init__()

        # (150, 150, 1) -> (хз, хз, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to(device)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(device)

        self.layer4 = nn.Linear(18496, 100, bias=True).to(device)
        self.layer5 = nn.Sequential(
            nn.Linear(100, 5, bias=True)
        ).to(device)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
