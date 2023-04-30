import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm
import pandas as pd
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F

import mkdataset
import neironetwork

device = torch.device("cuda")

x_train, y_train, x_test, y_test = mkdataset.init_dataset()

n_epochs = 50
eta = 0.001
criterion = torch.nn.CrossEntropyLoss()
model = neironetwork.MyModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=eta)

batch_size = 64

for epoch in range(n_epochs):
    correct_answers_train = 0
    correct_answers_test = 0

    optimizer.zero_grad()
    for batch_idx in tqdm(range(x_train.shape[0] // batch_size)):
        x_i = x_train[batch_size * batch_idx : batch_size * (batch_idx + 1)]\
            .type(torch.float64).to(device)
        y_i = y_train[batch_size * batch_idx : batch_size * (batch_idx + 1)]\
            .type(torch.float64).to(device)

        x_i = x_i.reshape(x_i.size()[0], 1, x_i.size()[1], x_i.size()[2])

        y_hat = model(x_i)

        loss = criterion(y_hat, y_i).to(device)
        for a, b in zip(y_hat.argmax(dim=1), y_i.argmax(dim=1)):
            correct_answers_train += int(a == b)
        loss.backward()
        
    optimizer.step()
    
    for batch_idx in range(x_test.shape[0] // batch_size):
        x_i = x_test[batch_size * batch_idx : batch_size * (batch_idx + 1)]\
            .type(torch.float64).to(device)
        y_i = y_test[batch_size * batch_idx : batch_size * (batch_idx + 1)]\
            .type(torch.float64).to(device)

        x_i = x_i.reshape(x_i.size()[0], 1, x_i.size()[1], x_i.size()[2])

        with torch.no_grad():
            y_hat = model(x_i)

        for a, b in zip(y_hat.argmax(dim=1), y_i.argmax(dim=1)):
            correct_answers_test += int(a == b)

    train_acc = correct_answers_train / x_train.shape[0]
    test_acc = correct_answers_test / x_test.shape[0]

    print(
        f'Epoch: {epoch + 1}, Train acc: {train_acc:.5f}, Test acc: {test_acc:.5f}')