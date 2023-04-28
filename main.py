import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import mkdataset
import neironetwork

device = torch.device("cuda")

x_train, y_train, x_test, y_test = mkdataset.init_dataset()

print(x_train.shape)

n_epochs = 50
eta = 0.001
model = neironetwork.MyModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=eta)

batch_size = 64

for epoch in range(n_epochs):
    correct_answers_train = 0
    correct_answers_test = 0

    # shuffle по x_train, y_train
    # last batch

    optimizer.zero_grad()
    for idx in tqdm(range(x_train.shape[0])):
        x_i = x_train[idx].reshape(1, 150, 150)
        y_i = y_train[idx]

        y_hat = model(x_i)

        loss = F.mse_loss(y_hat, y_i).to(device)
        for a, b in zip(y_hat.argmax(dim=1), y_i.argmax(dim=1)):
            correct_answers_train += int(a == b)
        
        loss.backward()
        
    optimizer.step()
    
    for batch_idx in range(x_test.shape[0]):
        x_i = x_test[idx]
        y_i = y_test[idx]

        with torch.no_grad():
            y_hat = model(x_i)

        for a, b in zip(y_hat.argmax(dim=1), y_i.argmax(dim=1)):
            correct_answers_test += int(a == b)

    train_acc = correct_answers_train / x_train.shape[0]
    test_acc = correct_answers_test / x_test.shape[0]

    print(
        f'Epoch: {epoch + 1}, Train acc: {train_acc:.5f}, Test acc: {test_acc:.5f}')