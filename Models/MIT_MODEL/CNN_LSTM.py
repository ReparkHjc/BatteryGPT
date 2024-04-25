import torch.nn as nn
import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision

from math import sqrt
from datetime import datetime


# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error

# 3cell
# class Net(nn.Module):

# 5cell
class Net(nn.Module):
    def __init__(self, cell_num, input_size, hidden_dim, num_layers, sequencen_len=20, n_class=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        if mode == 'LSTM':
            self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim * sequencen_len, n_class)

        self.CNN = nn.Conv2d(in_channels=3, out_channels=1, stride=1, padding=1, kernel_size=(3, 3))
        self.CNN1 = nn.Conv2d(in_channels=cell_num, out_channels=1, stride=1, padding=1, kernel_size=(3, 3))
        self.flatten = nn.Flatten(1, 2)

    def forward(self, x):

        cell1 = x[:, 0]
        cell2 = x[:, 1]

        batchsize = x.shape[0]

        output1 = self.CNN(cell1)
        output2 = self.CNN(cell2)

        # cell_all = torch.cat([output1, output2,output3, output4,output5], dim=1)
        cell_all = torch.cat([output1, output2], dim=1)

        output = self.CNN1(cell_all).squeeze()

        x = output.reshape(batchsize, 20, -1)
        out, _ = self.cell(x)
        out = self.flatten(out)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        return out


class CNN(nn.Module):
    def __init__(self, cell_num, n_class):
        super(CNN, self).__init__()

        self.CNN = nn.Conv2d(in_channels=3, out_channels=1, stride=1, padding=1, kernel_size=(3, 3))
        self.CNN1 = nn.Conv2d(in_channels=cell_num, out_channels=1, stride=1, padding=1, kernel_size=(3, 3))
        self.flatten = nn.Flatten(1, 3)
        self.linear = nn.Linear(60 * 60, 100)
        self.linear1 = nn.Linear(100, n_class)

    def forward(self, x):
        cell1 = x[:, 0]
        cell2 = x[:, 1]

        batchsize = x.shape[0]

        output1 = self.CNN(cell1)
        output2 = self.CNN(cell2)

        cell_all = torch.cat([output1, output2], dim=1)

        output = self.CNN1(cell_all)
        out = self.flatten(output)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        out = self.linear1(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size=2700, hidden_dim=25, num_layers=3, sequencen_len=20, n_class=1):
        super(LSTM, self).__init__()
        self.flatten = nn.Flatten(2, 4)
        self.flatten1 = nn.Flatten(1, 2)
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim * sequencen_len, n_class)

    def forward(self, x):
        x = self.flatten(x)
        batchsize = x.shape[0]
        x = x.reshape(batchsize, 20, -1)

        x, _ = self.cell(x)

        x = self.flatten1(x)

        out = self.linear(x)  # out shape: (batch_size, n_class=1)

        return out


class MLP(torch.nn.Module):
    def __init__(self, cell_num, n_feature, n_hidden, n_class):
        super(MLP, self).__init__()
        # 两层感知机
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # self.hidden1=nn.Linear(99,1)
        self.predict = torch.nn.Linear(33 * cell_num, n_class)
        self.flatten = nn.Flatten(2, 4)
        self.flatten1 = nn.Flatten(1, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.hidden(x))
        x = self.flatten1(x)
        x = self.predict(x)
        return x
