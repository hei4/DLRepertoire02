# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.initialization()

    def forward(self, x):
        h = self.pool(F.relu(self.cv1(x)))
        h = self.pool(F.relu(self.cv2(h)))
        h = h.view(-1, 64 * 7 * 7)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    def initialization(self):
        nn.init.xavier_normal_(self.cv1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal_(self.cv2.weight, gain=np.sqrt(2))
        nn.init.xavier_normal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal_(self.fc2.weight, gain=np.sqrt(2))


class LSTMCellNet(nn.Module):
    def __init__(self, n_unit):
        super(LSTMCellNet, self).__init__()
        self.fc1 = nn.Linear(1, n_unit)
        self.lstm = nn.LSTMCell(n_unit, n_unit)
        self.fc2 = nn.Linear(n_unit, 1)

        self.n_unit = n_unit

    def forward(self, x):
        ys = []
        h_t = torch.zeros(x.size(0), self.n_unit, dtype=torch.float) #.cuda()
        c_t = torch.zeros(x.size(0), self.n_unit, dtype=torch.float) #.cuda()

        for i, x_t in enumerate(x.chunk(x.size(1), dim=1)):
            h = self.fc1(x_t)
            h_t, c_t = self.lstm(h, (h_t, c_t))

            y = self.fc2(h_t)
            ys += [y]

        ys = torch.stack(ys, 1).squeeze(2)
        return ys

