# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from net import LSTMCellNet
from util import make_sin_data


def update_offset_list(n_samples, seq_len, batch_size):
    return np.random.randint(0, n_samples - 1 - seq_len, size=batch_size)


def main():
    parser = argparse.ArgumentParser(description='LSTM sine wave')

    parser.add_argument('--seqlen', '-s', type=int, default=50,
                        help='length of sequence')
    parser.add_argument('--batch', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--display', '-d', type=int, default=100,
                        help='Number of interval to show progress')
    parser.add_argument('--unit', '-u', type=int, default=32,
                        help='Number of units')

    args = parser.parse_args()

    seq_len = args.seqlen
    batch_size = args.batch
    epoch_size = args.epoch
    display_interval = args.display



    # データ作成
    data_per_cycle, n_cycle, train_ratio = 200, 5, 0.8
    train_data, test_data = make_sin_data(data_per_cycle=data_per_cycle, n_cycle=n_cycle, train_ratio=train_ratio)

    n_loop = len(train_data) // (seq_len * batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    net = LSTMCellNet(n_unit=args.unit)
    print(net)
    print()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epoch_size):
        running_loss = 0
        train_t_list = []
        train_y_list = []

        for loop in range(n_loop):
            offset_list = update_offset_list(n_samples=len(train_data), seq_len=seq_len, batch_size=batch_size)

            for sequence in range(seq_len):
                x = [train_data[offset + sequence] for offset in offset_list]
                t = [train_data[offset + sequence + 1] for offset in offset_list]
                train_t_list.extend(t)

                x = np.array(x, dtype=np.float32).reshape(-1, 1)
                t = np.array(t, dtype=np.float32).reshape(-1, 1)

                x = torch.from_numpy(x)
                t = torch.from_numpy(t)

                optimizer.zero_grad()
                y = net(x)
                train_y_list.extend(y.tolist())

                loss = criterion(y, t)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        test_t_list = list(test_data[1:])
        test_y_list = []
        with torch.no_grad():
            for i in range(len(test_data) - 1):
                x = test_data[i]
                t = test_data[i + 1]
                
                x = np.array(x, dtype=np.float32).reshape(-1, 1)
                t = np.array(t, dtype=np.float32).reshape(-1, 1)

                x = torch.from_numpy(x)
                t = torch.from_numpy(t)
                
                y = net(x)
                test_y_list.append(y)

        print('epoch: {}  loss: {:.6f}  train_r2: {:.3f} test_r2: {:.3f}'.format(
            epoch + 1, running_loss, r2_score(train_y_list, train_t_list), r2_score(test_y_list, test_t_list)))


if __name__ == '__main__':
    main()
