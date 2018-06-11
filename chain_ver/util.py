# -*- coding: utf-8 -*-
import math
import numpy as np

import chainer


def make_sin_data(data_per_cycle=200, n_cycle=5, train_ratio=0.8):
    np.random.seed(0)

    n_data = n_cycle * data_per_cycle
    theta = np.linspace(0., n_cycle * (2. * math.pi), num=n_data)

    X = np.sin(theta) + 0.1 * (2. * np.random.rand(n_data) - 1.)
    X /= np.std(X)
    X = X.astype(np.float32)

    n_train = int(n_data * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]

    return X_train, X_test


class SequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, seq_len, repeat=True):
        self.dataset = dataset
        self.n_samples = len(dataset)
        
        self.seq_len = seq_len  # for updater...
        self.batch_size = batch_size
        self.repeat = repeat

        self.epoch = 0
        self.detail_epoch = 0.
        self.iteration = 0
        self.update_offset_list()   # バッチサイズ分だけの起点のリスト
        
        self.sequence = 0

    def __next__(self):
        # for test iteration...
        if not self.repeat and self.iteration * self.batch_size >= self.n_samples:
            raise StopIteration

        x, t = self.get_data()
        self.iteration += 1
        
        self.sequence += 1

        new_epoch = int(self.epoch_detail)
        if new_epoch > self.epoch:
            self.epoch = new_epoch

        return list(zip(x, t))

    @property
    def epoch_detail(self):
        # for estimate time
        return self.iteration * self.batch_size / self.n_samples

    def update_offset_list(self):
        self.offset_list = np.random.randint(0, self.n_samples - 1 - self.seq_len, size=self.batch_size)

    def get_data(self):
        # offset_listから抜き出したoffsetにsequenceを足すことで、バッチごとに異なるx,tのセットとなる
        x = [self.dataset[offset + self.sequence    ][0] for offset in self.offset_list]
        t = [self.dataset[offset + self.sequence + 1][0] for offset in self.offset_list]
        return x, t

