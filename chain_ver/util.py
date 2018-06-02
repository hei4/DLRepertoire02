# -*- coding: utf-8 -*-
import numpy as np

import chainer


def make_sin_data(data_per_cycle=200, n_cycle=5):
    np.random.seed(0)

    n_data = n_cycle * data_per_cycle
    theta = np.linspace(0., n_cycle * (2. * np.pi), num=n_data)

    X = np.sin(theta) + 0.1 * (2. * np.random.rand(n_data) - 1.)
    X /= np.std(X)

    return X.astype(np.float32)


class SequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size=10, seq_len=5, repeat=True):
        self.dataset = dataset
        self.n_samples = len(dataset)
        # self.n_epoch = n_epoch

        self.seq_length = seq_len
        self.batch_size = batch_size
        self.repeat = repeat

        self.epoch = 0
        self.detail_epoch = 0.
        self.iteration = 0
        self.offset_list = np.random.randint(0, len(dataset), size=batch_size)  # バッチサイズ分だけ

        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.n_samples:
            raise StopIteration

        x, t = self.get_data()
        self.iteration += 1

        self.detail_epoch = self.iteration * self.batch_size / self.n_samples

        new_epoch = int(self.detail_epoch)
        if new_epoch > self.epoch:
            self.epoch = new_epoch
            self.offset_list = np.random.randint(0, self.n_samples, size=self.batch_size)  # 新しいepochでは起点を新しくする

        return list(zip(x, t))

    @property
    def epoch_detail(self):
        # for estimate time
        return self.detail_epoch

    def get_data(self):
        # offset_listから抜き出したoffsetにiterを足すことで、バッチごとに異なるx,tのセットとなる
        # さらにn_samplesでの剰余とすることで配列外へのアクセスを防いでいる
        x = [self.dataset[(offset + self.iteration) % self.n_samples][0]
             for offset in self.offset_list]
        t = [self.dataset[(offset + self.iteration + 1) % self.n_samples][0]
             for offset in self.offset_list]
        return x, t
