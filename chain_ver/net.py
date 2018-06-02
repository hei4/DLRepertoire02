# -*- coding: utf-8 -*-
import chainer.links as L
from chainer import Chain


class LSTMNet(Chain):

    def __init__(self, n_unit, n_out):
        super(LSTMNet, self).__init__(
            l1 = L.Linear(None, n_unit),
            l2 = L.LSTM(None, n_unit),
            l3 = L.Linear(None, n_out),
        )

    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x):
        # print('@network', x.shape)
        h1 = self.l1(x)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y
