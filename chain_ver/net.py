# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain


class LSTMNet(Chain):

    def __init__(self, n_unit, n_out):
        super(LSTMNet, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_unit)
            self.lstm = L.LSTM(None, n_unit)
            self.fc2 = L.Linear(None, n_out)

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x):
        h = self.fc1(x)
        h = self.lstm(h)
        return self.fc2(h)
        

class NStepLSTMNet(Chain):

    def __init__(self, n_layer, n_unit, n_out):
        super(NStepLSTMNet, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_unit)
            self.lstm = L.NStepLSTM(n_layers=n_layer, in_size=n_unit, out_size=n_unit, dropout=0.)
            self.fc2 = L.Linear(None, n_out)

            self.n_layer = n_layer
            self.n_unit = n_unit

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x[0].data)

        cx = F.concat(x, axis=0)
        cx = cx.reshape(-1, 1)

        ex = self.fc1(cx)

        x_len = [len(x_) for x_ in x]
        x_section = xp.cumsum(x_len[:-1])
        exs = F.split_axis(ex, x_section, 0, force_tuple=True)

        _, _, h = self.lstm(None, None, exs)

        ch = F.concat(h, axis=0)
        ch = ch.reshape(-1, self.n_unit)

        eh = self.fc2(ch)
        eh = eh.reshape(-1, )

        h_len = [len(h_) for h_ in h]
        h_section = xp.cumsum(h_len[:-1])
        ehs = F.split_axis(eh, h_section, 0, force_tuple=True)

        return ehs

