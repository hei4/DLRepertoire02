# -*- coding: utf-8 -*-
import chainer
from chainer import training


class SequentialUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device, converter):
        super(SequentialUpdater, self).__init__(train_iter, optimizer, device=device, converter=converter)
        self.seq_len = train_iter.seq_len
        
        self.train_iter = train_iter

    def update_core(self):
        loss = 0

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        for i in range(self.seq_len):
            batch = train_iter.__next__()
            new_batch = self.converter(batch, device=self.device)
            x, t = new_batch[0], new_batch[1]

            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        self.train_iter.update_offset_list()
        self.train_iter.sequence = 0

        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


class NStepSequentialUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device, converter):
        super(NStepSequentialUpdater, self).__init__(train_iter, optimizer, device=device, converter=converter)
        self.seq_len = train_iter.seq_len
        
        self.train_iter = train_iter

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()

        xs, ts = self.converter(batch, self.device)

        loss = optimizer.target(xs, ts)
        
        self.train_iter.update_offset_list()
        self.train_iter.sequence = 0

        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
