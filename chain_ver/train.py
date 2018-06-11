# -*- coding: utf-8 -*-
import argparse
import math
import numpy as np
import cupy as cp
import pandas as pd
from matplotlib import pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer.dataset.convert import concat_examples

from net import LSTMNet
from util import SequentialIterator, make_sin_data
from update import SequentialUpdater


def main():
    parser = argparse.ArgumentParser(description='sine curve training')

    parser.add_argument('--seqlen', '-s', type=int, default=50,
                        help='length of sequence')    
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')                        
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=32,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    
    args = parser.parse_args()

    net = LSTMNet(args.unit, 1)
    model = L.Classifier(net, lossfun=F.mean_squared_error, accfun=F.r2_score)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
        xp = cp
    else:
        xp = np

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # データ作成
    data_per_cycle, n_cycle, train_ratio = 200, 5, 0.8
    X_train, X_test = make_sin_data(data_per_cycle=data_per_cycle, n_cycle=n_cycle, train_ratio=train_ratio)
    
    if len(X_train) % (args.batchsize * args.seqlen) != 0:
        print('ABORT. {}(X_train) % ({}(batch) * {}(seq)) != 0 ...'.format(len(X_train), args.batchsize, args.seqlen))
        return

    train_dataset = tuple_dataset.TupleDataset(X_train)
    test_dataset = tuple_dataset.TupleDataset(X_test)

    train_iter = SequentialIterator(train_dataset, batch_size=args.batchsize, seq_len=args.seqlen)
    test_iter = SequentialIterator(test_dataset,  batch_size=args.batchsize, seq_len=args.seqlen, repeat=False)
    
    def simple_converter(batch, device=None, padding=None):
        new_batch = []
        for x, t in batch:
            x = np.array(x, dtype=np.float32).reshape(1)
            t = np.array(t, dtype=np.float32).reshape(1)
            new_batch.append((x, t))
        return concat_examples(new_batch, device=device, padding=padding)

    updater = SequentialUpdater(train_iter, optimizer, device=args.gpu, converter=simple_converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    eval_rnn.train = False

    trainer.extend(extensions.Evaluator(
            test_iter, eval_model, device=args.gpu, converter=simple_converter,
            eval_hook=lambda _: eval_rnn.reset_state()))

    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch', file_name='loss.png'))
        
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # visualizing...
    n_data = n_cycle * data_per_cycle
    theta = np.linspace(0., n_cycle * (2. * math.pi), num=n_data)

    n_train = int(n_data * train_ratio)
    theta_train = theta[:n_train]

    if args.gpu >=0:
        X_train = chainer.cuda.to_gpu(X_train)

    predictor = model.predictor
    predictor.reset_state()
    y_train = [X_train[0]]

    for i in range(n_train - 1):
        y = predictor(chainer.Variable(X_train[i].reshape((-1,1))))
        y_train.append(y.data)

    if args.gpu >= 0:
        X_train = chainer.cuda.to_cpu(X_train)
        y_train = chainer.cuda.to_cpu(y_train)

    # make&save DataFrame
    df = pd.DataFrame({'theta': theta_train,
                       'X_test': X_train,
                       'y_test': y_train})

    df.to_csv('{}/pred.csv'.format(args.out), index=False)

    plt.scatter(theta_train, X_train, s=20., alpha=0.5)
    plt.plot(theta_train, y_train, c='orange')
    plt.savefig('{}/train_graph.png'.format(args.out), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

