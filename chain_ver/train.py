# -*- coding: utf-8 -*-
import argparse
import numpy as np

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
    parser = argparse.ArgumentParser(description='sin curve LSTM')
    
    parser.add_argument('--batchsize', '-b', type=int, default=10,
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

    model = L.Classifier(LSTMNet(args.unit, 1), lossfun=F.mean_squared_error, accfun=F.r2_score)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # データ作成分割
    X = make_sin_data()
    n_data = len(X)
    
    # plt.scatter(theta, X)
    # plt.show() 

    # データセット
    n_train = int(n_data * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]

    train_dataset = tuple_dataset.TupleDataset(X_train)
    test_dataset = tuple_dataset.TupleDataset(X_test)

    train_iter = SequentialIterator(train_dataset, batch_size=args.batchsize, seq_len=50)
    test_iter = SequentialIterator(test_dataset,  batch_size=args.batchsize, seq_len=50, repeat=False)

    def simple_converter(batch, device=None, padding=None):
        new_batch = []
        for x, t in batch:
            x = np.array(x, dtype=np.float32).reshape(1)
            t = np.array(t, dtype=np.float32).reshape(1)
            # print('@converter', x.shape, t.shape)
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
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()

