# -*- code: utf-8 -*-
import argparse
import math
import numpy as np
import cupy as cp
import pandas as pd
from matplotlib import pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

from net import LSTMNet
from util import SequentialIterator, make_sin_data

DEFAULT_LOAD = 'result/snapshot_iter_200'

def main():
    parser = argparse.ArgumentParser(description='sine curve prediction')
    
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default=DEFAULT_LOAD,
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=32,
                        help='Number of units')
                        
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# units: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    net = LSTMNet(args.unit, 1)
    
    chainer.serializers.load_npz(args.resume, net, path='updater/model:main/predictor/')
    model = L.Classifier(net, lossfun=F.mean_squared_error, accfun=F.r2_score)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        xp = cp
    else:
        xp = np
    predictor = model.predictor

    n_cycle = 2
    data_per_cycle = 200
    n_data = n_cycle * data_per_cycle
    theta = xp.linspace(0., n_cycle * (2. * math.pi), num=n_data)

    X_test = xp.sin(theta)
    X_test /= xp.std(X_test)    # 本来はtrain.pyで使用した正規化係数を使う
    X_test = X_test.astype(xp.float32)

    model.predictor.reset_state()
    y_test = [X_test[0]]

    for i in range(n_data - 1):
        y = predictor(chainer.Variable(X_test[i].reshape((-1,1))))
        y_test.append(y.data)

    if args.gpu >= 0:
        theta = chainer.cuda.to_cpu(theta)
        X_test = chainer.cuda.to_cpu(X_test)
        y_test = chainer.cuda.to_cpu(y_test)

    # make&save DataFrame 
    df = pd.DataFrame({'theta': theta,
                       'X_test': X_test,
                       'y_test': y_test})

    df.to_csv('{}/pred.csv'.format(args.out), index=False)

    plt.scatter(theta, X_test, s=20., alpha=0.5)
    plt.plot(theta, y_test, c='orange')
    plt.savefig('{}/pred_graph.png'.format(args.out), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

