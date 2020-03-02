#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from model import NNetwork


# This function will load the dataset provided, splitting it via a passed in ratio
# between 0.0 and 1.0. The spliting can be randomized by passing in a different seed
# or None for the seed parameter. However, keeping the seed parameter constant ensures
# repeatability when testing.



# Main function for training: run in command line e.g. python train_proj2.py -b 20 -e 20
def main():
    parser = argparse.ArgumentParser(description='Keras example: 421 Project 1')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--snap', '-s', default=None,
                        help='Name of snapshot to load.')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='Number of sweeps over the dataset to train.')
    parser.add_argument('--network', '-n', type=str, default='mlp', choices=['mlp','conv'],
                        help='Type of network, either MLP or Convolutional.')
    parser.add_argument('--opt', '-o', type=str, default='sgd', choices=['sgd','adam','rmsprop'],
                        help='Type of optimizer to use when training.')
    parser.add_argument('--test_only', '-t', action="store_true",
                        help='Use when wanting to evaluate the accuracy of an already trained network.')
    args = parser.parse_args()

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
#    model = MLP()
    model = NNetwork(net_type=args.network,opt_type=args.opt,load_from=args.snap)
    x_train, y_train = model.load_data(datatype='train')
    x_test, y_test = model.load_data(datatype='test')
    
    if args.test_only:
        score = model.evaluate(x_test,y_test)
        print('accuracy of model on test set: %f' % score[1])
    else:
        model.train(x_train,y_train,x_test,y_test,epochs=args.epoch,batch_size=args.batchsize)

        # Save final snapshot, which is necessary for testing and saving progress
        model.save()
        # Plot loss and accuracy data
        model.plot()

if __name__ == '__main__':
    main()
