#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import datetime
import pickle
import h5py

from numpy.random import seed
from tensorflow import set_random_seed
seed(7)
set_random_seed(97)

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

import mnist_reader

INPUT_DIM = 784
OUTPUT_DIM = 10
OPTIMIZERS = {'sgd':SGD(lr=0.01, momentum=0.9, decay=0.9, nesterov=True), #nesterov is a variation on momentum...don't worry about these too much
              'adam':Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=True),
              'rmsprop':RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.9)}

# Network definition
class NNetwork():

    def __init__(self,net_type='mlp',opt_type='sgd',load_from=None):
        self._net_type = net_type
        if load_from:
            net_regex = re.compile('.*(mlp|conv).*')
            date_regex = re.compile('.*([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2})_.*')
            self._creation_date = date_regex.findall(load_from)[0]
#            self._curr_epoch = int(x.findall(load_from)[0])
            self._net_type = net_regex.findall(load_from)[0]
            try:
                with open('snapshots/' + self._creation_date + '_history_' + self._net_type + '_model', 'rb') as hist_file:
                    self._prev_history = pickle.load(hist_file)
            except(IOError):
                self._prev_history = {}
                print('\n#######################\n')
                print('looks like there is not a matching history file saved in the snapshots directory...\nUsing an empty history to start.')
                print('\n#######################\n')
            self._train_time = self._prev_history['train_time'] if 'train_time' in self._prev_history else 0.0
            self._curr_epoch = self._prev_history['curr_epoch'] if 'curr_epoch' in self._prev_history else 0
            self._net = keras.models.load_model('snapshots/' + load_from)
        elif net_type == 'mlp':
            self.MLP()
            self._creation_date = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
            self._curr_epoch = 0
            self._prev_history = {}
            self._train_time = 0.0
        elif net_type == 'conv':
            self.Conv()
            self._creation_date = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
            self._curr_epoch = 0
            self._prev_history = {}
            self._train_time = 0.0
        else:
            print('need to load a valid snapshot OR specify network type as \'mlp\' or \'conv\'')
            exit(-1)

        self._opt_type = opt_type
        self._net.compile(optimizer=OPTIMIZERS[opt_type],
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
                        
    def MLP(self):
        self._net = Sequential()



        self._net.add(Dense(40, activation='relu', input_dim=INPUT_DIM)) # DON'T change input_dim here
        self._net.add(Dense(38, activation='relu'))
        self._net.add(Dense(36, activation='relu'))
        self._net.add(Dense(17, activation='relu'))
        self._net.add(Dense(34, activation='relu'))
        self._net.add(Dense(19, activation='relu'))
        self._net.add(Dense(32, activation='relu'))
        self._net.add(Dense(30, activation='relu'))

        #This is the output layer, needs to have its output dimensions same as number of categories
        #Activation is 'softmax' such that the output is probabilities for each class
        self._net.add(Dense(OUTPUT_DIM, activation='softmax')) 
        
    def Conv(self):
        self._net = Sequential()



        self._net.add(Conv2D(8,(3,3),padding='same',activation='relu',input_shape=(28,28,1))) #DON'T change input shape here
        self._net.add(Conv2D(8,(3,3),padding='same',activation='relu')) #This is an example layer after the first layer

        self._net.add(Conv2D(8,(1,2), padding='same', activation='relu'))
        self._net.add(Conv2D(4,(1,2), padding='same', activation='relu'))

        self._net.add(MaxPooling2D(pool_size=(2,2)))
        #DON'T remove the flatten layer...keep Conv layers above this line and Dense layers below it.
        self._net.add(Flatten())

        #Dense layers can go here.
        self._net.add(Dense(40,activation='relu'))
        self._net.add(Dense(38,activation='relu'))


        self._net.add(Dense(OUTPUT_DIM, activation='softmax'))
        
    def train(self,data,labels,val_data,val_labels,epochs=3,batch_size=32):
        start_time = time.time()
        self._history = self._net.fit(data,labels,epochs=self._curr_epoch + epochs,batch_size=batch_size,
                                      initial_epoch=self._curr_epoch,validation_data=(val_data,val_labels),shuffle=True)
        total_time = time.time() - start_time
        self._train_time = self._train_time + total_time
        self._curr_epoch = self._curr_epoch + epochs
        print('time training (in seconds) %f' % self._train_time)
        
    def evaluate(self,data,labels,batch_size=32):
        return self._net.evaluate(data,labels,batch_size=batch_size)

    def load_data(self,file_name='fashion-mnist/data/fashion',datatype='train'):
        base_path = os.getcwd() + '/../'
        if datatype == 'train':
            x, y = mnist_reader.load_mnist(base_path + file_name,kind='train')
        elif datatype == 'test':
            x, y = mnist_reader.load_mnist(base_path + file_name,kind='t10k')
        else:
            print('datatype must be either \'train\' or \'test\'')
            exit(-1)

        if self._net_type == 'conv':
            x = x.reshape(-1,28,28,1)
        y = keras.utils.to_categorical(y, num_classes=10)
            
        return x,y
    
    def save(self):
        file_path = 'snapshots/' + self._creation_date + '_my_' + self._net_type + '_model'
        f = h5py.File(file_path,'w')
        self._net.save(f) #_epoch' + str(self._curr_epoch))

        for k in self._history.history.keys():
            if k in self._prev_history:
                self._history.history[k] = self._prev_history[k] + self._history.history[k]

        self._history.history['train_time'] = self._train_time
        self._history.history['curr_epoch'] = self._curr_epoch
        file_path = 'snapshots/' + self._creation_date + '_history_' + self._net_type + '_model'
        with open(file_path,'wb') as hist_file:
            pickle.dump(self._history.history,hist_file)

    def plot(self):

        plt.plot(self._history.history['acc'])
        plt.plot(self._history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('pics/' + self._creation_date + '_my_' + self._net_type + '_model_acc_results.png')

        plt.clf()

        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('pics/' + self._creation_date + '_my_' + self._net_type + '_model_loss_results.png')
        
