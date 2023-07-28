#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import pickle
import keras
import os
import random
from utils import getPaths, DynamicChunkSplitTrainingData
# Ignore warnings & Fix random seed
import warnings

warnings.filterwarnings("ignore")
random.seed(999)
random_seed = 99


class DataGenerator_w2v(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, root_dir, label_dir, batch_size, shuffle=True):
        'Initialization'
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._path1, self._path2, self._labels = getPaths(label_dir)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(getPaths(self.label_dir)[0]) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find Batch list of Loading Paths
        list_path1_temp = [self._path1[k] for k in indexes]
        list_path2_temp = [self._path2[k] for k in indexes]
        list_labels_temp = [self._labels[k] for k in indexes]

        # Generate data
        data, label = self.__data_generation(list_path1_temp, list_path2_temp, list_labels_temp)
        return data, label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        _path1, _path2, _labels = getPaths(self.label_dir)
        self.indexes = np.arange(len(_path1))
        if self.shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_path1_temp, list_path2_temp, list_labels_temp):
        'Generates data containing batch_size with fixed chunck samples'
        batch_x1 = []
        batch_x2 = []
        batch_y = []
        for i in range(len(list_path1_temp)):
            # Store Norm-Data
            #x1 = loadmat(self.root_dir + list_path1_temp[i].replace('.wav', '.mat'))["projected_states"][0]
            #isExist = 
            #isExist1 = os.path.exists(self.root_dir + list_path2_temp[i].replace('.wav', '.pk'))
            path1 = self.root_dir + list_path1_temp[i].replace('.wav', '.pk')
            path2 = self.root_dir + list_path2_temp[i].replace('.wav', '.pk') 
            if not os.path.exists(self.root_dir + list_path1_temp[i].replace('.wav', '.pk')):
                path1 = self.root_dir + list_path1_temp[0].replace('.wav', '.pk')
            if not os.path.exists(self.root_dir + list_path2_temp[i].replace('.wav', '.pk')):
                path2 = self.root_dir + list_path2_temp[0].replace('.wav', '.pk')    
            with open(path1, 'rb') as f:
                x1 = pickle.load(f)
            with open(path2, 'rb') as f1:
                x2 = pickle.load(f1)
            #x2 = loadmat(self.root_dir + list_path2_temp[i].replace('.wav', '.mat'))["projected_states"][0]
            # we use the Interspeech 2013 computational paralinguistics challenge LLDs feature set
            # which includes totally 130 features (i.e., the "IS13_ComParE" configuration)
            x1[np.isnan(x1)] = 0
            x2[np.isnan(x2)] = 0

            y = np.int(list_labels_temp[i])
            #if len(x1) < 50:
                #print(list_path1_temp[i])

            batch_x1.append(x1)
            batch_x2.append(x2)
            batch_y.append(y)

        # split sentences into fixed length and fixed number of small chunks


        batch_chunck_x1, batch_chunck_x2, batch_chunck_y = DynamicChunkSplitTrainingData(batch_x1, batch_x2, batch_y, m=50, C=11, n=1)
        #batch_chunck_x1.reshape((len(batch_chunck_x1), 50, 256))
        #print(batch_chunck_x1.shape)
        #for i in range(len(batch_chunck_x1)):
         #   print(batch_chunck_x1[i].shape)
        #print(batch_chunck_x2.shape)
        #print(batch_chunck_y)
        #print(len([batch_chunck_x1, batch_chunck_x2]))
        #print([batch_chunck_x1, batch_chunck_x2][0].shape)
        #print([batch_chunck_x1, batch_chunck_x2][1].shape)
        #batch_chunck_x = np.array([batch_chunck_x1, batch_chunck_x2])
        #print(batch_chunck_x.shape)

        return [batch_chunck_x1, batch_chunck_x2], batch_chunck_y
