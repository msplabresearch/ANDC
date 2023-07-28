#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import keras
import random
import pickle as pk
import os
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
        'Generates data containing batch_size with fixed chunk samples'
        batch_x1 = []
        batch_x2 = []
        batch_y = [] 
        
        for i in range(len(list_path1_temp)):
            
            path1 = self.root_dir + list_path1_temp[i].replace('.wav', '.pk')
            path2 = self.root_dir + list_path2_temp[i].replace('.wav', '.pk') 
            if not os.path.exists(self.root_dir + list_path1_temp[i].replace('.wav', '.pk')):
                path1 = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/MSP_1.10_w2v2LR/MSP-PODCAST_0001_0025.pk'
            if not os.path.exists(self.root_dir + list_path2_temp[i].replace('.wav', '.pk')):
                path2 = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/MSP_1.10_w2v2LR/MSP-PODCAST_0001_0063.pk'  
            with open(path1, 'rb') as f:
                x1 = pk.load(f)
            with open(path2, 'rb') as f1:
                x2 = pk.load(f1)
            
            
            x1[np.isnan(x1)] = 0
            x2[np.isnan(x2)] = 0

            y = np.int(list_labels_temp[i])
            
                
            x1 = np.mean(x1, axis = 0).tolist()
            x2 = np.mean(x2, axis = 0).tolist()
            batch_x1.append(x1)
            batch_x2.append(x2)
            batch_y.append(y)
            
            
            
            
            
            
        batch_x1 = np.asarray(batch_x1)
        batch_x2 = np.asarray(batch_x2)
        batch_y = np.asarray(batch_y)
        #Y = tf.stack((batch_y1,batch_y2,batch_y),axis=1)  
        #print(batch_y.shape)  
        #print(batch_x1.shape,batch_x2.shape,batch_y.shape)
        #print(len(batch_x1),len(batch_y))
        #exit()  
        return [batch_x1, batch_x2], batch_y
