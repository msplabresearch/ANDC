#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import pandas as pd
import numpy as np
from keras import backend as K


def getPaths(path_label):
    """
    This function is for filtering data by different constraints of label
    Args:
        path_label$ (str): path of label.
        split_set$ (str): 'Train', 'Validation' or 'Test' are supported.
        emo_attr$ (str): 'Act', 'Dom' or 'Val'
    """
    label_table = np.load(path_label)
    path1 = label_table[:, 0]
    path2 = label_table[:, 1]
    label_com = label_table[:, 4]
    return np.array(path1), np.array(path2), np.asarray(label_com).astype(np.float32)

def getPaths_test(path_label):
    """
    This function is for filtering data by different constraints of label
    Args:
        path_label$ (str): path of label.
        split_set$ (str): 'Train', 'Validation' or 'Test' are supported.
        emo_attr$ (str): 'Act', 'Dom' or 'Val'
    """
    samples = open(path_label, "r", encoding = 'UTF-8')
    file_list1 = samples.read().splitlines()
    return np.array(file_list1)


# Combining list of data arrays into a single large array
def CombineListToMatrix(Data):
    length_all = []
    for i in range(len(Data)):
        length_all.append(len(Data[i]))
    feat_num = len(Data[0].T)
    Data_All = np.zeros((sum(length_all), feat_num))
    idx = 0
    Idx = []
    for i in range(len(length_all)):
        idx = idx + length_all[i]
        Idx.append(idx)
    for i in range(len(Idx)):
        if i == 0:
            start = 0
            end = Idx[i]
            Data_All[start:end] = Data[i]
        else:
            start = Idx[i - 1]
            end = Idx[i]
            Data_All[start:end] = Data[i]
    return Data_All


# evaluated by CCC metric
def evaluation_metrics(true_value, predicted_value):
    corr_coeff = np.corrcoef(true_value, predicted_value)
    ccc = 2 * predicted_value.std() * true_value.std() * corr_coeff[0, 1] / (
                predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean()) ** 2)
    return (ccc, corr_coeff)


# CCC loss function
def cc_coef(y_true, y_pred):
    mu_y_true = K.mean(y_true)
    mu_y_pred = K.mean(y_pred)
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (
                K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))


# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitTrainingData(Batch_data1, Batch_data2, Batch_label, m, C, n):
    """
    Note! This function can't process sequence length which less than given m=62
    (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Batch_data$ (list): list of data arrays for a single batch.
        Batch_label$ (list): list of training targets for a single batch.
                  m$ (int) : chunk window length (i.e., number of frames within a chunk)
                  C$ (int) : number of chunks splitted for a sentence
                  n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n * C - 1  # Tmax = 11sec (for the MSP-Podcast corpus),
    # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data1 = []
    Split_Data2 = []
    Split_Label = np.array([])
    for i in range(len(Batch_data1)):
        data1 = Batch_data1[i]
        data2 = Batch_data2[i]
        label = Batch_label[i]
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size1 = int(int(len(data1) - m) / num_shifts)
        step_size2 = int(int(len(data2) - m) / num_shifts)
        # Calculate index of chunks
        start_idx1 = [0]
        end_idx1 = [m]
        start_idx2 = [0]
        end_idx2 = [m]
        for iii in range(num_shifts):
            start_idx1.extend([start_idx1[0] + (iii + 1) * step_size1])
            end_idx1.extend([end_idx1[0] + (iii + 1) * step_size1])
            start_idx2.extend([start_idx2[0] + (iii + 1) * step_size2])
            end_idx2.extend([end_idx2[0] + (iii + 1) * step_size2])
            # Output Split Data
        for iii in range(len(start_idx1)):
            Split_Data1.append(data1[start_idx1[iii]: end_idx1[iii]])
        for iii in range(len(start_idx2)):
            Split_Data2.append(data2[start_idx2[iii]: end_idx2[iii]])
            # Output Split Label
        '''''
        print(np.array(Split_Data1).shape)
        print(np.array(Split_Data2).shape)
        if len(np.array(Split_Data1).shape) == 1:
            print(len(data1))
            print(step_size1)
            print(start_idx1)
            print(end_idx1)
            print(data1[start_idx1[iii]: end_idx1[iii]].shape)
            #print(data1[start_idx1[iii]: end_idx1[iii]])
            print(np.sum(data1[start_idx1[iii]: end_idx1[iii]]))
        '''''
        split_label = np.repeat(label, len(start_idx1))
        Split_Label = np.concatenate((Split_Label, split_label))
    #print(np.asarray(Split_Data1).shape)
    #print(np.asarray(Split_Data2).shape)
    #print(Split_Label.shape)
    return np.asarray(Split_Data1), np.asarray(Split_Data2), Split_Label


# split original batch data into batch small-chunks data with
# proposed dynamic window step size which depends on the sentence duration 
def DynamicChunkSplitTestingData(Online_data, m, C, n):
    """
    Note! This function can't process sequence length which less than given m=62
    (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
    Please make sure all your input data's length are greater then given m.
    
    Args:
         Online_data$ (list): list of data array for a single sentence
                   m$ (int) : chunk window length (i.e., number of frames within a chunk)
                   C$ (int) : number of chunks splitted for a sentence
                   n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n * C - 1  # Tmax = 11sec (for the MSP-Podcast corpus),
    # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
    Split_Data = []
    for i in range(len(Online_data)):
        data = Online_data[i]
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(len(data) - m) / num_shifts)
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii + 1) * step_size])
            end_idx.extend([end_idx[0] + (iii + 1) * step_size])
            # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append(data[start_idx[iii]: end_idx[iii]])
    return np.asarray(Split_Data)
