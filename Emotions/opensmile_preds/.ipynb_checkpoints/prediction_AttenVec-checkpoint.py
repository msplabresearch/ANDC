#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import SimpleRNN, LSTM, Lambda, Input, Dot, Concatenate
import tensorflow as tf
import random
import os, sys
from keras import backend as K
from scipy.io import loadmat
import csv
import json
import argparse
from pathlib import Path
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from csv import writer
from tqdm import tqdm





def mean():  
    def func(x):
        return K.mean(x, axis=1, keepdims=False)
    return Lambda(func)

def reshape_1():
    def func(x):
        feat_num = 130
        chunk_num = 11
        return K.reshape(x, (1, chunk_num, feat_num))
    return Lambda(func)

def reshape_2():
    def func(x):
        feat_num = 130
        chunk_num = 17
        return K.reshape(x, (1, chunk_num, feat_num))
    return Lambda(func)

# Split Original batch Data into Small-Chunk batch Data Structure with different step window size
def DiffRslChunkSplitTestingData(Batch_data, chunk_num):
    """
    Note!!! This function can't process sequence length which less than given chunk_size (i.e.,1sec=62frames)
    Please make sure all your input data's length are greater then given chunk_size
    """
    chunk_size = 62  # (62-frames*0.016sec) = 0.992sec
    n = 1
    num_shifts = n*chunk_num-1  # max_length = 11sec, chunk needs to shift 10 times to obtain total 11 chunks for each utterance
    Split_Data = []
    for i in range(len(Batch_data)):
        data = Batch_data[i]
        # Shifting-Window size varied by differenct length of input utterance => Different Resolution
        step_size = int(int(len(data)-chunk_size)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [chunk_size]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
    return np.array(Split_Data)
###############################################################################

#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args
print(file_dir)
print(file_dir)
print(file_dir)
print(file_dir)


args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(os.path.join(output_location, "Short_files.json")), "r") as openfile:
    audio_files = json.load(openfile)
    
    
# Parameters
dataset = args.dataset


# Data/Label Dir
model_path = os.path.join(file_dir,'AttenVec',dataset,'Model')
meta_path = os.path.join(file_dir,'AttenVec',dataset,'NormTerm')
Feat_mean_All = loadmat(os.path.join(meta_path, 'feat_norm_means.mat'))['normal_para']
Feat_std_All = loadmat(os.path.join(meta_path, 'feat_norm_stds.mat'))['normal_para']
Label_mean_act = loadmat(os.path.join(meta_path, 'act_norm_means.mat'))['normal_para'][0][0]
Label_std_act = loadmat(os.path.join(meta_path, 'act_norm_stds.mat'))['normal_para'][0][0]
Label_mean_dom = loadmat(os.path.join(meta_path, 'dom_norm_means.mat'))['normal_para'][0][0]
Label_std_dom = loadmat(os.path.join(meta_path, 'dom_norm_stds.mat'))['normal_para'][0][0]
Label_mean_val = loadmat(os.path.join(meta_path, 'val_norm_means.mat'))['normal_para'][0][0]
Label_std_val = loadmat(os.path.join(meta_path, 'val_norm_stds.mat'))['normal_para'][0][0] 

# Setting Model Graph
time_step = 62
feat_num = 130

# Shared LSTM Layer
inputs = Input((time_step, feat_num))
shared = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
shared = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(shared)
shared = BatchNormalization()(shared)
if dataset=='MSP-Podcast':
    shared = reshape_1()(shared)
else:
    shared = reshape_2()(shared)   
# Act Attention Layer
encode_act = SimpleRNN(units=feat_num, activation='tanh', return_sequences=True)(shared)
score_first_part_act = Dense(feat_num, use_bias=False)(encode_act)
h_t_act = Lambda(lambda x: x[:, -1, :], output_shape=(feat_num,))(encode_act)
score_act = Dot(axes=(2, 1))([score_first_part_act, h_t_act])
attention_weights_act = Activation('softmax')(score_act)
context_vector_act = Dot(axes=(1, 1))([encode_act, attention_weights_act])
pre_activation_act = Concatenate(axis=1)([context_vector_act, h_t_act])
attention_vector_act = Dense(feat_num, use_bias=False, activation='tanh')(pre_activation_act)

# Dom Attention Layer
encode_dom = SimpleRNN(units=feat_num, activation='tanh', return_sequences=True)(shared)
score_first_part_dom = Dense(feat_num, use_bias=False)(encode_dom)
h_t_dom = Lambda(lambda x: x[:, -1, :], output_shape=(feat_num,))(encode_dom)
score_dom = Dot(axes=(2, 1))([score_first_part_dom, h_t_dom])
attention_weights_dom = Activation('softmax')(score_dom)
context_vector_dom = Dot(axes=(1, 1))([encode_dom, attention_weights_dom])
pre_activation_dom = Concatenate(axis=1)([context_vector_dom, h_t_dom])
attention_vector_dom = Dense(feat_num, use_bias=False, activation='tanh')(pre_activation_dom)

# Val Attention Layer
encode_val = SimpleRNN(units=feat_num, activation='tanh', return_sequences=True)(shared)
score_first_part_val = Dense(feat_num, use_bias=False)(encode_val)
h_t_val = Lambda(lambda x: x[:, -1, :], output_shape=(feat_num,))(encode_val)
score_val = Dot(axes=(2, 1))([score_first_part_val, h_t_val])
attention_weights_val = Activation('softmax')(score_val)
context_vector_val = Dot(axes=(1, 1))([encode_val, attention_weights_val])
pre_activation_val = Concatenate(axis=1)([context_vector_val, h_t_val])
attention_vector_val = Dense(feat_num, use_bias=False, activation='tanh')(pre_activation_val)

# Output Layer
output_act = Dense(units=feat_num, activation='relu')(attention_vector_act)
output_act = Dense(units=1, activation='linear')(output_act)  
output_dom = Dense(units=feat_num, activation='relu')(attention_vector_dom)
output_dom = Dense(units=1, activation='linear')(output_dom) 
output_val = Dense(units=feat_num, activation='relu')(attention_vector_val)
output_val = Dense(units=1, activation='linear')(output_val) 
with tf.device('/cpu:0'):
    model = Model(inputs=inputs, outputs=[output_act, output_dom, output_val])

# Shared LSTM Parameters
lstm1_w = np.load(model_path+'/lstm1_w.npy')
lstm1_c = np.load(model_path+'/lstm1_c.npy')
lstm1_b = np.load(model_path+'/lstm1_b.npy')
model.layers[1].set_weights([lstm1_w, lstm1_c, lstm1_b])
lstm2_w = np.load(model_path+'/lstm2_w.npy')
lstm2_c = np.load(model_path+'/lstm2_c.npy')
lstm2_b = np.load(model_path+'/lstm2_b.npy')
model.layers[2].set_weights([lstm2_w, lstm2_c, lstm2_b])
bn_gamma = np.load(model_path+'/bn_gamma.npy')
bn_beta = np.load(model_path+'/bn_beta.npy')
bn_mean = np.load(model_path+'/bn_mean.npy')
bn_var = np.load(model_path+'/bn_var.npy')
model.layers[3].set_weights([bn_gamma, bn_beta, bn_mean, bn_var])
# Act Attention Parameters
atten_rnn_w_act = np.load(model_path+'/atten_rnn_w_act.npy')  
atten_rnn_c_act = np.load(model_path+'/atten_rnn_c_act.npy')
atten_rnn_b_act = np.load(model_path+'/atten_rnn_b_act.npy')
model.layers[5].set_weights([atten_rnn_w_act, atten_rnn_c_act, atten_rnn_b_act])
atten_w1_act = np.load(model_path+'/atten_w1_act.npy')
model.layers[8].set_weights([atten_w1_act])
atten_w2_act = np.load(model_path+'/atten_w2_act.npy')
model.layers[26].set_weights([atten_w2_act])
# Dom Attention Parameters
atten_rnn_w_dom = np.load(model_path+'/atten_rnn_w_dom.npy')  
atten_rnn_c_dom = np.load(model_path+'/atten_rnn_c_dom.npy')
atten_rnn_b_dom = np.load(model_path+'/atten_rnn_b_dom.npy')
model.layers[6].set_weights([atten_rnn_w_dom, atten_rnn_c_dom, atten_rnn_b_dom])
atten_w1_dom = np.load(model_path+'/atten_w1_dom.npy')
model.layers[10].set_weights([atten_w1_dom])
atten_w2_dom = np.load(model_path+'/atten_w2_dom.npy')
model.layers[27].set_weights([atten_w2_dom])    
# Val Attention Parameters
atten_rnn_w_val = np.load(model_path+'/atten_rnn_w_val.npy')  
atten_rnn_c_val = np.load(model_path+'/atten_rnn_c_val.npy')
atten_rnn_b_val = np.load(model_path+'/atten_rnn_b_val.npy')
model.layers[7].set_weights([atten_rnn_w_val, atten_rnn_c_val, atten_rnn_b_val])
atten_w1_val = np.load(model_path+'/atten_w1_val.npy')
model.layers[12].set_weights([atten_w1_val])
atten_w2_val = np.load(model_path+'/atten_w2_val.npy')
model.layers[28].set_weights([atten_w2_val])    
# Act Output Parameters
output1_w_act = np.load(model_path+'/output1_w_act.npy')
output1_b_act = np.load(model_path+'/output1_b_act.npy')
model.layers[29].set_weights([output1_w_act, output1_b_act])
output2_w_act = np.load(model_path+'/output2_w_act.npy')
output2_b_act = np.load(model_path+'/output2_b_act.npy')
model.layers[32].set_weights([output2_w_act, output2_b_act])
# Dom Output Parameters
output1_w_dom = np.load(model_path+'/output1_w_dom.npy')
output1_b_dom = np.load(model_path+'/output1_b_dom.npy')
model.layers[30].set_weights([output1_w_dom, output1_b_dom])
output2_w_dom = np.load(model_path+'/output2_w_dom.npy')
output2_b_dom = np.load(model_path+'/output2_b_dom.npy')
model.layers[33].set_weights([output2_w_dom, output2_b_dom])
# Val Output Parameters
output1_w_val = np.load(model_path+'/output1_w_val.npy')
output1_b_val = np.load(model_path+'/output1_b_val.npy')
model.layers[31].set_weights([output1_w_val, output1_b_val])
output2_w_val = np.load(model_path+'/output2_w_val.npy')
output2_b_val = np.load(model_path+'/output2_b_val.npy')
model.layers[34].set_weights([output2_w_val, output2_b_val])


def TryToFloat(single_data):
    try:
        return float(single_data)
    except:
        return None
    
def LoadFeature(filename):
    content = open(filename, 'r').read()
    data = content.split('@data\n')[1].split('\n')
    data = filter(None, data)
    feature = [[TryToFloat(data_split) for data_split in d.split(',') \
                if TryToFloat(data_split)!=None] for d in data]
    return np.array(feature)




# Testing Data & Label
pred_type = 'AttenVec_'+dataset
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['opensmile_lld']
    data = LoadFeature(f_path)
    data = data[:,1:]                           # remove time-info
    data = (data-Feat_mean_All)/Feat_std_All    # Feature Normalization  
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3
    if dataset == 'MSP-Podcast':
        chunk_data = DiffRslChunkSplitTestingData([data], chunk_num=11)
    else:
        chunk_data = DiffRslChunkSplitTestingData([data], chunk_num=17)
    # Recording prediction time cost
    pred_act, pred_dom, pred_val = model.predict(chunk_data)
    # Output prediction results    
    pred_act, pred_dom, pred_val = np.mean(pred_act), np.mean(pred_dom), np.mean(pred_val)
    
    # Regression Task => Prediction & De-Normalize Target
    pred_act = (Label_std_act*pred_act)+Label_mean_act
    pred_dom = (Label_std_dom*pred_dom)+Label_mean_dom
    pred_val = (Label_std_val*pred_val)+Label_mean_val
    
    f_info[pred_type] = {}
    f_info[pred_type]['act'] = [pred_act,'Not_Used']
    f_info[pred_type]['dom'] = [pred_dom,'Not_Used']
    f_info[pred_type]['val'] = [pred_val,'Not_Used']
    
print("Saving updated json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)