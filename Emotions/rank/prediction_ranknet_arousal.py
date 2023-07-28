#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
import os, sys
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from keras import optimizers
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LSTM, Input, Multiply, Concatenate, Subtract, Activation
import random
# import matplotlib.pyplot as plt
from dataloader import DataGenerator_w2v
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from model_utils import crop, reshape, mean, repeat
from model_utils import atten_gated, atten_rnn, atten_selfMH, output_net
from transformer import ScaledDotProductAttention, LayerNormalization
from utils import getPaths_test, DynamicChunkSplitTestingData
import time
import argparse
from pathlib import Path
from scipy.io import loadmat
from tensorflow.keras.utils import plot_model
# Ignore warnings & Fix random seed
import warnings
import json
from tqdm import tqdm

warnings.filterwarnings("ignore")
random.seed(999)
random_seed = 99


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


###############################################################################



# Attention on LSTM chunk output => RNN-Attention/MultiHead(MH)-Self Attention
def UttrAtten_AttenVec(atten):
    time_step = 50  # same as the number of frames within a chunk (i.e., m)
    feat_num = 256  # number of LLDs features
    chunk_num = 11  # number of chunks splitted for a sentence (i.e., C)
    
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))

    encode = tf.keras.layers.LSTM(units=feat_num, activation='tanh', kernel_regularizer='l1', dropout=0.5,
                                  return_sequences=True)(inputs)
    encode = tf.keras.layers.LSTM(units=feat_num, activation='tanh', kernel_regularizer='l1', dropout=0.5,
                                  return_sequences=False)(encode)
    encode = BatchNormalization()(encode)

    # Uttr Attention Layer
    batch_atten_out = []
    for uttr_idx in range(0, batch_size * chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx + chunk_num

        encode_crop = crop(0, _start, _end)(encode)

        encode_crop = reshape()(encode_crop)
        atten_out = atten(encode_crop)
        batch_atten_out.append(atten_out)
    # Output-Layer

    concat_atten_out = tf.keras.layers.Concatenate(axis=0)(batch_atten_out)
    outputs = output_net(feat_num)(concat_atten_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)
    return model

def Rank_net(utteratten):
    time_step = 50
    feat_num = 256
    inputs1 = Input((50, 256))
    inputs2 = Input((50, 256))
    first_out = utteratten(inputs1)
    second_out = utteratten(inputs2)
    diff = Subtract()([first_out, second_out])
    outputs = tf.keras.activations.sigmoid(diff)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model



###############################################################################
#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args



args = get_args()
root = args.root
output_location = os.path.join(root, "Outputs")

with open(os.path.join(output_location, "Short_files.json"), "r") as openfile:
    audio_files = json.load(openfile)
    

# Parameters
batch_size = int(1)
epochs = int(10)
#emo_attr = args['emo_attr']
atten_type = 'RnnAttenVec'


#model path
model_path = os.path.join(file_dir, 'Models', 'LSTM_model[epoch10-batch128]_RnnAttenVec_aro_weights.h5')

params_train = {'batch_size': batch_size,
                'shuffle': True}

params_valid = {'batch_size': batch_size,
                'shuffle': False}



    

# Model Architecture
if atten_type == 'GatedVec':
    model = Rank_net(UttrAtten_GatedVec(atten_gated(feat_num=256, C=11)))
elif atten_type == 'RnnAttenVec':
    model = Rank_net(UttrAtten_AttenVec(atten_rnn(feat_num=256, C=11)))
elif atten_type == 'SelfAttenVec':
    model = Rank_net(UttrAtten_AttenVec(atten_selfMH(ffeat_num=256, C=11)))
elif atten_type == 'NonAtten':
    model = Rank_net(UttrAtten_NonAtten())


# print(3)
model.load_weights(model_path)

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('subtract').output)

Test_pred_Val = []
File_Name = []

print("RankNet inference: ")
for filename, f_info in tqdm(audio_files.items()):
    f_path = f_info['filepaths']['w2v_base_feats']
    data = np.load(f_path)[0]
    data[np.isnan(data)] = 0
    data1 = y = np.expand_dims(data, axis=0)
    chunk_data = DynamicChunkSplitTestingData(data1, m=50, C=11, n=1)
    pred = intermediate_layer_model.predict([chunk_data, np.zeros_like(chunk_data)]).tolist() 
    f_info['ranknet_pred'] = pred[0][0]
   

print("Writing json file")
json_object = json.dumps(audio_files, indent=4)
with open(os.path.join(output_location, "Short_files.json"), "w") as outfile:
    outfile.write(json_object)    
    
    
print('Rank Done!')