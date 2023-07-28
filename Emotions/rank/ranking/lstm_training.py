#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from keras import optimizers
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LSTM, Input, Multiply, Subtract, Activation
from tensorflow.keras.layers import Concatenate
import random
import matplotlib.pyplot as plt
from dataloader import DataGenerator_w2v
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from model_utils import crop, reshape, mean, repeat
from model_utils import atten_gated, atten_rnn, atten_selfMH, output_net
from transformer import ScaledDotProductAttention, LayerNormalization
from utils import cc_coef
import time
import argparse
from tensorflow.keras.utils import plot_model
# Ignore warnings & Fix random seed
import warnings

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


# Attention on LSTM chunk output => Weighted Mean of the gated-Attention model
def UttrAtten_GatedVec(atten):
    time_step = 62  # same as the number of frames within a chunk (i.e., m)
    feat_num = 130  # number of LLDs features
    chunk_num = 11  # number of chunks splitted for a sentence (i.e., C)
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    # Uttr Attention Layer
    batch_atten_out = []
    for uttr_idx in range(0, batch_size * chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx + chunk_num
        encode_crop = crop(0, _start, _end)(encode)
        encode_crop = reshape()(encode_crop)
        atten_weights = atten(encode_crop)
        atten_out = Multiply()([encode_crop, atten_weights])
        atten_out = mean()(atten_out)
        batch_atten_out.append(atten_out)
    # Output-Layer
    concat_atten_out = Concatenate(axis=0)(batch_atten_out)
    outputs = output_net(feat_num)(concat_atten_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)
    return model



# Attention on LSTM chunk output => RNN-Attention/MultiHead(MH)-Self Attention
def UttrAtten_AttenVec(atten):
    time_step = 50  # same as the number of frames within a chunk (i.e., m)
    feat_num = 1024  # number of LLDs features
    chunk_num = 11  # number of chunks splitted for a sentence (i.e., C)
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))
    #print(inputs.shape)
    encode = tf.keras.layers.LSTM(units=feat_num, activation='tanh', kernel_regularizer='l1', dropout=0.5, return_sequences=True)(inputs)
    encode = tf.keras.layers.LSTM(units=feat_num, activation='tanh', kernel_regularizer='l1', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    #print(encode.shape)
    # Uttr Attention Layer
    batch_atten_out = []
    for uttr_idx in range(0, batch_size * chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx + chunk_num
        #print(encode.shape)
        encode_crop = crop(0, _start, _end)(encode)
        #print(encode_crop.shape)
        encode_crop = reshape()(encode_crop)
        atten_out = atten(encode_crop)
        batch_atten_out.append(atten_out)
    # Output-Layer
    print(len(batch_atten_out))
    concat_atten_out = Concatenate(axis=0)(batch_atten_out)
    outputs = output_net(feat_num)(concat_atten_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)
    return model

def Rank_net(utteratten):
    time_step = 50
    feat_num = 1024
    #inputs = Input((None, 50, 256))
    #print(inputs.shape)
    inputs1 = Input((time_step, feat_num))
    inputs2 = Input((time_step, feat_num))
    #print(inputs1.shape)
    #print(inputs2.shape)
    #inputs1 = tf.reshape(inputs1, [None, 50, 256])
    #inputs2 = tf.reshape(inputs2, [None, 50, 256])
    first_out = utteratten(inputs1)
    second_out = utteratten(inputs2)
    diff = Subtract()([first_out, second_out])
    outputs = tf.keras.activations.sigmoid(diff)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
# Attention on LSTM chunk output => directly average without Attention
def UttrAtten_NonAtten():
    time_step = 62  # same as the number of frames within a chunk (i.e., m)
    feat_num = 130  # number of LLDs features
    chunk_num = 11  # number of chunks splitted for a sentence (i.e., C)
    # Input & LSTM Layer
    inputs = Input((time_step, feat_num))
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=True)(inputs)
    encode = LSTM(units=feat_num, activation='tanh', dropout=0.5, return_sequences=False)(encode)
    encode = BatchNormalization()(encode)
    # Uttr Attention Layer
    batch_out = []
    for uttr_idx in range(0, batch_size * chunk_num, chunk_num):
        _start = uttr_idx
        _end = uttr_idx + chunk_num
        encode_crop = crop(0, _start, _end)(encode)
        encode_crop = reshape()(encode_crop)
        encode_out = mean()(encode_crop)
        batch_out.append(encode_out)
    # Output-Layer
    concat_out = Concatenate(axis=0)(batch_out)
    outputs = output_net(feat_num)(concat_out)
    outputs = repeat()(outputs)  # for matching the input batch size
    model = Model(inputs=inputs, outputs=outputs)
    return model


###############################################################################

argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
#argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-atten", "--atten_type", required=True)
args = vars(argparse.parse_args())

# Parameters
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
#emo_attr = args['emo_attr']
atten_type = args['atten_type']

# Paths Setting
root_dir = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/MSP_1.10_w2v2LR/'
label_dir_train = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/Rank_net_Chunk-Level-Attention/all_labels/train_labs_val.npy'
label_dir_Validation = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/Rank_net_Chunk-Level-Attention/all_labels/dev_labs_val.npy'

params_train = {'batch_size': batch_size,
                'shuffle': True}

params_valid = {'batch_size': batch_size,
                'shuffle': False}

# Generators
training_generator = DataGenerator_w2v(root_dir, label_dir_train, **params_train)
validation_generator = DataGenerator_w2v(root_dir, label_dir_Validation, **params_valid)

# Optimizer
adam = tf.keras.optimizers.Adam(lr=0.00001)

# Model Saving Settings 
if os.path.exists('./Models'):
    pass
else:
    os.mkdir('./Models/')
filepath = './Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
    batch_size) + ']_' + atten_type + '_' + 'val' + '.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
time_callback = TimeHistory()
callbacks_list = [checkpoint, time_callback]

# Model Architecture
if atten_type == 'GatedVec':
    model = Rank_net(UttrAtten_GatedVec(atten_gated(feat_num=256, C=11)))
elif atten_type == 'RnnAttenVec':
    model = Rank_net(UttrAtten_AttenVec(atten_rnn(feat_num=1024, C=11)))
elif atten_type == 'SelfAttenVec':
    model = Rank_net(UttrAtten_AttenVec(atten_selfMH(ffeat_num=256, C=11)))
elif atten_type == 'NonAtten':
    model = Rank_net(UttrAtten_NonAtten())


#model.load_weights('./Models/LSTM_model[epoch2-batch128]_RnnAttenVec_dom_weights.h5')
print(model.summary())
plot_model(model, "./my_val_model_with_shape_info.png", show_shapes=True)


def show_shapes(): # can make yours to take inputs; this'll use local variable values
    print("Expected: (num_samples, timesteps, channels)")
    print("Sequences: {}".format(Sequences.shape))
    print("Targets:   {}".format(Targets.shape))

# Model Compile Settings
model.compile(optimizer=adam, loss="binary_crossentropy")
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=12,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list)

# Show training & validation loss
v_loss = model.history.history['val_loss']
t_loss = model.history.history['loss']
plt.plot(t_loss, 'b')
plt.plot(v_loss, 'r')
plt.savefig('./Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
    batch_size) + ']_' + atten_type + '_' + 'val' + '.png')
# Record training time cost per epoch
print('Epochs: ' + str(epochs) + ', ')
print('Batch_size: ' + str(batch_size) + ', ')
#print('Emotion: ' + emo_attr + ', ')
print('Chunk_type: dynamicOverlap, ')
print('Model_type: LSTM, ')
print('Atten_type: ' + atten_type + ', ')
print('Avg. Training Time(s/epoch): ' + str(np.mean(time_callback.times)) + ', ')
print('Std. Training Time(s/epoch): ' + str(np.std(time_callback.times)))

####### Saving Model Weights/Bias seperately due to different info-flow in the testing stage
model = None  # clean gpu-memory
if atten_type == 'SelfAttenVec':
    best_model = load_model(filepath, custom_objects={'LayerNormalization': LayerNormalization})
else:
    best_model = load_model(filepath)

# Saving trained model weights only
best_model.save_weights('./Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
    batch_size) + ']_' + atten_type + '_' + 'val' + '_weights.h5')
