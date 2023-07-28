#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
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
from utils import cc_coef, MTL_loss, cc_ladder_loss
import time
import argparse
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.initializers import glorot_normal, Zeros, Ones
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Concatenate, Reshape, LocallyConnected1D, Multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
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

def Rank_net(prediction_model):
    feat_num = 1024
    label_num = 1
    inputs1 = Input((feat_num))
    inputs2 = Input((feat_num))
    
    #first_out, first_abs = prediction_model(inputs1)
    #second_out, second_abs = prediction_model(inputs2)
    first, z_hat_0_1, z1_1, z2_1, z_bn_1_1, z_bn_2_1 = prediction_model(inputs1)
    second, z_hat_0_2, z1_2, z2_2, z_bn_1_2, z_bn_2_2 = prediction_model(inputs2)
    outputs1 = reconstruction_loss(inputs1,[z_hat_0_1, z1_1, z2_1, z_bn_1_1, z_bn_2_1])
    outputs2 = reconstruction_loss(inputs2,[z_hat_0_2, z1_2, z2_2, z_bn_1_2, z_bn_2_2])
    # Absolute score
    
    
    # Rank 
    diff = Subtract()([first, second])
    outputs = tf.keras.activations.sigmoid(diff)
    #outputs = tf.squeeze(outputs)
    model = Model(inputs=[inputs1, inputs2], outputs=[outputs,outputs1,outputs2])
    return model
    

class AddNoiseGaussian(Layer):

    def __init__(self, noise,  **kwargs):
        self.stddev = noise
        super(AddNoiseGaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(AddNoiseGaussian, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        output = x + K.random_normal(shape=K.shape(x),
                                            mean=0.,
                                            stddev=self.stddev, seed=1)
        return output

class BiasLayer(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.activation == 'relu':
            self.beta = self.add_weight(name='beta',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)
        super(BiasLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.activation == 'relu':
            output = (x + self.beta)
            output = Activation(self.activation)(output)
        else:
            output = Activation(self.activation)(x)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

class DecoderLayer(Layer):

    def __init__(self, output_dim, last_layer, **kwargs):
        self.output_dim = output_dim
        self.last_layer = last_layer
        super(DecoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a1 = self.add_weight(name='kernel1',
                                  shape=(self.output_dim,),
                                  initializer=Zeros(),
                                  trainable=True)
        self.a2 = self.add_weight(name='kernel2',
                                  shape=(self.output_dim,),
                                  initializer=Ones(),
                                  trainable=True)
        self.a3 = self.add_weight(name='kernel3',
                                       shape=(self.output_dim,),
                                       initializer=Zeros(),
                                       trainable=True)
        self.a4 = self.add_weight(name='kernel4',
                                     shape=(self.output_dim,),
                                     initializer=Zeros(),
                                     trainable=True)
        self.a5 = self.add_weight(name='kernel5',
                                       shape=(self.output_dim, ),
                                       initializer=Zeros(),
                                       trainable=True)
        self.a6 = self.add_weight(name='kernel6',
                                  shape=(self.output_dim,),
                                  initializer=Zeros(),
                                  trainable=True)
        self.a7 = self.add_weight(name='kernel7',
                                  shape=(self.output_dim,),
                                  initializer=Ones(),
                                  trainable=True)
        self.a8 = self.add_weight(name='kernel8',
                                  shape=(self.output_dim,),
                                  initializer=Zeros(),
                                  trainable=True)
        self.a9 = self.add_weight(name='kernel9',
                                  shape=(self.output_dim,),
                                  initializer=Zeros(),
                                  trainable=True)
        self.a10 = self.add_weight(name='kernel10',
                                  shape=(self.output_dim,),
                                  initializer=Zeros(),
                                  trainable=True)
        if self.last_layer:
            self.u = self.add_weight(name='kernel11',
                                  shape=(self.output_dim,),
                                  initializer=Zeros(),
                                  trainable=False)
        super(DecoderLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.last_layer:
            mu = self.a1 * K.sigmoid(self.a2 * self.u + self.a3) + self.a4 * self.u + self.a5
            v = K.sigmoid(self.a6 * K.sigmoid(self.a7 * self.u + self.a8) + self.a9 * self.u + self.a10)
            output = (x - mu) * v + mu
        else:
            mu = self.a1 * K.sigmoid(self.a2 * x[1] + self.a3) + self.a4 * x[1] + self.a5
            v = K.sigmoid(self.a6 * K.sigmoid(self.a7 * x[1] + self.a8) + self.a9 * x[1] + self.a10)
            output = (x[0] - mu) * v + mu
        return output

class BatchNorm(Layer):
    def __init__(self,  mean_val, std_val, **kwargs):
        self.is_placeholder = True
        self.mean_val = mean_val
        self.std_val = std_val
        super(BatchNorm, self).__init__(**kwargs)

    def call(self, inputs):
        return (inputs - self.mean_val)/self.std_val

def encoder_layer(layer_input, noise, W, nonlinear):
    z_pre = W(layer_input)
    m = K.mean(z_pre, axis=0)
    v = K.std(z_pre, axis=0)
    z = BatchNormalization(center=False, scale=False, epsilon=1e-10)(z_pre)
    if noise:
        z = GaussianNoise(noise)(z)
    h = nonlinear(z)
    return z, h, m, v
    
def reconstruction_loss(true_val, predicted_val):
        loss = 1 * K.mean(K.square(true_val - predicted_val[0])) + 1 * K.mean(K.square(predicted_val[1] - predicted_val[3])) + 1 * K.mean(K.square(predicted_val[2] - predicted_val[4]))
        return loss

def ladder_network(num_nodes, noise, seeding):
    

    inputs = Input(shape=(1024,), dtype="float32", name="input_unuspervised")
    dropout0 = Dropout(0.1)
    dropout1 = Dropout(0.1)
    z_tilda_0 = inputs
    # noisy encoder
    z_tilda_0 = GaussianNoise(noise)(z_tilda_0)
    h_tilda_0 = dropout0(z_tilda_0)
    W1 = Dense(num_nodes, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)
    nonlinear1 = BiasLayer(num_nodes, activation='relu')
    z_tilda_1, h_tilda_1, m_tilda_1, v_tilda_1 = encoder_layer(h_tilda_0, 0.3, W1, nonlinear1)

    h_tilda_1 = dropout1(h_tilda_1)
    W2 = Dense(num_nodes, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)
    nonlinear2 = BiasLayer(num_nodes, activation='relu')
    z_tilda_2, h_tilda_2, m_tilda_2, v_tilda_2 = encoder_layer(h_tilda_1, 0.3, W2, nonlinear2)

    W_3_1 = Dense(1, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)
    nonlinear_3_1 = BiasLayer(num_nodes, activation='linear')
    z_tilda_3_1, h_tilda_3_1, m_tilda_3_1, v_tilda_3_1 = encoder_layer(h_tilda_2, 0.3, W_3_1, nonlinear_3_1)

    

    # clean encoder
    z1, h1, mean_1, std_1 = encoder_layer(dropout0(inputs), 0, W1, nonlinear1)
    h1 = dropout1(h1)
    z2, h2, mean_2, std_2 = encoder_layer(h1, 0, W2, nonlinear2)
    z_3_1, h_3_1, mean_3_1, std_3_1 = encoder_layer(h2, 0, W_3_1, nonlinear_3_1)
    

    # decoder
    z_hat_3 = Dense(1, kernel_initializer=Ones(), bias_initializer=Zeros())(z_tilda_3_1)
    

    #z_hat_3 = Concatenate()([z_hat_3_1, z_hat_3_2, z_hat_3_3])
    u2 = Dense(num_nodes, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)(z_hat_3)
    u2 = BatchNormalization(center=False, scale=False, epsilon=1e-10)(u2)
    z_tilda_2_reshape = Reshape((num_nodes, 1))(z_tilda_2)
    u2_reshape = Reshape((num_nodes, 1))(u2)
    z_hat_2_reshape = Concatenate()([z_tilda_2_reshape, u2_reshape, Multiply()([z_tilda_2_reshape, u2_reshape])])
    z_hat_2_reshape = LocallyConnected1D(4, 1, activation='relu', kernel_initializer=glorot_normal(seed=seeding), input_shape=(num_nodes, 3))(z_hat_2_reshape)
    z_hat_2_reshape = LocallyConnected1D(1, 1, activation='linear', input_shape=(num_nodes, 4))(z_hat_2_reshape)
    z_hat_2 = Reshape((num_nodes,))(z_hat_2_reshape)
    z_bn_2 = BatchNorm(mean_2, std_2)(z_hat_2)

    u1 = Dense(num_nodes, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)(z_hat_2)
    u1 = BatchNormalization(center=False, scale=False, epsilon=1e-10)(u1)
    z_tilda_1_reshape = Reshape((num_nodes, 1))(z_tilda_1)
    u1_reshape = Reshape((num_nodes, 1))(u1)
    z_hat_1_reshape = Concatenate()([z_tilda_1_reshape, u1_reshape, Multiply()([z_tilda_1_reshape, u1_reshape])])
    z_hat_1_reshape = LocallyConnected1D(4, 1, activation='relu', kernel_initializer=glorot_normal(seed=seeding), input_shape=(num_nodes, 3))(z_hat_1_reshape)
    z_hat_1_reshape = LocallyConnected1D(1, 1, activation='linear', input_shape=(num_nodes, 4))(z_hat_1_reshape)
    z_hat_1 = Reshape((num_nodes,))(z_hat_1_reshape)
    z_bn_1 = BatchNorm(mean_1, std_1)(z_hat_1)

    u0 = Dense(1024, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)(z_hat_1)
    u0 = BatchNormalization(center=False, scale=False, epsilon=1e-10)(u0)
    z_hat_0 = DecoderLayer(1024, 0)([z_tilda_0, u0])

    #adam1 = Nadam(lr=0.0001)
    #adam2 = Nadam(lr=0.0001)
    model = Model(inputs=inputs, outputs=[h_tilda_3_1, z_hat_0, z1, z2, z_bn_1, z_bn_2])
    #model.compile(optimizer=adam1,
    #              loss=[reconstruction_loss, cc_coef, cc_coef, cc_coef, None, None, None],
    #              loss_weights=[1, 0.1 * alpha, 0.1 * beta, 0.1 * gamma, 1, 1, 1])
    #unsupervised_model = Model(inputs=inputs, outputs=z_hat_0)
    #unsupervised_model.compile(optimizer=adam2, loss=reconstruction_loss)
    return model#, unsupervised_model


###############################################################################

argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", default=1 )#, required=True)
argparse.add_argument("-batch", "--batch_size", default=32)#, required=True)
argparse.add_argument("-lr", "--learning_rate", default=0.000005)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-atten", "--atten_type", default="RnnAttenVec")#, required=True)
args = vars(argparse.parse_args())

# Parameters
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']
atten_type = args['atten_type']


# Paths Setting
root_dir = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/MSP_1.10_w2v2LR/'
label_dir_train = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/Rank_net_Chunk-Level-Attention/all_labels/train_labs_aro.npy'
label_dir_Validation = '/home/abinay/Documents/Projects/LAS/MSP_Podcast_1.9/Rank_net_Chunk-Level-Attention/all_labels/dev_labs_aro.npy'


params_train = {'batch_size': batch_size,
                'shuffle': True}

params_valid = {'batch_size': batch_size,
                'shuffle': False}

# Generators
training_generator = DataGenerator_w2v(root_dir, label_dir_train, **params_train)
validation_generator = DataGenerator_w2v(root_dir, label_dir_Validation, **params_valid)

# Optimizer
learn_rate = float(args['learning_rate'])
adam = tf.keras.optimizers.Adam(lr = learn_rate)

# Model Saving Settings 
if os.path.exists('./Models'):
    pass
else:
    os.mkdir('./Models/')
    #save model with args input
#filepath = './Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
#    batch_size) + '-lr' + str(learn_rate) + ']_' + atten_type + '_' + 'val' + '.hdf5'

#save specific model
filepath = './Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
    batch_size) + '-lr' + str(learn_rate) + ']_' + atten_type + '_' + emo_attr + '.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq = 'epoch')
time_callback = TimeHistory()
callbacks_list = [checkpoint, time_callback]

# Model Architecture
if atten_type == 'GatedVec':
    model = Rank_net(UttrAtten_GatedVec(atten_gated(feat_num=256, C=11)))
elif atten_type == 'RnnAttenVec':
   # model = Rank_net(UttrAtten_AttenVec(atten_rnn(feat_num=1024, C=11)))
    model = Rank_net(ladder_network(num_nodes=256, noise=0.3, seeding=0))
elif atten_type == 'SelfAttenVec':
    model = Rank_net(UttrAtten_AttenVec(atten_selfMH(ffeat_num=256, C=11)))
elif atten_type == 'NonAtten':
    model = Rank_net(MTL_FC_layer())


#model.load_weights('./Models/LSTM_model[epoch3-batch128-lr5e-05]_RnnAttenVec_val_weights.h5')
print(model.summary())
#plot_model(model, "./my_val_model_with_shape_info.png", show_shapes=True)


def show_shapes(): # can make yours to take inputs; this'll use local variable values
    print("Expected: (num_samples, timesteps, channels)")
    print("Sequences: {}".format(Sequences.shape))
    print("Targets:   {}".format(Targets.shape))

# Model Compile Settings
model.compile(optimizer=adam, loss= cc_ladder_loss)
#model.compile(optimizer=adam, loss="binary_crossentropy")
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=4,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list)

# Show training & validation loss
v_loss = model.history.history['val_loss']
t_loss = model.history.history['loss']
#plt.plot(t_loss, 'b')
plt.plot(v_loss, 'r')
with open('./Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
    batch_size) + '-lr' + str(learn_rate) + ']_' + atten_type + '_' + emo_attr  + '_v_loss.txt', 'w') as wf:
    for elem in v_loss:
        wf.write(str(elem) + '\n')
        
wf.close()
    
plt.savefig('./Models/LSTM_model[epoch' + str(epochs) + '-batch' + str(
    batch_size) + '-lr' + str(learn_rate) + ']_' + atten_type + '_' + emo_attr + '.png')
# Record training time cost per epoch
print('Epochs: ' + str(epochs) + ', ')
print('Batch_size: ' + str(batch_size) + ', ')
print('Emotion: ' + emo_attr + ', ')
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
    batch_size) + '-lr' + str(learn_rate) + ']_' + atten_type + '_' + emo_attr + '_weights.h5')
