#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:41:12 2020

@author: winston
"""
from keras.models import Model
from keras.initializers import glorot_normal, Zeros, Ones
from keras.layers import Dense, Input, Activation, Dropout, Concatenate, Reshape, LocallyConnected1D, Multiply, GaussianNoise
from keras.optimizers import Nadam
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.engine.topology import Layer


def cc_coef(y_true, y_pred):
    mu_y_true = K.mean(y_true)
    mu_y_pred = K.mean(y_pred)                                                                                                                                                                                              
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))

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

def ladder_network_multi(num_nodes, noise, alpha, beta, gamma, seeding):
    def reconstruction_loss(true_val, predicted_val):
        loss = 1 * K.mean(K.square(true_val - predicted_val)) + 1 * K.mean(K.square(z1 - z_bn_1)) + \
               1 * K.mean(K.square(z2 - z_bn_2)) + 0.1 * alpha * K.mean(K.square(z_3_1 - z_bn_3_1)) + \
               0.1 * beta * K.mean(K.square(z_3_2 - z_bn_3_2)) + 0.1 * gamma * K.mean(K.square(z_3_3 - z_bn_3_3))
        return loss

    inputs = Input(shape=(6373,), dtype="float32", name="input_unuspervised")
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

    W_3_2 = Dense(1, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)
    nonlinear_3_2 = BiasLayer(num_nodes, activation='linear')
    z_tilda_3_2, h_tilda_3_2, m_tilda_3_2, v_tilda_3_2 = encoder_layer(h_tilda_2, 0.3, W_3_2, nonlinear_3_2)

    W_3_3 = Dense(1, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)
    nonlinear_3_3 = BiasLayer(num_nodes, activation='linear')
    z_tilda_3_3, h_tilda_3_3, m_tilda_3_3, v_tilda_3_3 = encoder_layer(h_tilda_2, 0.3, W_3_3, nonlinear_3_3)

    # clean encoder
    z1, h1, mean_1, std_1 = encoder_layer(dropout0(inputs), 0, W1, nonlinear1)
    h1 = dropout1(h1)
    z2, h2, mean_2, std_2 = encoder_layer(h1, 0, W2, nonlinear2)
    z_3_1, h_3_1, mean_3_1, std_3_1 = encoder_layer(h2, 0, W_3_1, nonlinear_3_1)
    z_3_2, h_3_2, mean_3_2, std_3_2 = encoder_layer(h2, 0, W_3_2, nonlinear_3_2)
    z_3_3, h_3_3, mean_3_3, std_3_3 = encoder_layer(h2, 0, W_3_3, nonlinear_3_3)

    # decoder
    z_hat_3_1 = Dense(1, kernel_initializer=Ones(), bias_initializer=Zeros())(z_tilda_3_1)
    z_bn_3_1 = BatchNorm(mean_3_1, std_3_1)(z_hat_3_1)
    z_hat_3_2 = Dense(1, kernel_initializer=Ones(), bias_initializer=Zeros())(z_tilda_3_2)
    z_bn_3_2 = BatchNorm(mean_3_2, std_3_2)(z_hat_3_2)
    z_hat_3_3 = Dense(1, kernel_initializer=Ones(), bias_initializer=Zeros())(z_tilda_3_3)
    z_bn_3_3 = BatchNorm(mean_3_3, std_3_3)(z_hat_3_3)

    z_hat_3 = Concatenate()([z_hat_3_1, z_hat_3_2, z_hat_3_3])
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

    u0 = Dense(6373, kernel_initializer=glorot_normal(seed=seeding), use_bias=False)(z_hat_1)
    u0 = BatchNormalization(center=False, scale=False, epsilon=1e-10)(u0)
    z_hat_0 = DecoderLayer(6373, 0)([z_tilda_0, u0])

    adam1 = Nadam(lr=0.0001)
    adam2 = Nadam(lr=0.0001)
    model = Model(inputs=inputs, outputs=[z_hat_0, h_tilda_3_1, h_tilda_3_2, h_tilda_3_3, h_3_1, h_3_2, h_3_3])
    model.compile(optimizer=adam1,
                  loss=[reconstruction_loss, cc_coef, cc_coef, cc_coef, None, None, None],
                  loss_weights=[1, 0.1 * alpha, 0.1 * beta, 0.1 * gamma, 1, 1, 1])
    unsupervised_model = Model(inputs=inputs, outputs=z_hat_0)
    unsupervised_model.compile(optimizer=adam2, loss=reconstruction_loss)
    return model, unsupervised_model
