#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:27:48 2024

@author: ivo
"""

import numpy as np
import tensorflow as tf
import keras_tuner as tuner
from tensorflow import keras
from tensorflow.keras import layers
import os
import dill

class SamplingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_idem():
    input = keras.Input(shape=(10,))
    x = keras.layers.BatchNormalization(name='norm_idem',
                                        scale=False,center=False)(input)
    output = keras.layers.Dense(10,name='idem_mean')(x)
    return keras.Model(input,output,name='idem')

def build_encoder():
    input = keras.Input(shape=(10,))
    x = keras.layers.BatchNormalization(name='norm_encoder',
                                        center=False, scale=False)(input, training=True)
    #x = input
    zmean = keras.layers.Dense(10,name='encoder_mean')(x)
    zvar  = keras.layers.Dense(10,name='encoder_var')(x)
    zsample = SamplingLayer(name='encoder_sample')((zmean,zvar))
    return keras.Model(input,(zmean,zvar,zsample),name='encoder')

def build_decoder():
    input = keras.Input(shape=(10,))
    #x = keras.layers.BatchNormalization(name='norm_decoder')(input)
    x = input
    xmean = keras.layers.Dense(10,name='decoder_mean')(x)
    xvar  = keras.layers.Dense(10,name='decoder_var')(x)
    return keras.Model(input,(xmean,xvar),name='decoder')

class VAE(keras.Model):
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.loss_tracker = keras.metrics.Mean(name='total_loss')
        
    def reconstruction_loss(self, data, z):
        x_mean, x_var = self.decoder(z)
        error = data - x_mean 
        loss  = x_var 
        loss += tf.square(error) / (tf.exp(x_var) + tf.constant(1e-6))
        return 0.5 * tf.reduce_sum(loss, axis=-1)
        
    def kl_loss(self, data, z):
        z_mean, z_log_var, _ = self.encoder(data)
        k = data.shape[1]
        loss = -z_log_var - k + tf.square(z_mean) + tf.exp(z_log_var)
        return 0.5 * tf.reduce_sum(loss, axis=-1)
    
    def loss(self,data,z):
        return self.reconstruction_loss(data, z) + self.kl_loss(data, z)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            r_loss, kl_loss = 0, 0
            _, _, z = self.encoder(data)
            kl_loss = self.kl_loss(data,z)
            r_loss = self.reconstruction_loss(data, z)
            total_loss = kl_loss + r_loss
            total_loss = tf.reduce_mean(total_loss)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        
        return {'total_loss':self.loss_tracker.result()}

xx = tf.random.normal(shape=(10000, 10)) + tf.constant(2.) 
yy = tf.constant(2.)*xx + tf.constant(4.)

#%%

idem = build_idem()
opt = keras.optimizers.Adam()
loss = keras.losses.MSE
idem.compile(optimizer=opt,loss=loss,metrics=[keras.metrics.MSE])
idem.fit(xx, yy, batch_size=64, epochs=100)

#%%

encoder = build_encoder()
decoder = build_decoder()
opt = keras.optimizers.Adam()
vae = VAE(encoder, decoder)
vae.compile(optimizer=opt)
vae.fit(xx, batch_size=64, epochs=100)

_,_,z = encoder(xx[10:30])
xmean, xvar = decoder(z)
xxz = tf.random.normal(shape=xmean.shape)*tf.exp(0.5*xvar) + xmean
print(np.mean(xxz.numpy(),axis=0), np.std(xxz.numpy(),axis=0))

