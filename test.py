#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:12:12 2024

@author: ivo
"""

import numpy as np
import tensorflow as tf
import keras_tuner as tuner
from tensorflow import keras
from tensorflow.keras import layers
import os
import dill


class Model1(keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu", name='L11')
        self.dense2 = keras.layers.Dense(5, activation="relu", name='L12')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
    
            
class Model2(keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu", name='L21')
        self.dense2 = keras.layers.Dense(5, activation="relu", name='L22')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
        
class Model12(keras.Model):
    
    def __init__(self, model1, model2, **kwargs):
        super().__init__(**kwargs)
        self.model1 = model1 
        self.model2 = model2
        self.diags = {'epoch':tf.Variable(0, trainable=False)}
        
    @property 
    def metrics(self):
        return self.model1.metrics + self.model2.metrics
        
    
    def train_step(self, data):
        
        epoch = tf.math.mod(self.diags['epoch'],2)
        loss = tf.cond(tf.equal(epoch,0), 
                       lambda : self.model1.train_step(data), 
                       lambda : self.model2.train_step(data))
        
        return {
                'loss1':self.model1.loss_tracker.result(),
                'loss2':self.model2.loss_tracker.result(),
                'loss3':self.model2.loss_tracker.result(),
                'loss4':self.model2.loss_tracker.result(),
                'loss':loss['loss'],
                }
    
class Model12D1(keras.Model):
    
    def __init__(self, model1, model2, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss1')
        self.model1 = model1 
        self.model2 = model2
        
    @property 
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            loss  = tf.square(self.model1(data) - data) 
            loss += tf.square(self.model2(data) - self.model1(data))
            loss  = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)
            
        # Update weights
        grads = tape.gradient(loss, self.model1.trainable_weights)        
        self.optimizer.apply_gradients(zip(grads, 
                                       self.model1.trainable_weights))
            
        self.loss_tracker.update_state(loss)
            

        return {'loss':self.loss_tracker.result()}
    
class Model12D2(keras.Model):
    
    def __init__(self, model1, model2, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss2')
        self.model1 = model1 
        self.model2 = model2
        
    @property 
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            loss  = tf.square(self.model1(data) - data) 
            loss += tf.square(self.model2(data) - self.model1(data))
            loss  = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)
            
        # Update weights
        grads = tape.gradient(loss, self.model2.trainable_weights)        
        self.optimizer.apply_gradients(zip(grads, 
                                       self.model2.trainable_weights))
            
        self.loss_tracker.update_state(loss)
            
        return {'loss':self.loss_tracker.result()}
    
class DiagCallback(keras.callbacks.Callback):

    def __init__(self, **kwargs):
        self.diags = kwargs

    def on_epoch_begin(self, epoch, logs={}):
        if 'epoch' in self.diags:
            self.diags['epoch'].assign(epoch)

    def on_batch_begin(self, batch, logs={}):
        if 'batch' in self.diags:
            self.diags['batch'].assign(batch)
            
data = np.random.normal(size=(1000,5))
model1, model2 = Model1(), Model2()
model12D1 = Model12D1(model1, model2)
model12D2 = Model12D2(model1, model2)
model12D1.compile()
model12D2.compile()
model12 = Model12(model12D1,model12D2)
model12.compile(loss='loss',metrics=['loss1','loss2'])
model12.fit(data,epochs=10,callbacks=[DiagCallback(**model12.diags)])

    