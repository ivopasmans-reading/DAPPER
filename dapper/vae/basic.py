#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:12:21 2023

Basic Variational Autoencoder taken from https://keras.io/examples/generative/vae/

@author: ivo
"""
import numpy as np
import tensorflow as tf
import keras_tuner as tuner
from tensorflow import keras
from tensorflow.keras import layers

PI = tf.constant(np.pi)


#%% Simple example

def simple_example():
    #Dataset 
    def create_data(N, seed=1000):
        np.random.seed(seed)
        x=np.random.normal(size=(N,2))
        y = -np.hypot(x[:,0], 2*x[:,1])
        return x, y
        
    
    #NN
    def create_model(dim_in):
        L_in = layers.Input(shape=(dim_in,), name='L_in')
        x = layers.BatchNormalization()(L_in)
        #x=L_in
        for l in range(3):
            x = layers.Dense(16, activation='relu', name='L_hidden'+str(l))(x)
            #x = layers.BatchNormalization()(x)
        L_out = layers.Dense(1, name='L_out')(x)
        
        model = keras.Model(L_in,L_out)
        model.compile(optimizer=keras.optimizers.SGD(),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError(),
                               keras.metrics.MeanAbsoluteError()]
                      )
        
        return model
       
    k = 100
    X,Y = create_data(1000) 
    model = create_model(2)
    model.fit(X[:k],Y[:k], epochs=100, validation_data=(X[k:],Y[k:]))

    
#%% Simple dense network. 

#Stopping 
stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                           patience=5, verbose=True,
                                           restore_best_weights=True,
                                           min_delta=0.05) 

class SamplingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  
        
class VAE(keras.Model):
    """ Variational autoencoder model. """
    
    def __init__(self, encoder, decoder, mc_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.mc_samples = mc_samples
        self.encoder = encoder
        self.decoder = decoder
        
        self.l2_rotation = 1.0e-2
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.angle_loss_tracker = keras.metrics.Mean(name="angle_loss")

    @property
    def metrics(self):
        metrics = [self.total_loss_tracker,
                   self.reconstruction_loss_tracker,
                   self.kl_loss_tracker,
                   self.angle_loss_tracker]
        return metrics
    
    def reconstruction_loss(self, data):
        _, _, z = self.encoder(data)
        x_mean, _, _ = self.decoder(z)
        loss = tf.square(data - x_mean)
        return 0.5 * tf.reduce_sum(loss, axis=1)
    
    def mc_reconstruction_loss(self, data, z):
        x_mean, x_log_var, x_sin  = self.decoder(z)  
        x_cos = tf.sqrt(1 - x_sin**2)
        #Turn error into its principal component
        error   = data - x_mean
        error0  = x_cos[:,0:1] * error[:,0:1] - x_sin[:,0:1] * error[:,1:2] 
        error1  = x_sin[:,0:1] * error[:,0:1] + x_cos[:,0:1] * error[:,1:2]
        error12 = tf.keras.layers.Concatenate(axis=-1)([error0, error1])
        #Penalty large variances
        loss  = tf.reduce_sum(x_log_var, axis=1)
        #Penalty errors in principal components frame. 
        loss  += tf.reduce_sum(tf.square(error12) / tf.exp(x_log_var), axis=-1)
        return 0.5 * loss
    
    def mc_angle_loss(self, data, z):
        _, _, x_sin  = self.decoder(z)  
        #L2 regularization term for angles. 
        loss  = tf.reduce_sum(tf.square(x_sin), axis=-1)
        return 0.5 * self.l2_rotation * loss
        
    def kl_loss(self, data):
        #Dimension 
        k = data.shape[1]
        #KL loss
        z_mean, z_log_var, _ = self.encoder(data)
        #Trace Sigma + log 1/det(Sigma) - dim + ||mu-0||**2
        loss = tf.exp(z_log_var) - z_log_var - k + tf.square(z_mean)
        return 0.5 * tf.reduce_sum(loss, axis=1)
        
    def mc_kl_loss(self, data, z):
        z_mean, z_log_var, _ = self.encoder(data)
        loss  = tf.square(z)
        loss -= tf.square(z - z_mean) / tf.exp(z_log_var)
        loss -= z_log_var
        return 0.5 * tf.reduce_sum(loss, axis=1)

    def train_step(self, data):
        
        #Create cost function.
        with tf.GradientTape() as tape:
            kl_loss, reconstruction_loss, angle_loss = 0, 0, 0
            kl_loss = self.kl_loss(data)
            for _ in range(self.mc_samples):
                _, _, z = self.encoder(data)
                reconstruction_loss += self.mc_reconstruction_loss(data, z) * (1/self.mc_samples)
                angle_loss += self.mc_angle_loss(data, z) * (1/self.mc_samples)
        
            kl_loss = tf.reduce_mean(kl_loss)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            angle_loss = tf.reduce_mean(angle_loss)
            total_loss = reconstruction_loss + kl_loss + angle_loss
            
        #Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        #Update losses.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        
        #Output losses.
        losses = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "angle_loss": self.angle_loss_tracker.result(),
            }        
        
        return losses
    
class DenseVae(tuner.HyperModel):
    """ Creates encoders, decoders using dense neural networks. """
    
    def build(self, hp):
        #Set hyperparameters.
        self._build_default_hp(hp)
        
        #Build decoder/encoder-pair
        state_dim = hp.Fixed('state_dim', 2)
        latent_dim = hp.Fixed('latent_dim', 2)
        encoder = self._build_encoder(state_dim, latent_dim)
        decoder = self._build_decoder(state_dim, latent_dim)
        
        #Build actual model. 
        model = VAE(encoder, decoder, mc_samples=self.hp.get('mc_samples'))
        
        #Build learning function.
        def lr_func(epoch):
            return self.hp.get('lr_init') * 0.1**(2*epoch / self.hp.get('epochs'))
        self.lr = tf.keras.callbacks.LearningRateScheduler(lr_func)
        
        self.lr = tf.keras.callbacks.ReduceLROnPlateau("reconstruction_loss",
                                                       factor=0.5, patience=2,
                                                       min_delta=.5, mode='min',
                                                       verbose=True)
        #lr = tf.keras.optimizers.schedules.PolynomialDecay(2e-3, steps, power=power)
        
        #Compile before use and return. 
        lr = self.hp.get('lr_init')
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr))
        return model
    
    def fit(self, hp, model, *args, **kwargs):        
        fit_args = {'epochs':hp.get('epochs'), 
                    'batch_size':hp.get('batch_size'),
                    'shuffle':True,
                    'callbacks':[],
                    **kwargs
                    }
        fit_args['callbacks'] = fit_args['callbacks'] + [stopper, self.lr]
        return model.fit(*args, **fit_args)
    
    def _build_default_hp(self, hp):
        self.hp = hp
        self.hp.Fixed('epochs', 40)
        self.hp.Int( 'batch_size', default=64, min_value=1, max_value=1024, sampling='log')
        self.hp.Int('mc_samples', min_value=1, max_value=20, step=1)
        self.hp.Float('lr_init', default=1e-2, min_value=1e-3, max_value=1e-2, step=1e-3)
        self.hp.Int('no_layers', default=4, min_value=0, max_value=8, step=1)
        self.hp.Int('no_nodes', default=50, min_value=2, max_value=1024, sampling='log')
        self.hp.Boolean('use_rotation', default=False)
        self.hp.Float('l2_rotation', default=1.0e-4, min_value=0.0, max_value=1.0e-3, step=1.0e-4)
        
        step = 1e-2 / self.hp.get('no_nodes')
        self.hp.Float('l1', min_value=0., max_value=10 * step, step=step)
    
    def _build_network(self, input_layer):
        no_layers = self.hp.get('no_layers')
        no_nodes  = self.hp.get('no_nodes')
        
        x = layers.BatchNormalization()(input_layer)
        for l in range(no_layers):
            x = layers.Dense(no_nodes,
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.L1(self.hp.get('l1'))
                             )(x)
            x = layers.LeakyReLU(0.3)(x)  
            x = layers.BatchNormalization()(x)
            
        return x
    
    def _build_encoder(self, state_dim, latent_dim):    
        """ Build the encoder. """
        input_layer = keras.Input(shape=(state_dim,))
        x = self._build_network(input_layer)
        
        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z         = SamplingLayer(name='z_sample')([z_mean, z_log_var])
        
        encoder = keras.Model(input_layer, [z_mean, z_log_var, z], 
                              name="encoder")
        return encoder
        
    def _build_decoder(self, state_dim, latent_dim):
        """ Build the decoder. """
        input_layer = keras.Input(shape=(latent_dim,))
        x = self._build_network(input_layer)
          
        x_mean    = layers.Dense(state_dim  , name="x_mean")(x)
        x_log_var = layers.Dense(state_dim  , name="x_log_var")(x)   
        
        if self.hp.get('use_rotation') is True:
            x_sin = layers.Dense(state_dim-1, name="x_sin", activation='tanh',
                                 kernel_initializer='zeros', 
                                 bias_initializer='zeros')(x)
        else:
            x_sin = 0.0 * x_mean[:,0:1]
        
        decoder = keras.Model(input_layer, [x_mean, x_log_var, x_sin], 
                              name="decoder")
        return decoder

#%% Tuning 

hp = tuner.HyperParameters()

def tune_DenseVae(x, trials=20):
    hypermodel = DenseVae()   
    file_dir = '/home/ivo/Code/VAE/tmp'
    
    #Writer logs 
    tensorboard_writer = tf.keras.callbacks.TensorBoard(file_dir)

    #Tune layers/nodes 
    hp = tuner.HyperParameters()
    hp.Int("no_layers", min_value=0, max_value=8, step=1)
    hp.Int("no_nodes", min_value=4, max_value=256, sampling='log')
    hp.Int('mc_samples', min_value=1, max_value=20, step=1)
    hp.Float('lr_init',  default=5e-3, min_value=1e-3, max_value=5e-3, step=1e-3)

    architecture = tuner.Hyperband(hypermodel=hypermodel,
                                              objective=tuner.Objective('loss', direction='min'),
                                              hyperparameters=hp, 
                                              tune_new_entries=False, 
                                              #max_trials=trials,
                                              max_epochs = 400,
                                              directory=file_dir,
                                              overwrite=True,
                                              hyperband_iterations=3)
    #Carry out the search
    architecture.search(x,callbacks=[tensorboard_writer])
