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

import os, dill

PI = tf.constant(np.pi)

#%% Variational autoencoder based on dense neural network. 

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
    
    def __init__(self, encoder, decoder, mc_samples=1, l2_rotation=1.0e-2, 
                 **kwargs):
        super().__init__(**kwargs)
        self.mc_samples = mc_samples
        self.encoder = encoder
        self.decoder = decoder
        
        self.l2_rotation = l2_rotation 
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
        """ Reconstruction loss estimate used by others. """
        _, _, z = self.encoder(data)
        x_mean, _, _ = self.decoder(z)
        loss = tf.square(data - x_mean)
        return 0.5 * tf.reduce_sum(loss, axis=1)
    
    def mc_reconstruction_loss(self, data, z):
        """ Reconstruction loss estimated from Monte-Carlo approximation. """
        x_mean, x_log_var, x_sin  = self.decoder(z)  
        x_cos = tf.sqrt(1 - x_sin**2)
        #Turn error into its principal component
        error   = data - x_mean
        error0  = x_cos[:,0:1] * error[:,0:1] + x_sin[:,0:1] * error[:,1:2] 
        error1  = x_sin[:,0:1] * error[:,0:1] - x_cos[:,0:1] * error[:,1:2]
        error12 = tf.keras.layers.Concatenate(axis=-1)([error0, error1])
        #-2 log p(z|x)
        loss  = x_log_var
        loss += tf.square(error12) / tf.exp(x_log_var)
        return 0.5 * tf.reduce_sum(loss, axis=-1)
    
    def mc_angle_loss(self, data, z):
        """ Regularization term to keep polar angle 1st principal component
        small. """
        _, _, x_sin  = self.decoder(z)  
        #L2 regularization term for angles. 
        loss  = tf.reduce_sum(tf.square(x_sin), axis=-1)
        return 0.5 * self.l2_rotation * loss
        
    def kl_loss(self, data):
        """ Exact Kullbeck-Leibler divergence for Gaussian and normal 
        distribution. """ 
        #Dimension 
        k = data.shape[1]
        #KL loss
        z_mean, z_log_var, _ = self.encoder(data)
        #Trace Sigma + log 1/det(Sigma) - dim + ||mu-0||**2
        KL = -z_log_var - k + tf.square(z_mean) + tf.exp(z_log_var)
        loss = 0.5 * tf.reduce_sum(KL, axis=1)
        return loss
        
    def mc_kl_loss(self, data, z):
        """ 
        Kullbeck-Leibler divergence between Gaussian and normal distribution 
        calculated based on Monte-Carlo approximation. 
        """
        z_mean, z_log_var, _ = self.encoder(data)
        #log p(z|x)
        loss  = -0.5 * tf.square(z - z_mean) / tf.exp(z_log_var)
        loss -= 0.5 * z_log_var
        #log 1/p(z)
        loss += 0.5 * tf.square(z) 
        return tf.reduce_sum(loss, axis=1)

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
        self.hp = self._build_default_hp(hp)
        
        #Build decoder/encoder-pair
        encoder = self._build_encoder(hp.get('state_dim'), hp.get('latent_dim'))
        decoder = self._build_decoder(hp.get('state_dim'), hp.get('latent_dim'))
        
        #Build actual model. 
        self.model = VAE(encoder, decoder, mc_samples=self.hp.get('mc_samples'),
                         l2_rotation=self.hp.get('l2_rotation'))
        
        #Build learning function.
        self.lr = tf.keras.callbacks.ReduceLROnPlateau("reconstruction_loss",
                                                       factor=0.5, patience=2,
                                                       min_delta=.5, mode='min',
                                                       verbose=True)
        
        #Build stopper 
        self.stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                        patience=5, verbose=True,
                                                        restore_best_weights=True,
                                                        min_delta=0.01) 
        
        #Compile before use and return. 
        lr = self.hp.get('lr_init')
        self.model.compile(optimizer = keras.optimizers.Adam(learning_rate=lr))
        return self.model
    
    def fit(self, hp, model, *args, **kwargs):  
        fit_args = {'epochs':hp.get('epochs'), 
                    'batch_size':hp.get('batch_size'),
                    'shuffle':True,
                    'callbacks':[],
                    **kwargs
                    }
        fit_args['callbacks'] = fit_args['callbacks'] + [self.stopper, self.lr]
        return model.fit(*args, **fit_args)
    
    def _file_paths(self, file_path):
        return {'dir'     : os.path.split(file_path)[0],
                'name'    : os.path.split(file_path)[1],
                'encoder' : file_path + '_encoder.tf',
                'decoder' : file_path + '_decoder.tf',
                'hp'      : file_path + '_hp.pkl'}
    
    def save(self, file_path):
        paths = self._file_paths(file_path)
        print(paths)
        if not os.path.exists(paths['dir']):
            os.mkdir(paths['dir'])
            
        self.model.encoder.save(paths['encoder'], save_format='tf')
        self.model.decoder.save(paths['decoder'], save_format='tf')
        with open(paths['hp'], 'bw') as stream:
            dill.dump(self.hp.get_config(), stream)
            
    def load(self, file_path):
        paths = self._file_paths(file_path)
        for key in ['encoder','decoder','hp']:
            if not os.path.exists(paths[key]):
                msg = "File {:s} missing.".format(paths[key])
                raise FileNotFoundError(msg)
                
        encoder = tf.keras.saving.load_model(paths['encoder'])
        decoder = tf.keras.saving.load_model(paths['decoder'])
        with open(paths['hp'], 'rb') as stream:
            hp = dill.load(stream)
            hp = hp['values']            
        
        model = VAE(encoder, decoder, mc_samples=hp['mc_samples'],
                    l2_rotation=hp['l2_rotation'])
        
        return model, hp
        
    
    def build_hp(self, *args, **kwargs):
        """ 
        Return HyperParameter object setting hyperparameter that differ from
        default. 
        
        args : tuple
            Tuple with HyperParameter objects to add. 
        kwargs : dict 
            hyperparameter name, value-pairs. 
        
        """ 
        hp = tuner.HyperParameters()  
        
        #Overwrite defaults
        for arg in args:
            hp.merge(arg)
        
        fixed = tuner.HyperParameters()
        for key, value in kwargs.items():
            fixed.Fixed(key, value)
        hp.merge(fixed)
        
        return hp
    
    def _build_default_hp(self, hp):
        """ Set default hyperparameters. """
        #Training setup
        hp.Fixed('epochs', 50)
        hp.Int('batch_size', default=64, min_value=1, max_value=1024, 
               sampling='log')
        hp.Float('lr_init', default=5e-3, min_value=5e-4, max_value=1e-2, 
                 step=5e-4)
        
        #Basic layer setup
        hp.Int('no_layers', default=4, min_value=0, max_value=8, step=1)
        hp.Int('no_nodes', default=64, min_value=2, max_value=1024, 
               sampling='log')
        hp.Fixed('latent_dim', 2)
        hp.Fixed('state_dim', 2)
        
        #Fine network details
        hp.Int('mc_samples', min_value=1, max_value=10, step=1)
        hp.Boolean('use_rotation', default=False)
        hp.Float('l2_rotation', default=1.0e-2, min_value=1.0e-4, 
                 max_value=1.0, sampling='log')
        hp.Float('l1', min_value=0., max_value=1.0e-2, step = 1.0e-3)
        
        return hp
               
    def _build_network(self, input_layer):
        """ Build dense layers for encoder and decoder. """
        no_layers = self.hp.get('no_layers')
        no_nodes  = self.hp.get('no_nodes')
        
        x = layers.BatchNormalization()(input_layer)
        for l in range(no_layers):
            x = layers.Dense(no_nodes,
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.L1(self.hp.get('l1'))
                             )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.3)(x)  
            
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

def tune_DenseVae(x, trials=20):
    """ Function to tune the hyperparameters in DenseVae. """
    
    hypermodel = DenseVae()   
    file_dir = '/home/ivo/Code/VAE/tmp'
    
    #Writer logs 
    tensorboard_writer = tf.keras.callbacks.TensorBoard(file_dir)

    #Tune layers/nodes 
    hp = tuner.HyperParameters()
    hp.Int("no_layers", min_value=0, max_value=8, step=1)
    hp.Int("no_nodes", min_value=4, max_value=256, sampling='log')
    hp.Boolean('use_rotation', default=True)
    hp.Int('mc_samples', min_value=1, max_value=10, step=1)
    hp.Float('lr_init',  default=5e-3, min_value=5e-4, max_value=5e-3, step=5e-4)
    hp.Float('l2_rotation', default=1e-2)

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
