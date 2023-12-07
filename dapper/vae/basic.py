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

#constant pi
PI = tf.constant(np.pi)
#clear 
tf.keras.saving.get_custom_objects().clear()

def rotate(x, theta, axis=-1):
    x = np.swapaxes(x, axis, 0)
    if np.any(np.shape(theta) != np.shape(x[0])):
        raise ValueError("Shape theta does not match that of input.")
    x = np.stack((np.cos(theta)*x[0]-np.sin(theta)*x[1],
                  np.sin(theta)*x[0]+np.cos(theta)*x[1]), axis=0)
    x = np.swapaxes(x, 0, axis)
    return x


#%% Variational autoencoder based on dense neural network. 

@tf.keras.saving.register_keras_serializable(package="VAE")
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
    
 
@tf.keras.saving.register_keras_serializable(package="VAE")       
class VAE(keras.Model):
    """ Variational autoencoder model. """
    
    def __init__(self, encoder, decoder, mc_samples=1, l2_rotation=0.0, 
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
    
    def get_config(self):
        return {**super().get_config(), 
                'encoder':self.encoder.get_config(),
                'decoder':self.decoder.get_config(),
                'mc_samples':self.mc_samples,
                'l2_rotation':self.l2_rotation}
    
    @classmethod 
    def from_config(cls, config):
        encoder = config.pop('encoder')
        encoder = tf.keras.Model.from_config(encoder)
        decoder = config.pop('decoder')
        decoder = tf.keras.Model.from_config(decoder)
        return cls(encoder, decoder, **config)
    
    @property 
    def built(self):
        return self.encoder.built and self.decoder.built
    
    @built.setter
    def built(self, is_build):
        pass
    
    def reconstruction_loss(self, data):
        """ Reconstruction loss estimate used by others. """
        _, _, z = self.encoder(data)
        x_mean, _, _ = self.decoder(z)
        loss = tf.square(data - x_mean)
        return 0.5 * tf.reduce_sum(loss, axis=1)
    
    def mc_reconstruction_loss(self, data, z):
        """ Reconstruction loss estimated from Monte-Carlo approximation. """
        EPS = tf.constant(1e-6)
        
        x_mean, x_log_var, x_sin  = self.decoder(z)  
        x_cos = tf.sqrt(1 - x_sin**2)
        #Turn error into its principal component
        error   = data - x_mean
        if x_mean.shape[1]==2:
            error0  = x_cos[:,0:1] * error[:,0:1] + x_sin[:,0:1] * error[:,1:2] 
            error1  = x_sin[:,0:1] * error[:,0:1] - x_cos[:,0:1] * error[:,1:2]
            error12 = tf.keras.layers.Concatenate(axis=-1)([error0, error1])
        else:
            error12 = error 
        #-2 log p(z|x)
        loss  = x_log_var
        loss += tf.square(error12) / (tf.exp(x_log_var) + EPS)
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
        if self.hp.is_active('l2_rotation'):
            l2_rotation = self.hp.get('l2_rotation')
        else:
            l2_rotation = 0.0
            
        model = VAE(encoder, decoder, mc_samples=self.hp.get('mc_samples'),
                    l2_rotation=l2_rotation)
        
        #Set default weights for observation operators 
        self.set_state_obs(model)
        self.set_latent_obs(model)
        
        #Build learning function.
        self.lr = [tf.keras.callbacks.ReduceLROnPlateau("reconstruction_loss",
                                                        factor=.5, patience=2,
                                                        min_delta=.05, mode='min',
                                                        verbose=False),
                   ]
        
        #Build stopper 
        self.stopper = [tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                         patience=5, verbose=False,
                                                         restore_best_weights=True,
                                                         min_delta=0.0,
                                                         start_from_epoch=10)]
        
        #Compile before use and return.  
        self.compile(model)
        
        return model
    
    def compile(self, model):
        #Set trainable layer
        self.set_trainable(model)
        
        #Compile with optimizer
        lr = self.hp.get('lr_init')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    def fit(self, hp, model, *args, **kwargs):  
        fit_args = {'epochs':hp.get('epochs'), 
                    'batch_size':hp.get('batch_size'),
                    'shuffle':True,
                    'callbacks':[],
                    'verbose':True,
                    **kwargs
                    }
        fit_args['callbacks']  = fit_args['callbacks']
        fit_args['callbacks'] += self.stopper + self.lr
        
        return model.fit(*args, **fit_args)
    
    def set_trainable(self, model):
        #Switch everything off
        notrain = model.encoder.layers + model.decoder.layers 
        for l in notrain:
            l.trainable = False 
            
        if self.hp.get('training') == 'full':
            train = model.encoder.layers[2:] + model.decoder.layers[2:] 
        elif self.hp.get('training') == 'full_nobatch':
            train = model.encoder.layers[2:] + model.decoder.layers[2:]
            train = [l for l in train if not isinstance(l, layers.BatchNormalization)]
        elif self.hp.get('training') == 'transfer':
            train = model.encoder.layers[-3:] + model.decoder.layers[-3:]
            
        for l in train:
            l.trainable = True
            
        with self.hp.conditional_scope('use_rotation',[True]):
            layer = model.decoder.layers[-1]
            layer.trainable = self.hp.get('use_rotation') and layer.trainable
            
            
    def set_state_obs(self, model, H=None, state=None):
        state_dim, hidden_dim = self.hp.get('state_dim'), self.hp.get('hidden_dim')
        if state is None:
            state = np.zeros((hidden_dim,))
        if H is None:
            H = np.eye(hidden_dim, state_dim)
            
        model.encoder.layers[1].set_weights([H.T, state]) 
            
    def set_latent_obs(self, model, H=None, state=None):
        latent_dim, hidden_dim = self.hp.get('latent_dim'), self.hp.get('hidden_dim')
        if state is None:
            state = np.zeros((hidden_dim),)
        if H is None:
            H = np.eye(hidden_dim, latent_dim)
                   
        model.decoder.layers[1].set_weights([H.T, state])       
        
    def fit_bkg(self, hp, model, *args, **kwargs):
        hpt = hp 
        dim = np.size(args[0],1)
        
        #Train only output layers. 
        hpt.values['training'] = 'transfer' 
        transfer = self.build(hpt)
        transfer.set_weights(model.get_weights())
        self.set_state_obs(transfer, np.eye(dim), np.zeros((dim,)))      
        output = self.fit(hp, transfer, *args, **kwargs)

        #Now update all layers 
        hpt.values['training'] = 'full_nobatch'
        hpt.values['lr_init'] = 2*float(transfer.optimizer.learning_rate)
        self.compile(transfer)
        output = self.fit(hpt, transfer, *args, **kwargs)

        return output, transfer
    
    def fit_obs(self, hp, model, *args, **kwargs):
        hpt = hp 
        
        #Train only output layers. 
        hpt.values['state_dim'] = np.size(args[0],1)
        hpt.values['training'] = 'transfer' 
        transfer = self.build(hpt)
        
        for l,l0 in zip(transfer.encoder.layers[2:], model.encoder.layers[2:]):
            l.set_weights(l0.get_weights())
        for l,l0 in zip(transfer.decoder.layers[:-3], model.decoder.layers[:-3]):
            l.set_weights(l0.get_weights())
        
        H, state = None, None
        if 'H' in kwargs:
            H = kwargs.pop('H')
        if 'state' in kwargs:
            state = kwargs.pop('state')
        self.set_state_obs(transfer, H, state)
        output = self.fit(hp, transfer, *args, **kwargs)

        #Now update all layers 
        hpt.values['training'] = 'full_nobatch'
        hpt.values['lr_init'] = 2*float(transfer.optimizer.learning_rate)
        self.compile(transfer)
        output = self.fit(hpt, transfer, *args, **kwargs)

        return output, transfer
    
    def _file_paths(self, file_path):
        return {'dir'     : os.path.split(file_path)[0],
                'name'    : os.path.split(file_path)[1],
                'encoder' : file_path + '_encoder.tf',
                'decoder' : file_path + '_decoder.tf',
                'hp'      : file_path + '_hp.pkl'}
    
    
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
        hp.Choice('training', ['full','full_nobatch','transfer'], default='full')
        hp.Int('no_nodes', default=64, min_value=2, max_value=1024, 
               sampling='log')
        hp.Fixed('latent_dim', 2)
        hp.Fixed('state_dim', 2)
        hp.Fixed('hidden_dim', 2)
        
        #Fine network details
        hp.Int('mc_samples', min_value=1, max_value=10, step=1)
        hp.Float('l1', min_value=0., max_value=1.0e-2, step = 1.0e-3)
        hp.Boolean('use_rotation', default=True)
        with hp.conditional_scope('use_rotation', [True]):
            hp.Float('l2_rotation', default=1.0e-2, min_value=1.0e-4, 
                     max_value=1.0, sampling='log')
        
        return hp
    
    def _add_layer_combo(self, x, no):
        name = 'hidden{:02d}'.format(no)
        x = layers.Dense(self.hp.get('no_nodes'), name=name+'_dense',
                         kernel_regularizer=tf.keras.regularizers.L1(self.hp.get('l1')),
                         kernel_initializer='he_normal',
                         )(x)
        x = layers.BatchNormalization(name=name+'_norm')(x)
        x = layers.LeakyReLU(0.3, name=name+'_activation')(x)  
        return x
    
    def _copy_layer(self, layer_in, tensor_in, **kwargs):
        layer_out = layer_in.from_config(layer_in.get_config())
        tensor_out = layer_out(tensor_in)
        if hasattr(layer_out, 'trainable') and 'trainable' in kwargs:
            layer_out.trainable = kwargs['trainable']
        if hasattr(layer_in, 'get_weights'):
            layer_out.set_weights(layer_in.get_weights())
        return layer_out, tensor_out
    
    def _build_encoder(self, state_dim, latent_dim):    
        """ Build the encoder. """
        hidden_dim = self.hp.get('hidden_dim')
        
        #Input processing. 
        input_layer = keras.Input(shape=(state_dim,), name='input')     
        x = layers.Dense(hidden_dim, name='obs_op')(input_layer)
        x = layers.BatchNormalization(name='obs_norm')(x)
       
        #Hidden layers
        for n in range(self.hp.get('no_layers')):
            x = self._add_layer_combo(x, n)
        
        #Latent output 
        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z         = SamplingLayer(name='z_sample')([z_mean, z_log_var])
        
        encoder = keras.Model(input_layer, [z_mean, z_log_var, z], 
                              name="encoder")
        
        return encoder
        
    def _build_decoder(self, state_dim, latent_dim):
        """ Build the decoder. """
        hidden_dim = self.hp.get('hidden_dim')
        
        #Input processing 
        input_layer = keras.Input(shape=(latent_dim,), name='input')
        x = layers.Dense(hidden_dim, name='sampling')(input_layer)
        x = layers.BatchNormalization(name='sampling_norm')(x)
        
        #Hidden layers
        for n in range(self.hp.get('no_layers')):
            x = self._add_layer_combo(x, n)
          
        #Output 
        x_mean    = layers.Dense(state_dim, name="x_mean")(x)
        x_log_var = layers.Dense(state_dim, name="x_log_var")(x)  
        x_sin     = layers.Dense(state_dim-1, name="x_sin", activation='tanh',
                                 kernel_initializer='zeros', 
                                 bias_initializer='zeros',
                                 trainable=False)(x) 
        
        decoder = keras.Model(input_layer, [x_mean, x_log_var, x_sin], 
                              name="decoder")
        
        with self.hp.conditional_scope('use_rotation',[True]):
            decoder.layers[-1].trainable = self.hp.get('use_rotation')
        
        return decoder
    
       
  

#%% Tuning 

def tune_DenseVae(x):
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
    hp.Float('lr_init',  default=5e-3, min_value=5e-4, max_value=5e-3, step=5e-4)
    with hp.conditional_scope('use_rotation', [True]):
        hp.Float('l2_rotation', default=1e-2, min_value=1e-4, max_value=1, 
                 sampling='log')

    architecture = tuner.Hyperband(hypermodel=hypermodel,
                                   objective=tuner.Objective('loss', direction='min'),
                                   hyperparameters=hp, 
                                   tune_new_entries=False, 
                                   max_epochs = 50,
                                   directory=file_dir,
                                   overwrite=True,
                                   max_retries_per_trial=1,
                                   hyperband_iterations=1)
    #Carry out the search
    architecture.search(x, callbacks=[tensorboard_writer])
    
    return architecture 
