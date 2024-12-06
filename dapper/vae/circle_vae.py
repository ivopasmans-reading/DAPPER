#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:11:36 2024

Module containing classes to create DNN-VAE. 

@author: ivo
"""

import numpy as np
import dill, random, os
from abc import ABC, abstractmethod 

import tensorboard
import tensorflow as tf
import keras_tuner as tuner
import keras 
from keras import layers 
from keras import backend as K

#Directory to store logs from VAE optimization. 
USE_TENSORBOARD = False
if USE_TENSORBOARD:
    LOG_DIR = '/home/ivo/dpr_data/vae/tensorboard/logs'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# constant pi
PI = keras.ops.convert_to_tensor(np.pi)
# clear
#tf.keras.utils.get_custom_objects().clear()
# Small number
EPS = keras.ops.convert_to_tensor(1e-6)

def rotate(x, theta, axis=-1):
    x = np.swapaxes(x, axis, 0)
    if np.any(np.shape(theta) != np.shape(x[0])):
        raise ValueError("Shape theta does not match that of input.")
    x = np.stack((np.cos(theta)*x[0]-np.sin(theta)*x[1],
                  np.sin(theta)*x[0]+np.cos(theta)*x[1]), axis=0)
    x = np.swapaxes(x, 0, axis)
    return x

#%% Neural-network builders creating layers. 

class CoderBuilder(ABC):
    """ 
    Build encoder and decoder. 
    """
    
    def reset(self):
        self.encoder, self.decoder = None, None 
        self.stopper = None 
        self.lr = None 
        
    @abstractmethod 
    def build_encoder(self):
        """ Build encoder network. """
        pass
    
    @abstractmethod 
    def build_decoder(self):
        """ Build decoder network. """
        pass
    
    def build_model(self, hp):
        """ Build model from layers using settings in hp."""
        # Build actual model.
        if hp.is_active('l2_rotation'):
            l2_rotation = hp.get('l2_rotation')
        else:
            l2_rotation = 0.0

        # Create the VAE
        self.model = VAE(self.encoder, self.decoder, 
                         mc_samples=hp.get('mc_samples'),
                         l2_rotation=l2_rotation, alpha=self.alpha)
    
    def build_alpha(self, func):
        """ Decoder variance nudging factor. """
        self.alpha = func
    
    def build_stopper(self, **kwargs):
        """ Build stopper for VAE training. """ 
        stopper_options = {'monitor':'loss','patience':5, 'verbose':False,
                           'restore_best_weights':True, 'min_delta':0.01,
                           'start_from_epoch':20, 'mode':'min'}
        stopper_options = {**stopper_options, **kwargs}
        
        self.stopper = [keras.callbacks.EarlyStopping(**stopper_options),
                        keras.callbacks.TerminateOnNaN()]
        
    def build_lr(self, **kwargs):
        """ Method that adjust learning rate as function of epoch. """
        learning_options = {'monitor':"loss", 'factor':.5, 'patience':2,
                            'min_delta':.1, 'mode':'min', 'min_lr':1e-6,
                            'verbose':False}
        learning_options = {**learning_options, **kwargs}
        self.lr = [keras.callbacks.ReduceLROnPlateau(**learning_options)]
        
        
    def build_scale_layer(self, hp):
        """ Build the layer that centers latent space distribution. """
        latent_dim = hp.get('latent_dim')
        self.scale_layer = layers.Dense(latent_dim, name="z_mean_rescale",
                                        kernel_initializer='identity',
                                        trainable=False,
                                        kernel_constraint=keras.constraints.NonNeg())
        
    def _add_model_layers(self, hp, input_layer, name):
        """ Create a set of linear layer, activation function pair. """
        hidden_dim = hp.get('hidden_dim')
        nodes = hp.get('no_nodes')

        x = input_layer
        for n in range(hp.get('no_layers')):
            l1 = keras.regularizers.L1(hp.get('l1'))
            x  = layers.Dense(nodes, name=f'hidden{n:02d}_{name}_dense',
                              kernel_regularizer=l1,
                              kernel_initializer='he_normal',
                             )(x)
            x  = layers.LeakyReLU(0.1, 
                                  name=f'hidden{n:02d}_{name}_activation')(x)

        return x
        
class StateCoderBuilder(CoderBuilder):
    """ 
    VAE applied to model states using different networks for 
    mean and variance. 
    """
    
    def build_encoder(self, hp):
        state_dim = hp.get('state_dim')
        latent_dim = hp.get('latent_dim')
        
        # Input to decoder
        input_layer = layers.Input(shape=(state_dim,), name='x_input')

        # Mean
        z_mean = self._add_model_layers(hp, input_layer, name='z_mean')
        z_mean = layers.Dense(latent_dim, name="output_z_mean")(z_mean)
        z_mean = self.scale_layer(z_mean)

        # Var
        z_log_var = self._add_model_layers(hp, input_layer, name='z_log_var')
        z_log_var = layers.Dense(latent_dim, name="output_z_log_var")(z_log_var)
        z_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='z_var_rescale')(z_log_var)

        # Sample
        z_sample = SamplingLayer(name='z_sample')([z_mean, z_log_var])

        # Different models
        z_mean_model = keras.Model(input_layer, z_mean, name='encoder_mean')
        z_var_model = keras.Model(input_layer, z_log_var, name='encoder_var')
        z_sample_model = keras.Model(input_layer, z_sample,
                                     name='encoder_sample')
        encoder = keras.Model(input_layer, [z_mean, z_log_var, z_sample],
                              name='encoder')

        self.encoder = encoder 
        
    def build_decoder(self, hp):
        state_dim = hp.get('state_dim')
        latent_dim = hp.get('latent_dim')
        hidden_dim = hp.get('hidden_dim')

        # Input processing.
        input_layer = layers.Input(shape=(latent_dim,), name='z_input')
        trans_input = InvertScalingLayer(self.scale_layer,
                                         trainable=False,
                                         name='sampling_rescale')(input_layer)

        # Mean
        x_mean = self._add_model_layers(hp, trans_input, 'x_mean')
        x_mean = layers.Dense(state_dim, name="output_x_mean")(x_mean)

        # Var
        x_log_var = self._add_model_layers(hp, trans_input, 'x_log_var')
        x_log_var = layers.Dense(state_dim, name="output_x_log_var")(x_log_var)

        # sin IP
        x_sin = ZeroLayer(name='output_x_sin')(x_mean[:, 0:1])
        with hp.conditional_scope('use_rotation', [True]):
            if hp.get('use_rotation'):
                #x_sin = layers.Lambda(lambda x: keras.ops.stop_gradient(x),
                #                      output_shape=(None,latent_dim),
                #                      name='rotation_stop_gradient')(trans_input)
                x_sin = self._add_model_layers(hp, x_sin, 'x_sin')
                x_sin = layers.Dense(state_dim-1, name="output_x_sin", 
                                     activation='tanh',
                                     kernel_initializer='he',
                                     bias_initializer='zeros',
                                     trainable=False)(x_sin)

        # Sample
        x_sample = SamplingLayer(name='output_x_sample')([x_mean, x_log_var])
        x_sample = RotateLayer(name='rotation')(x_sample, x_sin)

        # Different models
        x_mean_model = keras.Model(input_layer, x_mean, name='decoder_mean')
        x_var_model = keras.Model(input_layer, x_log_var, name='decoder_var')
        x_sample_model = keras.Model(input_layer, x_sample, name='decoder_sample')
        decoder = keras.Model(input_layer, [x_mean, x_log_var, x_sin, x_sample],
                              name='decoder')
        
        self.decoder = decoder
        
class TrunkCoderBuilder(CoderBuilder):
    """ 
    VAE applied to model states with mean and variance 
    sharing layers. 
    """
    
    def build_encoder(self, hp):
        state_dim = hp.get('state_dim')
        latent_dim = hp.get('latent_dim')
        
        # Input to decoder
        input_layer = layers.Input(shape=(state_dim,), name='x_input')

        # Mean
        z_trunk = self._add_model_layers(hp, input_layer, name='z')
        z_mean = layers.Dense(latent_dim, name="output_z_mean")(z_trunk)
        z_mean = self.scale_layer(z_mean)

        # Var
        z_log_var = layers.Dense(latent_dim, name="output_z_log_var")(z_trunk)
        z_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='z_var_rescale')(z_log_var)

        # Sample
        z_sample = SamplingLayer(name='z_sample')([z_mean, z_log_var])

        # Different models
        encoder = keras.Model(input_layer, [z_mean, z_log_var, z_sample],
                              name='encoder')

        self.encoder = encoder 
        
    def build_decoder(self, hp):
        state_dim = hp.get('state_dim')
        latent_dim = hp.get('latent_dim')
        hidden_dim = hp.get('hidden_dim')

        # Input processing.
        input_layer = layers.Input(shape=(latent_dim,), name='z_input')
        trans_input = InvertScalingLayer(self.scale_layer,
                                         trainable=False,
                                         name='sampling_rescale')(input_layer)

        # Mean
        x_trunk = self._add_model_layers(hp, trans_input, 'x')
        x_mean = layers.Dense(state_dim, name="output_x_mean")(x_trunk)

        # Var
        x_log_var = layers.Dense(state_dim, name="output_x_log_var")(x_trunk)

        # sin IP
        x_sin = ZeroLayer(name='output_x_sin')(x_mean[:, 0:1])
        with hp.conditional_scope('use_rotation', [True]):
            if hp.get('use_rotation'):
                #x_sin = layers.Lambda(lambda x: keras.ops.stop_gradient(x),
                #                      output_shape=(None,latent_dim),
                #                      name='rotation_stop_gradient')(trans_input)
                x_sin = layers.Dense(state_dim-1, name="output_x_sin", 
                                     activation='tanh',
                                     kernel_initializer='he',
                                     bias_initializer='zeros',
                                     trainable=False)(x_trunk)

        # Sample
        x_sample = SamplingLayer(name='output_x_sample')([x_mean, x_log_var])
        x_sample = RotateLayer(name='rotation')(x_sample, x_sin)

        # Different models
        decoder = keras.Model(input_layer, [x_mean, x_log_var, x_sin, x_sample],
                              name='decoder')
        
        self.decoder = decoder
    
#%% Code to combine NN-layers into a NN-model. 

class DenseVae(tuner.HyperModel):
    """ Creates encoders, decoders using dense neural networks. """
    
    def __init__(self, builder, **kwargs):
        super().__init__(**kwargs)
        self.builder = builder
     
    def build(self, hp):
        """ Build the ML model and compile it. """
        
        # Set hyperparameters.
        self.hp = hp
        
        #Build architecture 
        if self.hp.get('architecture')=='clima':
            self.builder.reset()
            self.builder.build_scale_layer(self.hp)
            self.builder.build_encoder(self.hp)
            self.builder.build_decoder(self.hp)
            self.builder.build_alpha(lambda epoch : keras.ops.exp(-0.05*epoch)) #IP
            self.builder.build_stopper()
            self.builder.build_lr()
            self.builder.build_model(self.hp)
        elif self.hp.get('architecture')=='background':
            self.builder.reset()
            self.builder.build_scale_layer(self.hp)
            self.builder.build_encoder(self.hp)
            self.builder.build_decoder(self.hp)
            self.builder.build_alpha(lambda epoch : keras.ops.exp(-0.1*epoch))
            self.builder.build_stopper(start_from_epoch=100) #IP
            self.builder.build_lr(min_lr=5e-7) #IP
            self.builder.build_model(self.hp)
        elif self.hp.get('architecture')=='inno':
            self.builder.reset()
            self.builder.build_scale_layer(self.hp)
            self.builder.build_encoder(self.hp)
            self.builder.build_decoder(self.hp)
            self.builder.build_alpha(lambda epoch : keras.ops.exp(-0.1*epoch))
            self.builder.build_stopper(min_delta=.005)
            self.builder.build_lr(min_delta=.05, min_lr=5e-7)
            self.builder.build_model(self.hp)
            
        model = self.builder.model 

        # Callback that keeps track of epoch and other diagnostics.
        self.callbacks  = [self.builder.lr, self.builder.stopper,  DiagCallback()]
        if USE_TENSORBOARD:
            self.callbacks += [tensorboard_callback]

        # Compile before use and return.
        self.compile(model)

        return model
    
    def compile(self, model):
        """ Compile the NN-model and set optimizer. """
        # Set trainable layer
        self.set_trainable(model)

        # Compile with optimizer
        lr = self.hp.get('lr_init')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

        return model

    def set_trainable(self, model):
        """ Depending on settings in self.hp switch layers off/on for training. """
        
        #Default is no training.
        model.encoder.trainable = False
        model.decoder.trainable = False
        
        #Number of hidden layers to train.
        max_layers = min(self.hp.get('no_layers'), 
                         self.hp.get('training_hidden'))
        
        #Set trainable layers decoder.
        pairs = [(f'hidden{n:02d}_',layer.name) for n in range(max_layers)
                 for layer in model.decoder.layers]
        if self.hp.get('training_output_x'):
            pairs += [(f'output_',layer.name) for layer in model.decoder.layers]
        for name in [pair[1] for pair in pairs if pair[0] in pair[1]]:
            model.decoder.get_layer(name).trainable = True
            
        #Set trainable layers encoder.
        de2en = lambda n : self.hp.get('no_layers') - n - 1
        pairs = [(f'hidden{de2en(n):02d}_',layer.name) for n in range(max_layers)
                 for layer in model.encoder.layers]
        if self.hp.get('training_output_z'):
            pairs += [(f'output_',layer.name) for layer in model.encoder.layers]
        for name in [pair[1] for pair in pairs if pair[0] in pair[1]]:
            model.encoder.get_layer(name).trainable = True      
            
        #Set trainable of rotation layers. 
        with self.hp.conditional_scope('use_rotation', [True]):
            if self.hp.get('use_rotation'):
                layer = model.decoder.get_layer(name='output_x_sin')
                layer.trainable = self.hp.get('use_rotation') and self.hp.get('training_output_x')
            
        return model

    def fit(self, hp, model, *args, **kwargs):
        """ Calculate weights of the model. """
        fit_args = {'epochs': hp.get('epochs'),
                    'batch_size': hp.get('batch_size'),
                    'shuffle': True,
                    'callbacks': [],
                    'verbose': hp.get('verbose'),
                    **kwargs
                    }
        fit_args['callbacks']  = fit_args['callbacks']
        fit_args['callbacks'] += self.callbacks

        keras.utils.set_random_seed(1000)
        return model.fit(*args, **fit_args)
    
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
        hp = self._build_default_hp(hp)

        # Overwrite defaults
        for arg in args:
            hp.merge(arg)

        fixed = tuner.HyperParameters()
        for key, value in kwargs.items():
            fixed.Fixed(key, value)
        hp.merge(fixed)

        return hp
    
    def _build_default_hp(self, hp):
        """ Set default hyperparameters. """

        # Training setup
        hp.Fixed('verbose',False)
        hp.Fixed('epochs', 100) #IP
        hp.Int('batch_size', default=64, min_value=1, max_value=1024,
               sampling='log')
        hp.Float('lr_init', default=5e-3, min_value=5e-4, max_value=1e-2,
                 step=5e-4)

        # Basic layer setup
        hp.Int('no_layers', default=4, min_value=0, max_value=8, step=1)
        #hp.Choice('training', ['offline', 'online', 'obs'], default='offline')
        hp.Boolean('training_output_z', default=False)
        hp.Boolean('training_output_x', default=False)
        hp.Int('training_hidden', default=0, min_value=0, step=1,
               max_value=hp.get('no_layers'))
        
        hp.Choice('architecture', ['state','obs'], default='state')
        hp.Int('no_nodes', default=64, min_value=2, max_value=1024,
               sampling='log')
        hp.Fixed('latent_dim', 2)
        hp.Fixed('state_dim', 2)
        hp.Fixed('hidden_dim', 2)
        hp.Fixed('obs_dim', 0)

        # Fine network details
        hp.Int('mc_samples', min_value=1, max_value=10, step=1)
        hp.Float('l1', min_value=0., max_value=1.0e-2, step=1.0e-3)
        hp.Boolean('use_rotation', default=True)
        with hp.conditional_scope('use_rotation', [True]):
            hp.Float('l2_rotation', default=1.0e-2, min_value=1.0e-4,
                     max_value=1.0, sampling='log')

        return hp
    
    def clear(self):
        """ Remove model from memory. """
        keras.backend.clear_session(free_memory=True)
    
def tune_DenseVae(x):
    """ Function to tune the hyperparameters in DenseVae. """

    #Create VAE
    hypermodel = DenseVae()

    # Writer logs
    if USE_TENSORBOARD:
        tensorboard_writer = keras.callbacks.TensorBoard(LOG_DIR)
        callbacks = [tensorboard_writer]
    else:
        callbacks = []

    # Tune layers/nodes
    hp = tuner.HyperParameters()
    hp.Int("no_layers", min_value=0, max_value=8, step=1)
    hp.Int("no_nodes", min_value=4, max_value=256, sampling='log')
    hp.Boolean('use_rotation', default=True)
    hp.Float('lr_init',  default=1e-3, min_value=1e-4,
             max_value=5e-3, step=5e-4)
    with hp.conditional_scope('use_rotation', [True]):
        hp.Float('l2_rotation', default=1e-2, min_value=1e-4, max_value=1,
                 sampling='log')

    architecture = tuner.Hyperband(hypermodel=hypermodel,
                                   objective=tuner.Objective('loss', 
                                                             direction='min'),
                                   hyperparameters=hp,
                                   tune_new_entries=False,
                                   max_epochs=50,
                                   directory=LOG_DIR,
                                   overwrite=True,
                                   max_retries_per_trial=1,
                                   hyperband_iterations=1)
    # Carry out the search
    architecture.search(x, callbacks=callbacks)

    return architecture

#%% Class representing generic VAE model. 

@keras.utils.register_keras_serializable(package="VAE")
class VAE(keras.Model):
    """ Variational autoencoder model. """

    def __init__(self, encoder, decoder, mc_samples=1, l2_rotation=0.0, 
                 alpha = lambda epoch : 0.0,
                 var_min=.05**2, 
                 **kwargs):

        super().__init__(**kwargs)
        self.mc_samples = mc_samples
        self.encoder = encoder
        self.decoder = decoder
        self.l2_rotation = l2_rotation
        self.alpha_function = alpha
        
        self.alpha_tracker = keras.metrics.Mean(name='alpha')
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.angle_loss_tracker = keras.metrics.Mean(name="angle_loss")
        self.z_M1_tracker = keras.metrics.Mean(name='z_M1')
        self.z_M2_tracker = keras.metrics.Mean(name='z_M2')
        self.x_var_min = keras.ops.convert_to_tensor(var_min)
        
        self.alpha_factor = tf.Variable(0.0, trainable=False)
        self.noise_layer = NoiseLayer()

    def get_config(self):
        return {**super().get_config(),
                'encoders': self.encoder.get_config(),
                'decoders': self.decoder.get_config(),
                'mc_samples': self.mc_samples,
                'l2_rotation': self.l2_rotation,
                'alpha_function': self.alpha_function}
    
    @classmethod
    def from_config(cls, config):
        encoder = config.pop('encoders')
        encoder = keras.saving.deserialize_keras_object(encoder, safe_mode=False)
        decoder = config.pop('decoders')
        decoder = keras.saving.deserialize_keras_object(decoder, safe_mode=False)
        mc_samples = config.pop('mc_samples')
        l2_rotation = config.pop('l2_rotation')
        alpha = config.pop('alpha_function')
        return cls(encoder, decoder, mc_samples, l2_rotation, alpha)
    
    def set_alpha(self, epoch):
        self.alpha_factor.assign(self.alpha_function(epoch))
    
    def train_step(self, data):
        
        with tf.GradientTape(persistent=True) as tape:  
            #data = self.noise_layer(data1, self.alpha_factor)
            
            #Sample latent z \sim p(x|z) 
            _, _, z = self.encoder(data)
    
            #Different losses in cost function
            kl_loss = keras.ops.mean(self.kl_loss(data)) 
            rec_loss = keras.ops.mean(self.mc_reconstruction_loss(data, z, self.alpha_factor))
            angle_loss = keras.ops.mean(self.mc_angle_loss(data, z))
            total_loss = (rec_loss + kl_loss + angle_loss)

            #Moments latent distribution. 
            z_M1 = keras.ops.mean(z, axis=0)
            z_M2 = keras.ops.mean(keras.ops.square(z), axis=0)

        weights = self.trainable_weights
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))
        
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.z_M1_tracker.update_state(z_M1)
        self.z_M2_tracker.update_state(z_M2)

        # Output losses.
        losses = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "angle_loss": self.angle_loss_tracker.result(),
            'z_M1': self.z_M1_tracker.result(),
            'z_M2': self.z_M2_tracker.result(),
            'alpha': self.alpha_tracker.result(),
            }

        return losses

    @property
    def built(self):
        return self.encoder.built and self.decoder.built

    @built.setter
    def built(self, is_build):
        pass
    
    @property
    def metrics(self):
        metrics = [self.total_loss_tracker,
                   self.reconstruction_loss_tracker,
                   self.kl_loss_tracker,
                   self.angle_loss_tracker,
                   self.z_M1_tracker,
                   self.z_M2_tracker,
                   self.alpha_tracker,
                   ]
        return metrics

    def reconstruction_loss(self, data, z):
        """ Reconstruction loss estimate used by others. """
        x_mean, x_log_var, _, _ = self.decoder(z)
        loss = keras.ops.square(data - x_mean) / self.x_var_min
        return keras.ops.sum(loss, axis=1)
    
    def gauss_loss(self, z, alpha):
        
        batch = keras.ops.shape(z1)[0]
        dim = keras.ops.shape(z1)[1]
        epsilon = keras.random.normal(shape=(batch, dim)) 
        
        
        z = (1-alpha) * z1 + alpha * epsilon #IP

    def mc_reconstruction_loss(self, data, z, alpha):
        """ Reconstruction loss estimated from Monte-Carlo approximation. """
        self.alpha_tracker.update_state(alpha)

        z_mean, z_log_var, _ = self.encoder(data)
        x_mean, x_log_var, x_sin, _ = self.decoder(z)
        # Turn error into its principal component
        error = data - x_mean
        error = RotateLayer()(error, x_sin)

        # Force variances to preset value in first epochs.
        log_var0 = keras.ops.log(self.x_var_min)
        log_var = (1-alpha) * x_log_var + alpha * log_var0

        # -2 log p(z|x) [with regularization]
        loss = log_var
        loss += keras.ops.square(error) / (keras.ops.exp(log_var) + EPS)
        loss += keras.ops.square(x_log_var - log_var)

        return 0.5 * keras.ops.sum(loss, axis=-1)

    def mc_angle_loss(self, data, z):
        """ Regularization term to keep polar angle 1st principal component
        small. """
        _, _, x_sin, _ = self.decoder(z)
        # L2 regularization term for angles.
        loss = keras.ops.sum(keras.ops.square(x_sin), axis=-1)
        return 0.5 * self.l2_rotation * loss

    def kl_loss(self, data):
        """ Exact Kullbeck-Leibler divergence for Gaussian and normal 
        distribution. """
        # Dimension
        k = data.shape[1]
        # KL loss
        z_mean, z_log_var, _ = self.encoder(data)
        # Trace Sigma + log 1/det(Sigma) - dim + ||mu-0||**2
        KL = -z_log_var - k + keras.ops.square(z_mean) + keras.ops.exp(z_log_var)
        loss = 0.5 * keras.ops.sum(KL, axis=1)
        return loss

    def mc_kl_loss(self, data, z):
        """ 
        Kullbeck-Leibler divergence between Gaussian and normal distribution 
        calculated based on Monte-Carlo approximation. 
        """
        z_mean, z_log_var, _ = self.encoder(data)
        # log p(z|x)
        loss = -0.5 * keras.ops.square(z - z_mean) / keras.ops.exp(z_log_var)
        loss -= 0.5 * z_log_var
        # log 1/p(z)
        loss += 0.5 * keras.ops.square(z)
        return keras.ops.sum(loss, axis=1)

#%% Custom layers and callbacks.

class DiagCallback(keras.callbacks.Callback):
    """ Make epoch and batch indices available in the code. """

    def on_epoch_begin(self, epoch, logs=None):
        self.model.set_alpha(epoch)

    def on_batch_begin(self, batch, logs=None):
        pass
        

@keras.utils.register_keras_serializable(package="VAE")
class SamplingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        batch = keras.ops.shape(mean)[0]
        dim = keras.ops.shape(mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim)) 
        return mean + keras.ops.exp(0.5 * log_var) * epsilon  

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="VAE")
class NoiseLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, alpha):
        Ncopies = 50
        copies = keras.layers.Concatenate(axis=0)([inputs]*Ncopies)
        
        batch = keras.ops.shape(copies)[0]
        dim = keras.ops.shape(copies)[1]
        
        copies = copies + 0.05 * alpha * keras.random.normal(shape=(batch, dim))
        
        return copies 

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable(package="VAE")
class VarScalingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, source_layer, **kwargs):
        super().__init__(**kwargs)
        self.source = source_layer

    def call(self, inputs):
        # Create kernel
        kernel = self.source.kernel
        sigma2 = keras.ops.log(kernel)
        x = inputs + keras.ops.diag(sigma2)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {"source_layer" : keras.saving.serialize_keras_object(self.source)}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        source_layer = config.pop("source_layer")
        source_layer = keras.saving.deserialize_keras_object(source_layer, safe_mode=False)
        return cls(source_layer, **config)


@keras.utils.register_keras_serializable(package="VAE")
class InvertScalingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, source_layer, **kwargs):
        super().__init__(**kwargs)
        self.source = source_layer

    def call(self, inputs):
        # Create kernel
        kernel = tf.linalg.pinv(self.source.kernel)
        bias = self.source.bias
        x = tf.matmul(inputs - bias, kernel)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {"source_layer" : keras.saving.serialize_keras_object(self.source)}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        source_layer = config.pop("source_layer")
        source_layer = keras.saving.deserialize_keras_object(source_layer, safe_mode=False)
        return cls(source_layer, **config)
    
@keras.utils.register_keras_serializable(package="VAE")
class RotateLayer(layers.Layer):
    """ Rotates covariance axes. """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x, sin):
        if x.shape[1] == 2:
            cos = keras.ops.sqrt(1 - sin**2)
            x0 = cos[:, 0:1]*x[:, 0:1] - sin[:, 0:1]*x[:, 1:2]
            x1 = sin[:, 0:1]*x[:, 0:1] + cos[:, 0:1]*x[:, 1:2]
            return keras.layers.Concatenate(axis=-1)([x0, x1])
        else:
            return x
        
    def get_config(self):
        return super().get_config()
    
    def from_config(cls, config):
        return cls(**config)
   
@keras.utils.register_keras_serializable(package="VAE")
class ZeroLayer(layers.Layer):
    """ Creates a layer with zeros as output. """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
        return keras.ops.zeros_like(x)
    
    def get_config(self):
        return super().get_config()
    
    def from_config(cls, config):
        return cls(**config)

#%% TODA: depreciate

class ObsCoderBuilder(CoderBuilder):
    """ 
    VAE applied to model observations using different networks for 
    mean and variance. 
    """
    
    def build_decoder(self, hp):
        # Input
        latent_dim = hp.get('latent_dim')
        input_layer = layers.Input(shape=(latent_dim,), name='z_input')
        obs_dim = hp.get('obs_dim')

        # Mean
        e_mean = self._add_model_layers(hp, input_layer, name='x_mean')
        e_mean = layers.Dense(obs_dim, name='output_x_mean')(e_mean)
        e_mean = self.scale_layer(e_mean)

        # Var
        e_log_var = self._add_model_layers(hp, input_layer, 'x_log_var')
        e_log_var = layers.Dense(obs_dim, name="output_x_log_var")(e_log_var)
        e_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='x_var_rescale')(e_log_var)

        #Rotatation (not used)
        e_sin = ZeroLayer()(e_mean)

        # Sample
        e_sample = SamplingLayer(name='output_x_sample')([e_mean, e_log_var])

        # Model
        idecoder = keras.Model(input_layer, [e_mean, e_log_var, e_sin, e_sample],
                               name='idecoder')
        
        self.decoder = idecoder 
        
    def build_encoder(self, hp):
        nodes = hp.get('no_nodes')
        obs_dim = hp.get('obs_dim')
        latent_dim = hp.get('latent_dim')

        # Input
        input_layer = layers.Input(shape=(obs_dim,), name='x_input')

        # Mean
        d_mean = self._add_model_layers(hp, input_layer, name='z_mean')
        d_mean = layers.Dense(latent_dim, name='output_z_mean')(d_mean)
        d_mean = self.scale_layer(d_mean)

        # Var
        d_log_var = self._add_model_layers(hp, input_layer, 'z_log_var')
        d_log_var = layers.Dense(latent_dim, name="output_z_log_var")(d_log_var)
        d_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='z_var_rescale')(d_log_var)

        # Sample
        d_sample = SamplingLayer(name='output_z_sample')([d_mean, d_log_var])

        iencoder = keras.Model(input_layer, [d_mean, d_log_var, d_sample],
                               name='iencoder')

        self.encoder = iencoder
        
class TrunkObsCoderBuilder(CoderBuilder):
    """ Build VAE for innovations. """
    
    def build_decoder(self, hp):
        # Input
        latent_dim = hp.get('latent_dim')
        input_layer = layers.Input(shape=(latent_dim,), name='z_input')
        obs_dim = hp.get('state_dim')

        # Mean
        e_trunk = self._add_model_layers(hp, input_layer, name='x')
        e_mean = layers.Dense(obs_dim, name='output_x_mean')(e_trunk)
        e_mean = self.scale_layer(e_mean)

        # Var
        e_log_var = layers.Dense(obs_dim, name="output_x_log_var")(e_trunk)
        e_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='x_var_rescale')(e_log_var)

        #Rotatation (not used)
        e_sin = ZeroLayer()(e_mean)

        # Sample
        e_sample = SamplingLayer(name='output_x_sample')([e_mean, e_log_var])

        # Model
        idecoder = keras.Model(input_layer, [e_mean, e_log_var, e_sin, e_sample],
                               name='idecoder')
        
        self.decoder = idecoder 
        
    def build_encoder(self, hp):
        nodes = hp.get('no_nodes')
        obs_dim = hp.get('obs_dim')
        latent_dim = hp.get('latent_dim')

        # Input
        input_layer = layers.Input(shape=(obs_dim,), name='x_input')

        # Mean
        d_trunk = self._add_model_layers(hp, input_layer, name='z')
        d_mean = layers.Dense(latent_dim, name='output_z_mean')(d_trunk)
        d_mean = self.scale_layer(d_mean)

        # Var
        d_log_var = layers.Dense(latent_dim, name="output_z_log_var")(d_trunk)
        d_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='z_var_rescale')(d_log_var)

        # Sample
        d_sample = SamplingLayer(name='output_z_sample')([d_mean, d_log_var])

        iencoder = keras.Model(input_layer, [d_mean, d_log_var, d_sample],
                               name='iencoder')

        self.encoder = iencoder