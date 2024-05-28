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
import os
import dill, random
import tensorboard

LOG_DIR = '/home/ivo/dpr_data/vae/tensorboard/logs'
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# constant pi
PI = tf.constant(np.pi)
# clear
tf.keras.saving.get_custom_objects().clear()
# Small number
EPS = tf.constant(1e-6)


def rotate(x, theta, axis=-1):
    x = np.swapaxes(x, axis, 0)
    if np.any(np.shape(theta) != np.shape(x[0])):
        raise ValueError("Shape theta does not match that of input.")
    x = np.stack((np.cos(theta)*x[0]-np.sin(theta)*x[1],
                  np.sin(theta)*x[0]+np.cos(theta)*x[1]), axis=0)
    x = np.swapaxes(x, 0, axis)
    return x

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(0)
   tf.random.set_seed(1)
   keras.utils.set_random_seed(1)
   np.random.seed(1)
   random.seed(1)

# %% Variational autoencoder based on dense neural network.


@tf.keras.saving.register_keras_serializable(package="VAE")
class SamplingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim)) 
        return mean + tf.exp(0.5 * log_var) * epsilon  # IP

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="VAE")
class VarScalingLayer(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, source_layer, **kwargs):
        super().__init__(**kwargs)
        self.source = source_layer

    def call(self, inputs):
        # Create kernel
        kernel = self.source.kernel
        x = inputs + tf.math.log(kernel)
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


@tf.keras.saving.register_keras_serializable(package="VAE")
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


class DiagCallback(keras.callbacks.Callback):

    def __init__(self, **kwargs):
        self.diags = kwargs

    def on_epoch_begin(self, epoch, logs={}):
        if 'epoch' in self.diags:
            self.diags['epoch'].assign(epoch)

    def on_batch_begin(self, batch, logs={}):
        if 'batch' in self.diags:
            self.diags['batch'].assign(batch)


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
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.angle_loss_tracker = keras.metrics.Mean(name="angle_loss")
        self.z_M1_tracker = keras.metrics.Mean(name='z_M1')
        self.z_M2_tracker = keras.metrics.Mean(name='z_M2')
        self.diags = {'epoch': tf.Variable(0.0, trainable=False)}
        self.x_var_min = tf.constant(.05**2)  # IP

    @property
    def metrics(self):
        metrics = [self.total_loss_tracker,
                   self.reconstruction_loss_tracker,
                   self.kl_loss_tracker,
                   self.angle_loss_tracker,
                   self.z_M1_tracker,
                   self.z_M2_tracker]
        return metrics

    def get_config(self):
        return {**super().get_config(),
                'encoder': self.encoder.get_config(),
                'decoder': self.decoder.get_config(),
                'mc_samples': self.mc_samples,
                'l2_rotation': self.l2_rotation}

    @classmethod
    def from_config(cls, config):
        encoder = config.pop('encoder')
        encoder = keras.saving.deserialize_keras_object(encoder, safe_mode=False)
        decoder = config.pop('decoder')
        decoder = keras.saving.deserialize_keras_object(decoder, safe_mode=False)
        return cls(encoder, decoder, **config)

    @property
    def built(self):
        return self.encoder.built and self.decoder.built

    @built.setter
    def built(self, is_build):
        pass

    def reconstruction_loss(self, data, z):
        """ Reconstruction loss estimate used by others. """
        x_mean, x_log_var, _, _ = self.decoder(z)
        loss = tf.square(data - x_mean) / self.x_var_min
        return tf.reduce_sum(loss, axis=1)

    def mc_reconstruction_loss(self, data, z, alpha):
        """ Reconstruction loss estimated from Monte-Carlo approximation. """

        z_mean, z_log_var, _ = self.encoder(data)
        x_mean, x_log_var, x_sin, _ = self.decoder(z)
        # Turn error into its principal component
        error = data - x_mean
        error = rotate_layer(error, x_sin)

        # Force variances to preset value in first epochs.
        log_var0 = tf.math.log(self.x_var_min)
        log_var = (1-alpha) * x_log_var + alpha * log_var0

        # -2 log p(z|x) [with regularization]
        loss = log_var
        loss += tf.square(error) / (tf.exp(log_var) + EPS)
        loss += tf.square(x_log_var - log_var)

        return 0.5 * tf.reduce_sum(loss, axis=-1)

    def mc_angle_loss(self, data, z):
        """ Regularization term to keep polar angle 1st principal component
        small. """
        _, _, x_sin, _ = self.decoder(z)
        # L2 regularization term for angles.
        loss = tf.reduce_sum(tf.square(x_sin), axis=-1)
        return 0.5 * self.l2_rotation * loss

    def kl_loss(self, data):
        """ Exact Kullbeck-Leibler divergence for Gaussian and normal 
        distribution. """
        # Dimension
        k = data.shape[1]
        # KL loss
        z_mean, z_log_var, _ = self.encoder(data)
        # Trace Sigma + log 1/det(Sigma) - dim + ||mu-0||**2
        KL = -z_log_var - 1 + tf.square(z_mean) + tf.exp(z_log_var)
        loss = 0.5 * tf.reduce_sum(KL, axis=1)
        return loss

    def mc_kl_loss(self, data, z):
        """ 
        Kullbeck-Leibler divergence between Gaussian and normal distribution 
        calculated based on Monte-Carlo approximation. 
        """
        z_mean, z_log_var, _ = self.encoder(data)
        # log p(z|x)
        loss = -0.5 * tf.square(z - z_mean) / tf.exp(z_log_var)
        loss -= 0.5 * z_log_var
        # log 1/p(z)
        loss += 0.5 * tf.square(z)
        return tf.reduce_sum(loss, axis=1)

    def train_step(self, data):
        mc = tf.constant(1.0/self.mc_samples)

        # Create cost function.
        with tf.GradientTape() as tape:
            kl_loss, reconstruction_loss, angle_loss = 0, 0, 0
            kl_loss = self.kl_loss(data)
            for _ in range(self.mc_samples):
                _, _, z = self.encoder(data)
                reconstruction_loss += self.mc_reconstruction_loss(
                    data, z) * mc
                angle_loss += self.mc_angle_loss(data, z) * mc

            alpha = tf.math.sigmoid(0.2*(self.diags['epoch']-10.0))
            kl_loss = tf.reduce_mean(kl_loss)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            angle_loss = tf.reduce_mean(angle_loss)
            total_loss = reconstruction_loss + alpha*kl_loss + angle_loss

        # Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update losses.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.angle_loss_tracker.update_state(angle_loss)

        # Output losses.
        losses = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "angle_loss": self.angle_loss_tracker.result(),
        }

        return losses

@tf.keras.saving.register_keras_serializable(package="VAE")
class VaeMulti(VAE):

    def __init__(self, encoder, decoder, mc_samples=1, l2_rotation=0.0,
                 **kwargs):
        
        super().__init__(encoder[0], decoder[0], mc_samples,
                         l2_rotation, **kwargs)
        self.encoders = encoder
        self.decoders = decoder
        
    def get_config(self):
        return {**super().get_config(),
                'encoders': keras.saving.serialize_keras_object(self.encoders),
                'decoders': keras.saving.serialize_keras_object(self.decoders),
                'mc_samples': self.mc_samples,
                'l2_rotation': self.l2_rotation}

    @classmethod
    def from_config(cls, config):
        encoder = config.pop('encoders')
        encoder = keras.saving.deserialize_keras_object(encoder, safe_mode=False)
        decoder = config.pop('decoders')
        decoder = keras.saving.deserialize_keras_object(decoder, safe_mode=False)
        mc_samples = config.pop('mc_samples')
        l2_rotation = config.pop('l2_rotation')
        print('CONFIG ', config)
        return cls(encoder, decoder, mc_samples, l2_rotation)

    def alpha(self):
        return tf.exp(-0.1*self.diags['epoch'])

    def train_step(self, data):
        with tf.GradientTape() as tape:
            _, _, z = self.encoder(data)
            alpha = self.alpha()
            kl_loss = tf.reduce_mean(self.kl_loss(data))
            rec_loss = tf.reduce_mean(
                self.mc_reconstruction_loss(data, z, alpha))
            angle_loss = tf.reduce_mean(self.mc_angle_loss(data, z))
            total_loss = rec_loss + kl_loss + angle_loss

            z_M1 = tf.reduce_mean(z, axis=0)
            z_M2 = tf.reduce_mean(tf.square(z), axis=0)

        weights = self.trainable_weights
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        # Update losses.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.angle_loss_tracker.update_state(angle_loss)
        self.z_M1_tracker.update_state(z_M1)
        self.z_M2_tracker.update_state(z_M2)

        # Output losses.
        losses = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "angle_loss": self.angle_loss_tracker.result(),
            'z_M1': self.z_M1_tracker.result(),
            'z_M2': self.z_M2_tracker.result()
        }

        return losses


class VaeMultiBkg(VaeMulti):

    def alpha(self):
        return tf.constant(0.0)


class DenseVae(tuner.HyperModel):
    """ Creates encoders, decoders using dense neural networks. """

    def build(self, hp):
        # Set hyperparameters.
        self.hp = self._build_default_hp(hp)

        # Build decoder/encoder-pair
        latent_dim = self.hp.values['latent_dim']
        self.scale_layer = layers.Dense(latent_dim, name="z_mean_rescale",
                                        kernel_initializer='identity',
                                        trainable=False,
                                        kernel_constraint=keras.constraints.NonNeg())
        encoder = self._build_encoder_multi(
            hp.get('state_dim'), hp.get('latent_dim'))
        decoder = self._build_decoder_multi(
            hp.get('state_dim'), hp.get('latent_dim'))

        # Build actual model.
        if self.hp.is_active('l2_rotation'):
            l2_rotation = self.hp.get('l2_rotation')
        else:
            l2_rotation = 0.0

        # Create the VAE
        model = VaeMulti(encoder, decoder, mc_samples=self.hp.get('mc_samples'),
                         l2_rotation=l2_rotation)

        # Build learning function.
        self.lr = [tf.keras.callbacks.ReduceLROnPlateau("loss",
                                                        factor=.5, patience=2,
                                                        min_delta=.1, mode='min',
                                                        min_lr=1e-6,
                                                        verbose=False),
                   ]

        # Build stopper
        self.stopper = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience=5, verbose=False,
                                                         restore_best_weights=True,
                                                         min_delta=0.05,
                                                         start_from_epoch=20),
                        tf.keras.callbacks.TerminateOnNaN()]

        # Callback that keeps track of epoch.
        self.diag = [DiagCallback(**model.diags),tensorboard_callback]

        # Compile before use and return.
        self.compile(model)

        return model

    def build_bkg(self, hp):
        # Set hyperparameters.
        self.hp = self._build_default_hp(hp)

        # Build decoder/encoder-pair
        latent_dim = self.hp.values['latent_dim']
        self.scale_layer = layers.Dense(latent_dim, name="z_mean_rescale",
                                        kernel_initializer='identity',
                                        trainable=True,
                                        kernel_constraint=keras.constraints.NonNeg())
        encoder = self._build_encoder_multi(hp.get('state_dim'),
                                            hp.get('latent_dim'))
        decoder = self._build_decoder_multi(hp.get('state_dim'),
                                            hp.get('latent_dim'))

        # Build actual model.
        if self.hp.is_active('l2_rotation'):
            l2_rotation = self.hp.get('l2_rotation')
        else:
            l2_rotation = 0.0

        # Create the VAE
        model = VaeMultiBkg(encoder, decoder,
                            mc_samples=self.hp.get('mc_samples'),
                            l2_rotation=l2_rotation)

        # Build learning function.
        self.lr = [tf.keras.callbacks.ReduceLROnPlateau("loss",
                                                        factor=.5, patience=2,
                                                        min_delta=.1, mode='min',
                                                        min_lr=1e-6,
                                                        verbose=False),
                   ]

        # Build stopper
        self.stopper = [tf.keras.callbacks.EarlyStopping(monitor='kl_loss',
                                                         patience=5, verbose=False,
                                                         restore_best_weights=True,
                                                         min_delta=0.01,
                                                         start_from_epoch=20),
                        tf.keras.callbacks.TerminateOnNaN()]

        # Callback that keeps track of epoch.
        self.diag = [DiagCallback(**model.diags)]

        # Compile before use and return.
        self.compile(model)

        return model

    def build_inno(self, hp, model, obs_dim):
        latent_dim = self.hp.values['latent_dim']
        encoder = self._build_encoder_inno(obs_dim, 1)
        decoder = self._build_decoder_inno(1, obs_dim)

        # Create the VAE
        model = VaeMulti([encoder], [decoder])

        # Build learning function.
        self.lr = [tf.keras.callbacks.ReduceLROnPlateau("loss",
                                                        factor=.5, patience=2,
                                                        min_delta=.1, mode='min',
                                                        min_lr=1e-6,
                                                        verbose=False),
                   ]

        # Build stopper
        self.stopper = [tf.keras.callbacks.EarlyStopping(monitor='kl_loss',
                                                         patience=5, verbose=False,
                                                         restore_best_weights=True,
                                                         min_delta=0.01,
                                                         start_from_epoch=20),
                        tf.keras.callbacks.TerminateOnNaN()]

        # Callback that keeps track of epoch.
        self.diag = [DiagCallback(**model.diags)]

        # Compile before use and return.
        lr = self.hp.get('lr_init')
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr))

        return model

    def compile(self, model):
        # Set trainable layer
        self.set_trainable(model)

        # Compile with optimizer
        lr = self.hp.get('lr_init')
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr))

        return model

    def fit(self, hp, model, *args, **kwargs):
        fit_args = {'epochs': hp.get('epochs'),
                    'batch_size': hp.get('batch_size'),
                    'shuffle': True,
                    'callbacks': [],
                    'verbose': True,
                    **kwargs
                    }
        fit_args['callbacks'] = fit_args['callbacks']
        fit_args['callbacks'] += self.lr + self.stopper + self.diag

        keras.utils.set_random_seed(1000)
        return model.fit(*args, **fit_args)

    def create_tree(self, model):
        for layer in model.layers:
            layer.children = []

        for layer in model.layers:
            parents = layer.inbound_nodes[0].inbound_layers
            if hasattr(parents, '__iter__'):
                layer.parents = [layer.name for layer in parents]
            else:
                layer.parents = [parents.name]

            for parent in layer.parents:
                model.get_layer(parent).children += [layer.name]

        def set_depth(name, depth=0):
            layer = model.get_layer(name)
            if isinstance(layer, layers.Dense):
                depth += 1
            layer.depth = depth
            for child in layer.children:
                set_depth(child, depth)

        set_depth(model.layers[0].name)

    def set_trainable(self, model):
        self.create_tree(model.encoder)
        self.create_tree(model.decoder)
        encoder_depths = np.unique([l.depth for l in model.encoder.layers])
        decoder_depths = np.unique([l.depth for l in model.decoder.layers])

        if self.hp.get('training') == 'offline':
            model.encoder.trainable = True
            model.decoder.trainable = True
            model.encoder.get_layer('z_mean_rescale').trainable = False
            model.encoder.get_layer('z_var_rescale').trainable = False
            model.decoder.get_layer('sampling_rescale').trainable = False

        elif self.hp.get('training') == 1:
            model.encoder.trainable = False
            model.decoder.trainable = False
            model.encoder.get_layer('z_mean_rescale').trainable = True
            model.encoder.get_layer('z_var_rescale').trainable = True
            model.decoder.get_layer('sampling_rescale').trainable = True

        elif self.hp.get('training') == 2:
            model.encoder.trainable = False
            model.decoder.trainable = False

            maxdepth = np.max(encoder_depths)
            lays = (l for l in model.encoder.layers if l.depth >= .5*maxdepth)
            for l in lays:
                l.trainable = True
            lays = (l for l in model.decoder.layers if l.depth < .5*maxdepth)
            for l in lays:
                l.trainable = True

            model.encoder.get_layer('z_mean_rescale').trainable = False
            model.encoder.get_layer('z_var_rescale').trainable = False
            model.decoder.get_layer('sampling_rescale').trainable = False

        elif self.hp.get('training') == 3:
            model.encoder.trainable = False
            model.decoder.trainable = False

            maxdepth = np.max(encoder_depths)
            lays = (l for l in model.encoder.layers if l.depth <= .5*maxdepth)
            for l in lays:
                l.trainable = True
            lays = (l for l in model.decoder.layers if l.depth > .5*maxdepth)
            for l in lays:
                l.trainable = True

            model.encoder.get_layer('z_mean_rescale').trainable = False
            model.encoder.get_layer('z_var_rescale').trainable = False
            model.decoder.get_layer('sampling_rescale').trainable = False

        elif isinstance(self.hp.get('training'), (int, float, np.floating, np.integer)):
            model.encoder.trainable = False
            model.decoder.trainable = False

            for layer in model.decoders[0].layers:
                if layer.depth <= self.hp.get('training'):
                    layer.trainable = True
            for layer in model.encoders[0].layers:
                if layer.depth > self.hp.get('no_layers') + 2 - self.hp.get('training'):
                    layer.trainable = True
            for layer in model.encoders[1].layers:
                if layer.depth > self.hp.get('no_layers') + 2 - self.hp.get('training'):
                    layer.trainable = True

        with self.hp.conditional_scope('use_rotation', [True]):
            layer = model.decoder.get_layer(name='x_sin')
            layer.trainable = self.hp.get('use_rotation') and layer.trainable

    def set_state_obs(self, model, H=None, state=None):
        state_dim, hidden_dim = self.hp.get(
            'state_dim'), self.hp.get('hidden_dim')
        if state is None:
            state = np.zeros((hidden_dim,))
        if H is None:
            H = np.eye(hidden_dim, state_dim)

        model.encoder.layers[1].set_weights([H.T, state])

    def set_latent_obs(self, model, H=None, state=None):
        latent_dim, hidden_dim = self.hp.get(
            'latent_dim'), self.hp.get('hidden_dim')
        if state is None:
            state = np.zeros((hidden_dim),)
        if H is None:
            H = np.eye(hidden_dim, latent_dim)

        model.decoder.layers[1].set_weights([H.T, state])

    def _file_paths(self, file_path):
        return {'dir': os.path.split(file_path)[0],
                'name': os.path.split(file_path)[1],
                'encoder': file_path + '_encoder.tf',
                'decoder': file_path + '_decoder.tf',
                'hp': file_path + '_hp.pkl'}

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
        hp.Fixed('epochs', 50)
        hp.Int('batch_size', default=64, min_value=1, max_value=1024,
               sampling='log')
        hp.Float('lr_init', default=5e-3, min_value=5e-4, max_value=1e-2,
                 step=5e-4)

        # Basic layer setup
        hp.Int('no_layers', default=4, min_value=0, max_value=8, step=1)
        hp.Choice('training', ['offline', 'online', 'obs'], default='offline')
        hp.Int('no_nodes', default=64, min_value=2, max_value=1024,
               sampling='log')
        hp.Fixed('latent_dim', 2)
        hp.Fixed('state_dim', 2)
        hp.Fixed('hidden_dim', 2)

        # Fine network details
        hp.Int('mc_samples', min_value=1, max_value=10, step=1)
        hp.Float('l1', min_value=0., max_value=1.0e-2, step=1.0e-3)
        hp.Boolean('use_rotation', default=True)
        with hp.conditional_scope('use_rotation', [True]):
            hp.Float('l2_rotation', default=1.0e-2, min_value=1.0e-4,
                     max_value=1.0, sampling='log')

        return hp

    def _add_layer_combo(self, x, no):
        name = 'hidden{:02d}'.format(no)

        x = layers.Dense(self.hp.get('no_nodes'), name=name+'_dense',
                         kernel_regularizer=tf.keras.regularizers.L1(
                             self.hp.get('l1')),
                         kernel_initializer='he_normal',
                         )(x)

        x = layers.LeakyReLU(0.3, name=name+'_activation')(x)

        return x

    def _add_model_layers(self, input_layer, name):
        hidden_dim = self.hp.get('hidden_dim')
        nodes = self.hp.get('no_nodes')

        x = input_layer
        for n in range(self.hp.get('no_layers')):
            l1 = tf.keras.regularizers.L1(self.hp.get('l1'))
            x  = layers.Dense(nodes, name=f'hidden{n:02d}_{name}_dense',
                             kernel_regularizer=l1,
                             kernel_initializer='he_normal',
                             )(x)
            x  = layers.LeakyReLU(0.1, 
                                  name=f'hidden{n:02d}_{name}_activation')(x)

        return x

    def _copy_layer(self, layer_in, tensor_in, **kwargs):
        layer_out = layer_in.from_config(layer_in.get_config())
        tensor_out = layer_out(tensor_in)
        if hasattr(layer_out, 'trainable') and 'trainable' in kwargs:
            layer_out.trainable = kwargs['trainable']
        if hasattr(layer_in, 'get_weights'):
            layer_out.set_weights(layer_in.get_weights())
        return layer_out, tensor_out

    def _build_encoder(self, state_dim, latent_dim, var_min=.04**2):
        """ Build the encoder. """
        hidden_dim = self.hp.get('hidden_dim')
        self.log_var_min = tf.math.log(var_min)

        # Input processing.
        input_layer = keras.Input(shape=(state_dim,), name='x_input')
        x = input_layer

        # Hidden layers
        for n in range(self.hp.get('no_layers')):
            x = self._add_layer_combo(x, n)

        # Latent output
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean_rescale",
                              kernel_initializer='identity')(z_mean)

        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z_log_var = layers.Dense(latent_dim, name='z_var_rescale',
                                 kernel_initializer='identity')(z_log_var)

        z_sample = SamplingLayer(name='z_sample')([z_mean, z_log_var])

        encoder = keras.Model(input_layer, [z_mean, z_log_var, z_sample],
                              name="encoder")

        return encoder

    def _build_encoder_multi(self, state_dim, latent_dim):
        """ Build the encoder. """

        # Input to encoder
        input_layer = keras.Input(shape=(state_dim,), name='x_input')

        # Mean
        z_mean = self._add_model_layers(input_layer, 'z_mean')
        z_mean = layers.Dense(latent_dim, name="z_mean")(z_mean)
        z_mean = self.scale_layer(z_mean)

        # Var
        z_log_var = self._add_model_layers(input_layer, 'z_log_var')
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(z_log_var)
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

        return encoder, z_mean_model, z_var_model, z_sample_model

    def _build_encoder_inno(self, obs_dim, latent_dim):
        """ Encoder for innovations. """
        nodes = self.hp.get('no_nodes')

        # Input
        input_layer = layers.Input(shape=(obs_dim,), name='d_input')

        # Mean
        d_mean = self._add_model_layers(input_layer, name='d_mean')
        d_mean = layers.Dense(latent_dim, name='d_mean')(d_mean)
        d_mean = self.scale_layer(d_mean)

        # Var
        d_log_var = self._add_model_layers(input_layer, 'd_log_var')
        d_log_var = layers.Dense(latent_dim, name="d_log_var")(d_log_var)
        d_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='d_var_rescale')(d_log_var)

        # Sample
        d_sample = SamplingLayer(name='d_sample')([d_mean, d_log_var])

        iencoder = keras.Model(input_layer, [d_mean, d_log_var, d_sample],
                               name='iencoder')

        return iencoder

    def _build_decoder_inno(self, obs_dim, latent_dim):
        """ Decoder for innovations. """
        # Input
        input_layer = layers.Input(shape=(latent_dim,), name='e_input')

        # Mean
        e_mean = self._add_model_layers(input_layer, name='e_mean')
        e_mean = layers.Dense(obs_dim, name='e_mean')(e_mean)
        e_mean = self.scale_layer(e_mean)

        # Var
        e_log_var = self._add_model_layers(input_layer, 'e_log_var')
        e_log_var = layers.Dense(obs_dim, name="e_log_var")(e_log_var)
        e_log_var = VarScalingLayer(self.scale_layer,
                                    trainable=False,
                                    name='e_var_rescale')(e_log_var)

        #Rotatation (not used)
        e_sin = tf.zeros_like(e_mean)

        # Sample
        e_sample = SamplingLayer(name='e_sample')([e_mean, e_log_var])

        # Model
        idecoder = keras.Model(input_layer, [e_mean, e_log_var, e_sin, e_sample],
                               name='idecoder')

        return idecoder

    def _build_decoder(self, state_dim, latent_dim, var_min=.04**2):
        """ Build the decoder. """
        hidden_dim = self.hp.get('hidden_dim')

        # Input processing
        input_layer = keras.Input(shape=(latent_dim,), name='z_input')
        x = input_layer

        # Hidden layers
        for n in range(self.hp.get('no_layers')):
            x = self._add_layer_combo(x, n)

        # Output
        x_mean = layers.Dense(state_dim, name="x_mean")(x)

        x_log_var = layers.Dense(state_dim, name="x_log_var")(x)

        x_sin = layers.Dense(state_dim-1, name="x_sin", activation='tanh',
                             kernel_initializer='zeros',
                             bias_initializer='zeros',
                             trainable=False)(x)

        x_sample = SamplingLayer(name='x_sample')([x_mean, x_log_var])

        decoder = keras.Model(input_layer, [x_mean, x_log_var, x_sin, x_sample],
                              name="decoder")

        with self.hp.conditional_scope('use_rotation', [True]):
            decoder.layers[-1].trainable = self.hp.get('use_rotation')

        return decoder

    def _build_decoder_multi(self, state_dim, latent_dim):
        """ Build the encoder. """
        hidden_dim = self.hp.get('hidden_dim')

        # Input processing.
        input_layer = keras.Input(shape=(latent_dim,), name='z_input')
        trans_input = InvertScalingLayer(self.scale_layer,
                                         trainable=False,
                                         name='sampling_rescale')(input_layer)

        # Mean
        x_mean = self._add_model_layers(trans_input, 'x_mean')
        x_mean = layers.Dense(state_dim, name="x_mean")(x_mean)

        # Var
        #x_log_var = layers.Lambda(lambda x: tf.stop_gradient(x))(trans_input)
        x_log_var = self._add_model_layers(trans_input, 'x_log_var')
        x_log_var = layers.Dense(state_dim, name="x_log_var")(x_log_var)

        # sin IP
        x_sin = tf.zeros_like(x_mean[:, 0:1], name='x_sin')
        with self.hp.conditional_scope('use_rotation', [True]):
            x_sin = layers.Lambda(lambda x: tf.stop_gradient(x))(trans_input)
            x_sin = self._add_model_layers(x_sin, 'x_sin')
        x_sin = layers.Dense(state_dim-1, name="x_sin", activation='tanh',
                             kernel_initializer='zeros',
                             bias_initializer='zeros',
                             trainable=False)(x_sin)

        # Sample
        x_sample = SamplingLayer(name='x_sample')([x_mean, x_log_var])
        x_sample = rotate_layer(x_sample, x_sin)

        # Different models
        x_mean_model = keras.Model(input_layer, x_mean, name='decoder_mean')
        x_var_model = keras.Model(input_layer, x_log_var, name='decoder_var')
        x_sample_model = keras.Model(input_layer, x_sample, name='decoder_sample')
        decoder = keras.Model(input_layer, [x_mean, x_log_var, x_sin, x_sample],
                              name='decoder')

        return decoder, x_mean_model, x_var_model, x_sample_model


def rotate_layer(x, sin):
    """ Rotate over angle asin theta. """
    if x.shape[1] == 2:
        cos = tf.sqrt(1 - sin**2)
        x0 = cos[:, 0:1]*x[:, 0:1] - sin[:, 0:1]*x[:, 1:2]
        x1 = sin[:, 0:1]*x[:, 0:1] + cos[:, 0:1]*x[:, 1:2]
        return tf.keras.layers.Concatenate(axis=-1)([x0, x1])
    else:
        return x

# %% Tuning


def tune_DenseVae(x):
    """ Function to tune the hyperparameters in DenseVae. """

    hypermodel = DenseVae()
    file_dir = '/home/ivo/Code/VAE/tmp'

    # Writer logs
    tensorboard_writer = tf.keras.callbacks.TensorBoard(file_dir)

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
                                   objective=tuner.Objective(
                                       'loss', direction='min'),
                                   hyperparameters=hp,
                                   tune_new_entries=False,
                                   max_epochs=50,
                                   directory=file_dir,
                                   overwrite=True,
                                   max_retries_per_trial=1,
                                   hyperband_iterations=1)
    # Carry out the search
    architecture.search(x, callbacks=[tensorboard_writer])

    return architecture
