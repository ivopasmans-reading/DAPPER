#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:32:57 2024

Functions to run and train VAE on climatology. 

@author: ivo
"""

import tensorflow as tf
tf.config.experimental.list_physical_devices()
import numpy as np
import dapper.mods as modelling
import dapper.da_methods.ensemble as eda
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import circle_vae as vae
import shutil
import os, dill
import random
import keras
import xarray as xr


# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'
FIG_DIR = '/home/ivo/Figures/vae'
XP_NAMES = ['no DA','ETKF','single-clima','single-transfer',
            'double-clima','double-transfer']

# Number of ensemble member
Nens = 64

def run_model_default(K, dko, seed, obs_type='normal', amplitude=0.0, sigo=.1,
                      obs_func=None):
    """
    Function that creates the model for this experiment
    """

    Dyn = {'M': 2, 'model': circle.step_factory(amplitude=amplitude),
           'linear': circle.step_factory(amplitude=amplitude), 'noise': 0}

    # Actual observation operator.
    # Actual observation operator.
    if obs_func is None:
        obs_func = lambda e : e[0]
    obs = circle.create_obs_factory_func(obs_func, sigo)
    Obs = {'time_dependent': obs}

    # Time steps
    dt = 1
    tseq = modelling.Chronology(dt=dt, K=K, dko=dko, Tplot=K*dt, BurnIn=0)

    # State Space System setup.
    circle.X0.seed = seed
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, circle.X0)

    # Run the model
    reset_random_seeds(seed)
    xx, yy = HMM.simulate()
    climate = circle.data2pandas(xx)

    return HMM, xx, yy

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(0)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    circle.set_seed(seed)

def compare_layers(m0, m1):
    for l0, l1 in zip(m0.encoder.layers, m1.encoder.layers):
        for w0, w1 in zip(l0.get_weights(), l1.get_weights()):
            if np.any(w0 != w1):
                print(l0.name)

    for l0, l1 in zip(m0.decoder.layers, m1.decoder.layers):
        for w0, w1 in zip(l0.get_weights(), l1.get_weights()):
            if np.any(w0 != w1):
                print(l0.name)

class VaeExperiment:
    
    def save(self):
        with open(self.filepath,'wb') as stream:
            dill.dump(self.data, stream)
            
    def load(self):
        with open(self.filepath,'rb') as stream:
            self.data = dill.load(stream)

class ClimaExperiment(VaeExperiment):
    """ Run the climatology and use it for training weights. """
    
    def __init__(self, r_amplitude, run_model=run_model_default, do_plot=False):
        self.amplitude = r_amplitude
        self.do_plot = do_plot
        self.run_model = run_model
        
    def reset(self):
        self.seed = 1000 
        
    def __iter__(self):
        self.reset()
        return self 
    
    def __next__(self):
        self.create_model(self.seed)
        
        if os.path.exists(self.filepath):
            self.load(self.filepath)
        else:
            self.hypermodel.fit(self.hp, self.model, self.xx, 
                                verbose=False, shuffle=True)
            self.save(self.filepath)
            
        if self.do_plot:
            self.plot_clima()
        
        self.seed += 100
        return self
                
    @property 
    def filepath(self):
        """ Return default filepath for saving models."""
        A = int(self.amplitude*100)
        seed = int(self.seed)
        return os.path.join(MODEL_PATH, f'climaK3_{A:02d}_{seed:04d}.pkl')
                
    def create_model(self, seed):
        """ Run the climatology and train VAE."""

        # Generate climatology
        self.HMM, self.xx, _ = self.run_model(10000, 1, seed)

        # Create model
        builder = vae.StateCoderBuilder()
        self.hypermodel = vae.DenseVae(builder)
        self.hp = self.hypermodel.build_hp(no_layers=6, no_nodes=32, 
                                           use_rotation=False, batch_size=32, 
                                           latent_dim=1, mc_samples=1,
                                           architecture='clima',
                                           training_hidden=99,
                                           training_output_z=True,
                                           training_output_x=True,
                                           verbose=False)
        
        self.seed = seed
        reset_random_seeds(seed)
        self.model = self.hypermodel.build(self.hp)

    def save(self, filepath):
        with open(filepath, 'wb') as stream:
            dill.dump(self.model.get_weights(), stream)
            
    def transform_weights(self, wsaves, wmods):
        #Copy weights from Keras2 model to this new Keras3 model skipping
        #rotation layers. 
        m = -1
        for n,wmod in enumerate(wmods):
            m += 1
            if m>=30 and np.mod(m-30,6)==0:
                m += 2
            wmods[n] = wsaves[m]
            
        return wmods
            
    def load(self, filepath):
        with open(filepath,'rb') as stream:
            weights = dill.load(stream)   
           
        self.model.set_weights(weights)

    def plot_clima(self):
        filepath = self.filepath
        filepath = filepath.replace('.pkl', '.png')
        
        # Sample encoder
        dko = self.HMM.tseq.dko
        samples = self.xx[::dko]
        zz_mu, zz_sig, zz = self.model.encoder.predict([samples])
        zz_sig = np.exp(.5*zz_sig)

        # Sample decoder
        z = np.random.normal(size=(np.size(samples, 0), self.hp.get('latent_dim')))
        zxx_mu, zxx_sig, zxx_angle, zxx = self.model.decoder.predict(z)
        zxx_sig = np.exp(.5*zxx_sig)

        # Plot distributions
        plotReconstruction = plots.ReconstructionPlot(FIG_DIR)
        plotReconstruction.add_samples(zxx, zz)
        plotReconstruction.plot(filepath)
        plotReconstruction.save()
        
class XpsClass:
    
    def __init__(self, clima, HMM, Nens, No, names=XP_NAMES):
        self.names = names
        self.hp =  clima.hp 
        self.hypermodel = clima.hypermodel
        self.model = clima.model 
        self.factory = eda.EndaFactory()
        self.HMM = HMM
        self.No = No
        self.Nens = Nens
    
    def __iter__(self):
        self.names = iter(self.names)
        return self
    
    def __next__(self):
        name = self.names.__next__()
        if name is StopIteration:
            return StopIteration
        elif name=='no DA':
            xp = eda.EnDa(self.Nens, [], name='no DA')
        elif name=='ETKF':
            xp = self.factory.build(self.Nens, 'Sqrt svd', name='ETKF', rot=False)
        elif name=='single-transfer':
            bkg_trans = eda.BackgroundVaeTransform(self.hypermodel, self.hp, self.model)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='single-transfer',
                                    VaeTransforms=[bkg_trans])
        elif name=='double-transfer':
            bkg_trans = eda.BackgroundVaeTransform(self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp, self.model, self.No, 
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='double-transfer',
                                    VaeTransforms=[inno_trans,bkg_trans])
        elif name=='double-cycle':
            cycle_trans = eda.CyclingVaeTransform(self.hypermodel, self.hp, None)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='double-cycle',
                                    VaeTransforms=[cycle_trans])
        elif name=='single-clima':
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            xp = self.factory.build(self.Nens,'ETKF_D', No=self.No, name='single-clima',
                                    VaeTransforms=[vae_trans])
        elif name=='double-clima':
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp, self.model, self.No, 
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens,'ETKF_D', No=self.No, name='double-clima',
                                    VaeTransforms=[inno_trans,vae_trans])
        else:
            raise ValueError(f'{name} not a valid name for experiment.')
        
        return xp 