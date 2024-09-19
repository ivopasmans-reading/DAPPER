#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:32:57 2024

Functions to run and train VAE on climatology. 

@author: ivo
"""

import sys, os, dill, dataclasses, random, shutil
from typing import Callable
import xarray as xr
import numpy as np

from dapper.vae import circle_vae as vae
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.mods import ComplexCircle as circle
import dapper.da_methods.ensemble as eda
import dapper.mods as modelling

import keras 
import tensorflow as tf
from numba import cuda
tf.config.experimental.list_physical_devices()

# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'
# File path used to save figures. 
FIG_DIR = '/home/ivo/Figures/vae'
# Default names of the experimental configurations.
XP_NAMES = ['no DA', 'ETKF', 'single-clima', 'single-transfer',
            'double-clima', 'double-transfer']
# Number of ensemble member
Nens = 64

#%% General functions. 

def clear():
    """ Remove model from memory. """
    print('CLEAR')
    keras.backend.clear_session(free_memory=True)
    device = cuda.get_current_device()
    device.reset()
    cuda.close()
    import tensorflow as tf

def filter_data(data):
    """ Remove certain seeds from data. """
    seeds = data.coords['seed']
    seeds = [s for s in seeds if s < 1200 or s >= 1300]
    return data.sel(seed=seeds)

def assert_gpu_active():
    "Check whether the GPU is active."
    devices = tf.config.experimental.list_physical_devices()
    if not any([device.device_type == 'GPU' for device in devices]):
        raise RuntimeError("GPU not active.")
        
def reset_random_seeds(seed):
    """ Reset all the seeds used. """
    os.environ['PYTHONHASHSEED'] = str(0)
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    circle.set_seed(seed)
    
def compare_layers(m0, m1):
    """ Compare two layers of VAE decoder/encoder. """
    for l0, l1 in zip(m0.encoder.layers, m1.encoder.layers):
        for w0, w1 in zip(l0.get_weights(), l1.get_weights()):
            if np.any(w0 != w1):
                print(l0.name)

    for l0, l1 in zip(m0.decoder.layers, m1.decoder.layers):
        for w0, w1 in zip(l0.get_weights(), l1.get_weights()):
            if np.any(w0 != w1):
                print(l0.name)
                
#%% DAPPER model class. 

@dataclasses.dataclass 
class DapperModel:
    """ Class representing a model in DAPPER. """
    obs_func : Callable[[float], float] = lambda e : e[0]
    obs_type : str = 'normal'
    obs_sig : float = 0.1
    amplitude : float = 0.0
    
    def __call__(self, K, dko, seed):
        """ Run the DAPPER model and create truth and observations. """ 
        
        #DAPPER dynamic model object. 
        Dyn = {'M': 2, 'model': circle.step_factory(amplitude=self.amplitude),
               'linear': circle.step_factory(amplitude=self.amplitude), 'noise': 0}
        
        #DAPPER observation operator. 
        obs = circle.create_obs_factory(self.obs_func, self.obs_sig, 
                                        distribution=self.obs_type)
        Obs = {'time_dependent': obs}
        
        # Time steps model. 
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
    
#%% Classes representing experiments. 

class VaeExperiment:
    """ 
    Base class used to run experiments. 
    """
    
    @property
    def filepath(self):
        """ Filepath to save data."""
        return os.path.join(MODEL_PATH, self.save_name)
    
    def save(self):
        """ Save data to file set by filepath. """
        with open(self.filepath, 'wb') as stream:
            dill.dump((self.data_for, self.data_ana), stream)

    def load(self):
        """ Load data from file set by filepath. """
        self.create_data()

        if not os.path.exists(self.filepath):
            return

        with open(self.filepath, 'rb') as stream:
            self.data_for, self.data_ana = dill.load(stream)

        if len(self.data_ana['rmse']) > 0:
            # Check for which combinations (experiment,seed) all values are non-nan
            isnull = self.data_ana['rmse']['x'].isnull()
            coords = set(isnull.coords) - set(['seed', 'experiment'])
            isnull = isnull.reduce(
                lambda x, axis: np.any(x, axis=axis), dim=coords)
            self.done = [(xp, seed) for xp in list(isnull['experiment'].data)
                         for seed in list(isnull['seed'].data)
                         if not isnull.sel(experiment=xp, seed=seed)]

    def delete(self):
        """ Delete file. """
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            
    def _in_seed_range(self, seed):
        """ Check whether a model run for the seed should be executed. """
        if len(sys.argv) != 3:
            return True
        elif seed >= int(sys.argv[1]) and seed < int(sys.argv[2]):
            return True
        else:
            return False

    def create_data(self):
        """ Create storage for save data. """
        self.keys = dict([('crps', plots.CRPS),
                          ('histogram', plots.Histogram),
                          ('rmse', plots.EnsError),
                          ('time_error', plots.EnsTimeError)],
                         )
        self.data_for = dict([(key, xr.Dataset()) for key in self.keys])
        self.data_ana = dict([(key, xr.Dataset()) for key in self.keys])
        self.done = []

    
@dataclasses.dataclass
class ClimaExperiment(VaeExperiment):
    """
    Iterator that runs over different climatologies and trains VAE for each of
    them. 
    """
    amplitude : float = 1.0
    initial_seed : int = 1000
    do_plot : bool = False
    model : DapperModel = dataclasses.field(lambda : DapperModel())
    hp_parameters : dict = dataclasses.field(dict)
    
    def __post_init__(self):
        self.hp_parameters = {'no_layers' : 6,
                              'no_nodes' : 32,
                              'use_rotation' : False,
                              'batch_size' : 32,
                              'latent_dim' : 1,
                              'mc_samples' : 1,
                              'architecture' : 'clima',
                              'training_hidden' : 99,
                              'training_output_z' : True, 
                              'training_output_x' : True, 
                              'verbose' : False,
                              **self.hp_parameters}
    
    def reset(self):
        self.seed = self.initial_seed + 0

    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        self.create_model(self.seed)

        if os.path.exists(self.filepath):
            self.load(self.filepath)
        else:
            self.hypermodel.fit(self.hp, self.hypermodel, self.xx,
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
    
    def save(self, filepath):
        with open(filepath, 'wb') as stream:
            dill.dump(self.model.get_weights(), stream)
            
    def load(self, filepath):
        with open(filepath, 'rb') as stream:
            weights = dill.load(stream)

        self.model.set_weights(weights)
        
    def transform_weights(self, wsaves, wmods):
        """
        Copy weights from Keras2 model to this new Keras3 model skipping
        rotation layers. This is generally not used. 
        """
        m = -1
        for n, wmod in enumerate(wmods):
            m += 1
            if m >= 30 and np.mod(m-30, 6) == 0:
                m += 2
            wmods[n] = wsaves[m]

        return wmods
    
    def plot_clima(self):
        """ Plot figure with decoder and encoder reconstructions. """
        filepath = self.filepath
        filepath = filepath.replace('.pkl', '.png')

        # Sample encoder
        dko = self.HMM.tseq.dko
        samples = self.xx[::dko]
        zz_mu, zz_sig, zz = self.model.encoder.predict([samples])
        zz_sig = np.exp(.5*zz_sig)

        # Sample decoder
        z = np.random.normal(
            size=(np.size(samples, 0), self.hp.get('latent_dim')))
        zxx_mu, zxx_sig, zxx_angle, zxx = self.model.decoder.predict(z)
        zxx_sig = np.exp(.5*zxx_sig)

        # Plot distributions
        plotReconstruction = plots.ReconstructionPlot(FIG_DIR)
        plotReconstruction.add_samples(zxx, zz)
        plotReconstruction.plot(filepath)
        plotReconstruction.save()
        
    def create_hypermodel(self, seed):
        """ Run the climatology and train VAE."""

        # Generate climatology
        self.HMM, self.xx, _ = self.model(10000, 1, seed)

        # Create the VAE-hypermodel parameters. 
        builder = vae.StateCoderBuilder()
        self.hypermodel = vae.DenseVae(builder)
        self.hp = self.hypermodel.build_hp(**self.hp_parameters)

        # Create the VAE-hypermodel
        self.seed = seed
        reset_random_seeds(self.seed)
        self.hypermodel = self.hypermodel.build(self.hp)

class OldClimaExperiment(VaeExperiment):
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
        """
        Copy weights from Keras2 model to this new Keras3 model skipping
        rotation layers. This is generally not used. 
        """
        m = -1
        for n, wmod in enumerate(wmods):
            m += 1
            if m >= 30 and np.mod(m-30, 6) == 0:
                m += 2
            wmods[n] = wsaves[m]

        return wmods

    def load(self, filepath):
        with open(filepath, 'rb') as stream:
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
        z = np.random.normal(
            size=(np.size(samples, 0), self.hp.get('latent_dim')))
        zxx_mu, zxx_sig, zxx_angle, zxx = self.model.decoder.predict(z)
        zxx_sig = np.exp(.5*zxx_sig)

        # Plot distributions
        plotReconstruction = plots.ReconstructionPlot(FIG_DIR)
        plotReconstruction.add_samples(zxx, zz)
        plotReconstruction.plot(filepath)
        plotReconstruction.save()


class XpsClass:
    """
    This is an Iterator that creates different DAPPER experiments. 
    """

    def __init__(self, clima, HMM, Nens, No, names=XP_NAMES):
        self.names = names
        self.hp = clima.hp
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
        elif 'no DA' in name:
            xp = eda.EnDa(self.Nens, [], name='no DA')
        elif 'ETKF0' in name:
            xp = eda.EnKF('Sqrt', self.Nens)
            xp.name = name
        elif 'ETKF' in name:
            xp = self.factory.build(
                self.Nens, 'Sqrt svd', name=name, rot=False)
        elif 'single-transfer' in name:
            bkg_trans = eda.BackgroundVaeTransform(
                self.hypermodel, self.hp, self.model)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[bkg_trans])
        elif 'double-transfer' in name:
            bkg_trans = eda.BackgroundVaeTransform(
                self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp, self.model, self.No,
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[inno_trans, bkg_trans])
        elif 'double-cycle' in name:
            cycle_trans = eda.CyclingVaeTransform(
                self.hypermodel, self.hp, None)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[cycle_trans])
        elif 'single-clima' in name:
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[vae_trans])
        elif 'double-clima' in name:
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp, self.model, self.No,
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[inno_trans, vae_trans])
        else:
            raise ValueError(f'{name} not a valid name for experiment.')

        return xp
