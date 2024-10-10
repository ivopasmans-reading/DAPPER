#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:32:57 2024

Functions to run and train VAE on climatology. 

@author: ivo
"""

import sys, re
import os
import dill
import dataclasses
import random
import shutil
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

# %% General functions.


def clear():
    """ Remove model from memory. """
    print('CLEAR')
    keras.backend.clear_session(free_memory=True)
    device = cuda.get_current_device()
    device.reset()
    cuda.close()
    import tensorflow as tf

def assert_gpu_active():
    "Check whether the GPU is active."
    devices = tf.config.experimental.list_physical_devices()
    if not any([device.device_type == 'GPU' for device in devices]):
        raise RuntimeError("GPU not active.")


def reset_random_seeds(seed):
    """ Reset all the seeds used. """
    seed = int(seed)
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

# %% DAPPER model class.


@dataclasses.dataclass
class DapperModel:
    """ Class representing a model in DAPPER. """
    obs_func : Callable[[float], float] = lambda e: e[0]
    obs_type : tuple = ('normal',)
    obs_sig : float = 0.1
    amplitude : float = 0.0
    rotation_rate : float = 0.1

    def __call__(self, K, dko, seed):
        """ Run the DAPPER model and create truth and observations. """

        # DAPPER dynamic model object.
        Dyn = {'M': 2, 'model': circle.step_factory(amplitude=self.amplitude, 
                                                    rotation_rate=self.rotation_rate),
               'linear': circle.step_factory(amplitude=self.amplitude,
                                             rotation_rate=self.rotation_rate), 'noise': 0}

        # DAPPER observation operator.
        obs = circle.create_obs_factory(self.obs_func, self.obs_sig,
                                        distribution=self.obs_type)
        Obs = {'time_dependent': obs}

        # Time steps model.
        dt = 1
        tseq = modelling.Chronology(dt=dt, K=K, dko=dko, 
                                    Tplot=K*dt, BurnIn=0)

        # State Space System setup.
        circle.X0.seed = seed
        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, circle.X0)

        # Run the model
        reset_random_seeds(seed)
        xx, yy = HMM.simulate()
        climate = circle.data2pandas(xx)

        return HMM, xx, yy

# %% Classes representing experiments.

class VaeExperiment:
    """ 
    Base class used to run experiments. 
    """

    @property
    def filepath(self):
        """ Filepath to save data."""
        return os.path.join(MODEL_PATH, self.save_name+'.pkl')
    
    @property
    def filepath_output(self):
        """ Filepath to save data."""
        return os.path.join(MODEL_PATH, self.save_name+'_output'+'.nc')

    def save(self):
        """ Save data to file set by filepath. """
        filedir = os.path.dirname(self.filepath)
        if not os.path.exists(filedir):
            os.mkdir(filedir)
            
        with open(self.filepath, 'wb') as stream:
            dill.dump((self.data_for, self.data_ana), stream)
            
        self.outputs.to_netcdf(self.filepath_output)

    def load(self):
        """ Load data from file set by filepath. """
        if not os.path.exists(self.filepath):
            return
        
        if os.path.exists(self.filepath_output):
            self.outputs = xr.open_dataset(self.filepath_output)

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

@dataclasses.dataclass
class ClimaExperiment(VaeExperiment):
    """
    Iterator that runs over different climatologies and trains VAE for each of
    them. 
    """
    initial_seed: int = 1000
    do_plot: bool = False
    da_model: DapperModel = dataclasses.field(default_factory = lambda : DapperModel())
    hp_parameters: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.hp_parameters = {'no_layers': 6,
                              'no_nodes': 32,
                              'use_rotation': False,
                              'batch_size': 32,
                              'latent_dim': 1,
                              'mc_samples': 1,
                              'architecture': 'clima',
                              'training_hidden': 99,
                              'training_output_z': True,
                              'training_output_x': True,
                              'verbose': False,
                              **self.hp_parameters}

    def reset(self):
        self.seed = self.initial_seed + 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        self.create_hypermodel(self.seed)
        self.seed += 100
        return self

    @property
    def filepath(self):
        """ Return default filepath for saving models."""
        A = int(self.da_model.amplitude*100)
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
    
    def _run_da_model(self, seed):
        self.HMM, self.xx, _ = self.da_model(10000, 1, seed)

    def plot_clima(self):
        """ Plot figure with decoder and encoder reconstructions. """
        filepath = self.filepath
        filepath = filepath.replace('.pkl', '.png')
        
        #Create the truth run 
        self._run_da_model(self.seed)

        # Sample encoder
        dko = self.HMM.tseq.dko
        samples = self.xx[::dko]
        zz_mu, zz_sig, zz = self.hypermodel.encoder.predict([samples])
        zz_sig = np.exp(.5*zz_sig)

        # Sample decoder
        z = np.random.normal(
            size=(np.size(samples, 0), self.hp.get('latent_dim')))
        zxx_mu, zxx_sig, zxx_angle, zxx = self.hypermodel.decoder.predict(z)
        zxx_sig = np.exp(.5*zxx_sig)

        # Plot distributions
        plotReconstruction = plots.ReconstructionPlot(FIG_DIR)
        plotReconstruction.add_samples(zxx, zz)
        plotReconstruction.plot(filepath)
        plotReconstruction.save()

    def create_hypermodel(self, seed):
        """ Run the climatology and train VAE."""
        self.seed = int(seed)
        
        # Create the VAE-hypermodel parameters.
        builder = vae.StateCoderBuilder()
        self.hypermodel = vae.DenseVae(builder)
        self.hp = self.hypermodel.build_hp(**self.hp_parameters)

        # Create the VAE that actually holds the weights. 
        reset_random_seeds(self.seed)
        self.model = self.hypermodel.build(self.hp)
        
        if os.path.exists(self.filepath):
            self.load(self.filepath)
        else:
            # Generate climatology
            self._run_da_model(self.seed)

            # Fit hypermodel to output
            self.hypermodel.fit(self.hp, self.model, self.xx,
                                verbose=False, shuffle=True)
            #self.save(self.filepath)
            
        return self
    
@dataclasses.dataclass
class XpsClass:
    """
    This is an Iterator that creates different DAPPER experiments. 
    """
    clima: ClimaExperiment
    HMM: modelling.HiddenMarkovModel
    Nens: int = 64
    No: int = 256
    names: list = dataclasses.field(default_factory=lambda : XP_NAMES)
    factory: eda.EndaFactory = dataclasses.field(default_factory=lambda: eda.EndaFactory())

    def __post_init__(self):
        self.hp = self.clima.hp
        self.hypermodel = self.clima.hypermodel
        self.model = self.clima.model

    def __iter__(self):
        self.names = iter(self.names)
        return self

    def __next__(self):
        name = self.names.__next__()
        return self.create_xp(name)
        
    def create_xp(self, name):
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
            bkg_trans = eda.BackgroundVaeTransform(self.hypermodel, self.hp,
                                                   self.model)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[bkg_trans])
        elif 'double-transfer' in name:
            bkg_trans = eda.BackgroundVaeTransform(self.hypermodel, 
                                                   self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp,
                                              self.model, self.No,
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[inno_trans, bkg_trans])
        elif 'double-cycle' in name:
            cycle_trans = eda.CyclingVaeTransform(self.hypermodel, self.hp, None)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[cycle_trans])
        elif 'single-clima' in name:
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[vae_trans])
        elif 'double-clima' in name:
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp,
                                              self.model, self.No,
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name=name,
                                    VaeTransforms=[inno_trans, vae_trans])
        else:
            raise ValueError(f'{name} not a valid name for experiment.')

        return xp

@dataclasses.dataclass
class DaExperiment(VaeExperiment):
    save_name: str
    Nclima: int = 7
    Nruns: int = 7
    run_time: int = 500
    dko: int = 10
    da_model: DapperModel = dataclasses.field(default_factory=lambda: DapperModel())
    xp_parameters : dict = dataclasses.field(default_factory=dict)
    clima_parameters : dict = dataclasses.field(default_factory=dict)
    initial_seed : int = 1000
    
    def __post_init__(self):
        self.metrics = dict([('crps', plots.CRPS),
                             ('histogram', plots.Histogram),
                             ('rmse', plots.EnsError),
                             ('time_error', plots.EnsTimeError)])
        self.data_for = dict([(key, xr.Dataset()) for key in self.metrics])
        self.data_ana = dict([(key, xr.Dataset()) for key in self.metrics])
        self.outputs = xr.Dataset()
        self.done = []

    @property
    def seed_list(self):
        clima_seeds = np.arange(self.Nclima) * 100 
        xp_seeds = np.arange(self.Nruns) * 7 
        seed_list = [(self.initial_seed + clima_seed, 
                      self.initial_seed + clima_seed + xp_seed) for clima_seed
                     in clima_seeds for xp_seed in xp_seeds]
        return seed_list
    
    def run_climas(self):
        return [ClimaExperiment(**self.clima_parameters).create_hypermodel(seed)
                for seed in np.unique([seed for seed,_ in self.seed_list])]
    
    def run_seed(self, clima_seed, xp_seed):
        clima = ClimaExperiment(**self.clima_parameters).create_hypermodel(clima_seed)
        HMM, xx, yy = self.da_model(self.run_time, self.dko, xp_seed-100)
        xps = XpsClass(clima, HMM, **self.xp_parameters)
        completed = []
        
        for xp in iter(xps):
            if (xp.name, xp_seed) not in self.done:
                completed.append(self.run_xp(xp, xp_seed, HMM, xx, yy))
            
        return completed
        
    def run_xp(self, xp, seed, HMM, xx, yy):            
        # Run the DA experiment
        reset_random_seeds(seed)
        _, _, _ = self.da_model(self.run_time, self.dko, seed)
        xp.HMM = HMM
        try:
            print('RUNNING ', xp.name, seed)
            xp.assimilate(HMM, xx, yy, liveplots=False)
            xp.seed = seed
        except:
            raise RuntimeError((f"Experiment {xp.name} seed {seed} "
                                "failed to complete."))

        # Calculate CRPS and save in Xarray.
        for key, value in self.metrics.items():
            stat = plots.calculate_stat(value, xp=xp, xx=xx, seed=seed,
                                        stage='forecast')
            self.data_for[key] = xr.merge([self.data_for[key], stat])
            
            stat = plots.calculate_stat(value, xp=xp, xx=xx, seed=seed,
                                        stage='analysis')
            self.data_ana[key] = xr.merge([self.data_ana[key], stat])
            
        output = plots.calculate_output(xp)
        self.outputs = xr.merge([self.outputs, output])
            
        return xp
    
    def run(self):
        for clima_seed, xp_seed in self.seed_list:
            self.run_seed(clima_seed, xp_seed)
            
    def save_xps(self, xps):
        seed = xps[0].seed
        filepath = os.path.join(MODEL_PATH, f"{self.save_name}_xps_{seed:d}.nc") 
        filedir = os.path.dirname(filepath)
        
        if not os.path.join(filedir):
            os.mkdir(filedir)
            
        outputs = xr.merge([plots.calculate_output(xp) for xp in xps])
        outputs.to_netcdf(filepath)
            
        return outputs
            
#%% Function to run experiment from command line if necessary. 

def collect_files(exp, filedir):
    """ Combine all .pkl files into a single one."""
    filepath = filedir+'.pkl'
    filepath_output = filedir+'_output.nc'
    
    data_for = dict([(key, xr.Dataset()) for key in exp.metrics])
    data_ana = dict([(key, xr.Dataset()) for key in exp.metrics])
    outputs = xr.Dataset()
    for file in os.listdir(filedir):
        if re.match('[0-9]+_[0-9]+.pkl', file) is not None:
            with open(os.path.join(filedir,file),'rb') as stream:
                data_for1, data_ana1 = dill.load(stream)
            
                for key in exp.metrics:
                    data_for[key] = xr.merge([data_for[key], data_for1[key]])
                    data_ana[key] = xr.merge([data_ana[key], data_ana1[key]])
        elif re.match('[0-9]+_[0-9]+_output.nc', file) is not None:
            outputs1 = xr.open_dataset(os.path.join(filedir,file))
            outputs  = xr.merge([outputs, outputs1])
     
    #Save statistics
    with open(filepath,'wb') as stream:
        dill.dump((data_for, data_ana), stream)
    #Save ensemble output. 
    outputs.to_netcdf(filepath_output)
    
    
def run_exp(exp):
    """ Run the experiments. """
    exp_name = exp.save_name
    filepath = lambda clima_seed,xp_seed : os.path.join(exp_name, f"{clima_seed:d}_{xp_seed:d}")
    
    if len(sys.argv)>=2:
        index = int(sys.argv[1])
    else: 
        index = None
        
    if index is None:
        #Run all repetitions.
        exp.load()
        for clima_seed, xp_seed in exp.seed_list:
            exp.run_seed(clima_seed, xp_seed)
        exp.save()
    elif index>=len(exp.seed_list):
        #no more experiments to run
        sys.exit(1)
    elif index<0:
        #Storage for combined file.
        exp.save_name = exp_name
        collect_files(exp, exp.filepath.rsplit('.')[0])
    else:
        #Run the seed
        clima_seed, xp_seed = exp.seed_list[index]
        exp.save_name = filepath(clima_seed, xp_seed)
        
        exp.load()
        exp.run_seed(clima_seed, xp_seed)
        exp.save()
        
        sys.exit(0)
        
