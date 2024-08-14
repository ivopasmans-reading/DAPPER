#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:18:07 2024

Experiment for paper with truth moving along unit circle and observational 
error from Gaussian. 

@author: ivo
"""

import keras
import tensorflow as tf
tf.config.experimental.list_physical_devices()

import shutil
import os
import dill
from datetime import datetime
import random
import xarray as xr
import importlib
import numpy as np

import dapper.mods as modelling
import dapper.da_methods as da
import dapper.da_methods.ensemble as eda
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import circle_vae as vae
from dapper.tools.seeding import set_seed

# Directory in which the figures will be stored.
FIG_DIR = '/home/ivo/Figures/vae/exp07'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'
# Number of ensemble member
Nens = 64

# Copy this file
if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR, 'experiment.py'))

def run_model(K, dko, seed, obs_type='normal', amplitude=0.2, sigo=.1):
    """
    Function that creates the model for this experiment
    """

    Dyn = {'M': 2, 'model': circle.step_factory(amplitude=amplitude),
           'linear': circle.step_factory(amplitude=amplitude), 'noise': 0}

    # Actual observation operator.
    obs_func = lambda e : np.exp(e[0]**3+e[1]**2)
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
    """ Reset different seeds. """
    os.environ['PYTHONHASHSEED'] = str(0)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    circle.set_seed(seed)


def compare_layers(m0, m1):
    """ Compare if all NN weights are equal. """
    for l0, l1 in zip(m0.encoder.layers, m1.encoder.layers):
        for w0, w1 in zip(l0.get_weights(), l1.get_weights()):
            if np.any(w0 != w1):
                print(l0.name)

    for l0, l1 in zip(m0.decoder.layers, m1.decoder.layers):
        for w0, w1 in zip(l0.get_weights(), l1.get_weights()):
            if np.any(w0 != w1):
                print(l0.name)


class VaeExperiment:
    """ Base class representing single run of experiment. """

    def save(self):
        with open(self.filepath, 'wb') as stream:
            dill.dump(self.data, stream)

    def load(self):
        with open(self.filepath, 'rb') as stream:
            self.data = dill.load(stream)

# %% Generate climatology

class ClimaExperiment(VaeExperiment):
    """ Run the climatology. """

    def __init__(self, r_amplitude):
        self.amplitude = r_amplitude

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
                                verbose=True, shuffle=True)
            self.save(self.filepath)

        #self.plot_clima()

        self.seed += 100
        return self

    @property
    def filepath(self):
        A = int(self.amplitude*100)
        seed = int(self.seed)
        return os.path.join(MODEL_PATH, f'climaK3_{A:02d}_{seed:04d}.pkl')

    def create_model(self, seed):
        """ Run the climatology and train VAE."""

        # Generate climatology
        self.HMM, self.xx, _ = run_model(10000, 1, seed, 
                                         amplitude=self.amplitude)

        # Create model
        builder = vae.StateCoderBuilder()
        self.hypermodel = vae.DenseVae(builder)
        self.hp = self.hypermodel.build_hp(no_layers=7, no_nodes=32, 
                                           use_rotation=False, batch_size=32, 
                                           latent_dim=1, mc_samples=1,
                                           architecture='clima',
                                           training_hidden=99,
                                           training_output_z=True,
                                           training_output_x=True)
        
        
        self.seed = seed
        reset_random_seeds(seed)
        self.model = self.hypermodel.build(self.hp)

    def save(self, filepath):
        with open(filepath, 'wb') as stream:
            dill.dump(self.model.get_weights(), stream)

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
        z = np.random.normal(size=(np.size(samples, 0), 
                                   self.hp.get('latent_dim')))
        zxx_mu, zxx_sig, zxx_angle, zxx = self.model.decoder.predict(z)
        zxx_sig = np.exp(.5*zxx_sig)

        # Plot distributions
        plotReconstruction = plots.ReconstructionPlot(FIG_DIR)
        plotReconstruction.add_samples(zxx, zz)
        plotReconstruction.plot(filepath)
        plotReconstruction.save()


climas = ClimaExperiment(0.0)
climas.reset()
climas = next(climas)

# %% Experiment oscillation

class XpsClass:
    """ Iterator over experiment. """

    def __init__(self, clima, HMM, Nens, No):
        self.names = ['no DA', 'ETKF', 'single-transfer', 'double-transfer',
                        'single-clima','double-clima']
        self.names = ['no DA', 'ETKF', 'single-clima', 'single-transfer']
        #self.names = ['double-clima','double-transfer']
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
        elif name == 'no DA':
            xp = eda.EnDa(self.Nens, [], name='no DA')
        elif name == 'ETKF':
            xp = self.factory.build(self.Nens, 'Sqrt svd', name='ETKF', rot=False)
        elif name == 'single-transfer':
            bkg_trans = eda.BackgroundVaeTransform(self.hypermodel, self.hp, self.model)
            
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='single-transfer',
                                    VaeTransforms=[bkg_trans])
        elif name == 'double-transfer':
            bkg_trans = eda.BackgroundVaeTransform(self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp, self.model, self.No,
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='double-transfer',
                                    VaeTransforms=[inno_trans, bkg_trans])
        elif name == 'double-cycle':
            cycle_trans = eda.CyclingVaeTransform(self.hypermodel, self.hp, None)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='double-cycle',
                                    VaeTransforms=[cycle_trans])
        elif name == 'single-clima':
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='single-clima',
                                    VaeTransforms=[vae_trans])
        elif name == 'double-clima':
            vae_trans = eda.VaeTransform(self.hypermodel, self.hp, self.model)
            inno_trans = eda.InnoVaeTransform(self.hypermodel, self.hp, self.model, self.No,
                                              self.HMM.Obs(0).noise.add_sample)
            xp = self.factory.build(self.Nens, 'ETKF_D', No=self.No, name='double-clima',
                                    VaeTransforms=[inno_trans, vae_trans])
        else:
            raise ValueError(f'{name} not a valid name for experiment.')

        return xp

class OscillateExperiment(VaeExperiment):
    """ Experiment in which truth runs over unit circle with varying radius. """

    def __init__(self, N, dko, save_name='k3oscillation07.pkl'):
        self.dko = dko
        self.Nclima = max(1, int(np.sqrt(N)))
        self.N = int(N / self.Nclima)
        self.No = Nens * 4  # IP
        self.run_time = 500
        self.save_name = save_name

        self.climas = iter(ClimaExperiment(0.0))
        self.create_data()
        self.fails = []
        
    def run(self):
        # Run all models repeatedly
        climas = (next(self.climas) for _ in range(0, self.Nclima))
        for clima in climas: 
            for n in range(0, self.N):
                self.seed = clima.seed + n * 7
                self.run1(clima)
                
            del(clima)

    def check_done(self, xp):
        rmse = self.data_ana['rmse']

        if len(rmse) == 0:
            return False

        rmse = rmse['x']
        try:
            x = rmse.sel(experiment=xp.name, seed=self.seed)
        except:
            return False

        return not np.any(np.isnan(x.data))

    def run1(self, clima):
        # Create new run.
        reset_random_seeds(self.seed-100)
        HMM, xx, yy = run_model(self.run_time, self.dko, self.seed-100)

        # Create experiments
        self.xx, self.yy = xx, yy
        self.xps = iter(XpsClass(clima, HMM, Nens, self.No))

        for xp in self.xps:
            
            #Test if experiment has been loaded from file. 
            if (xp.name, self.seed) in self.done:
                print('\nDONE ', xp.name, self.seed)
                del(xp)
                continue
            else:
                print('\nRUNNING ', xp.name, self.seed)

            # Run the DA experiment
            reset_random_seeds(self.seed)
            # Needs to be here to prevent running with same see.
            _, _, _ = run_model(self.run_time, self.dko, self.seed)
            xp.HMM = HMM
            try:
                xp.assimilate(HMM, xx, yy, liveplots=False)
            except:
                self.fails.append((clima, xp, self.seed))
                continue

            # Calculate CRPS and save in Xarray.
            self.load()
            assert hasattr(self,'data_for')
            assert hasattr(self,'data_ana')
            for key, value in self.keys.items():
                kwargs = {'xp': xp, 'xx': xx,
                          'seed': self.seed, 'stage': 'forecast'}
                stat = plots.calculate_stat(value, **kwargs)
                self.data_for[key] = xr.merge([self.data_for[key], stat])

                kwargs = {'xp': xp, 'xx': xx,
                          'seed': self.seed, 'stage': 'analysis'}
                stat = plots.calculate_stat(value, **kwargs)
                self.data_ana[key] = xr.merge([self.data_ana[key], stat])
            
            #Save output
            self.save()
            self.done += [(xp.name, self.seed)]

            #Delete models to free memory. 
            del(self.data_for, self.data_ana)
            if hasattr(xp, 'hypermodel'):
                xp.hypermodel.clear()
            keras.backend.clear_session(free_memory=True)

    @property
    def filepath(self):
        return os.path.join(MODEL_PATH, self.save_name)

    def save(self):
        with open(self.filepath, 'wb') as stream:
            dill.dump((self.data_for, self.data_ana), stream)
            
            
    def create_data(self):
        self.keys = dict([('crps', plots.CRPS), ('histogram', plots.Histogram),
                          ('rmse', plots.EnsError)])
        self.data_for = dict([(key, xr.Dataset()) for key in self.keys])
        self.data_ana = dict([(key, xr.Dataset()) for key in self.keys])
        self.done = []

    def load(self):
        self.create_data()
        
        if not os.path.exists(self.filepath):
            return
        
        with open(self.filepath, 'rb') as stream:
            self.data_for, self.data_ana = dill.load(stream)
            
        if len(self.data_ana['rmse'])>0:
            #Check for which combinations (experiment,seed) all values are non-nan
            isnull = self.data_ana['rmse']['x'].isnull()
            coords = set(isnull.coords) - set(['seed','experiment'])
            isnull = isnull.reduce(lambda x, axis : np.any(x, axis=axis), dim=coords)
            self.done = [(xp, seed) for xp in list(isnull['experiment'].data)
                         for seed in list(isnull['seed'].data)
                         if not isnull.sel(experiment=xp, seed=seed)]

    def delete(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


# Run the experiment.
exp = OscillateExperiment(49, 10)
exp.load()
exp.run()

# %% Plot output statistics.

# Remove faulty 1200<=seed<1300
def filter_data(data):
    seeds = data.coords['seed']
    seeds = [s for s in seeds if s < 1200 or s >= 1300]
    return data.sel(seed=seeds)

for stage, data in zip(['forecast', 'analysis'], [exp.data_for, exp.data_ana]):

    plot_data = filter_data(data['histogram'])
    plotHist = plots.ProbDensityPlots(FIG_DIR, plot_data)
    plotHist.plot_scatter_density('scatter_'+stage)
    plotHist.save()
    
    plot_data = filter_data(data['histogram'])
    plotHist = plots.ErrorProbDensityPlots(FIG_DIR, plot_data)
    plotHist.plot_scatter_density('error_scatter_'+stage)
    plotHist.save()

    plot_data = filter_data(data['crps'])
    plotHist = plots.CrpsPlots(FIG_DIR, plot_data)
    plotHist.plot_crps('crps_'+stage)
    plotHist.save()

    plot_data = filter_data(data['rmse'])
    plotHist = plots.TaylorPlots(FIG_DIR, plot_data)
    plotHist.plot_taylor('taylor_'+stage)
    plotHist.save()
    
    plot_data = filter_data(data['crps'])
    plotHist = plots.SingleCrpsPlots(FIG_DIR, plot_data)
    plotHist.plot_crps('crps_single_'+stage)
    plotHist.save()

# %% Generate animation

def plot_movie(experiments, run_time, dko, No=Nens*4):
    climas = iter(ClimaExperiment(0.0))
    for n in range(1):
        clima = climas.__next__()

    # Create new run.
    reset_random_seeds(clima.seed-100)
    HMM, xx, yy = run_model(run_time, dko, clima.seed-100)
    xps = iter(XpsClass(clima, HMM, Nens, No))

    for xp in xps:
        print('XP ', xp.name)
        if xp.name not in experiments:
            continue
        # Run
        _, _, _ = run_model(run_time, dko, clima.seed)
        xp.HMM = HMM
        xp.assimilate(HMM, xx, yy, liveplots=False)
        # Plot
        circle = plots.CirclePlot(FIG_DIR)
        circle.add_track(xp.name, xp.HMM.tseq.tt, xx)
        circle.add_obs(xp.HMM.tseq.tto, yy)
        circle.add_ens_for(xp.name, xp.HMM.tseq.tto, xp.stats.E.f)
        circle.add_ens_ana(xp.name, xp.HMM.tseq.tto, xp.stats.E.a)
        circle.animate_time(xp.HMM.tseq.tto, fig_name='movie_'+xp.name)


plot_movie(['ETKF', 'single-transfer', 'double-transfer', 
            'single-clima','double-clima'], 500, 10)
