#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:18:07 2024

Experiment for paper with truth moving along unit circle and observational 
error from Gaussian. 

@author: ivo
"""

import tensorflow as tf
import importlib
import numpy as np
import dapper.mods as modelling
import dapper.da_methods as da
import dapper.da_methods.ensemble as eda
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import basic as vae
from dapper.tools.seeding import set_seed
import shutil
import os, dill
from datetime import datetime
from sklearn import preprocessing
import scipy
import random
import keras
import xarray as xr

# Directory in which the figures will be stored.
FIG_DIR = '/home/ivo/Figures/vae/tensorboard'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/tensorboard'
LOG_DIR = os.path.join(MODEL_PATH,'logs')
# Number of ensemble member
Nens = 64

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# Copy this file
if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR, 'experiment.py'))


def run_model(K, dko, seed, obs_type='normal', amplitude=0.0, sigo=.1):
    """
    Function that creates the model for this experiment
    """

    Dyn = {'M': 2, 'model': circle.step_factory(amplitude=amplitude),
           'linear': circle.step_factory(amplitude=amplitude), 'noise': 0}

    # Actual observation operator.
    obs = circle.create_obs_factory([0], sigo, obs_type)
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

#%% CalibrateNo

class CalibrateNo(VaeExperiment):
    """ Compare error ETKF_D to ETKF. """
    
    def __init__(self, N=1, filepath=None):
        self.N = N
        self.data = None
        self.factory = eda.EndaFactory()
        
        if filepath is None:
            self.filepath = os.path.join(FIG_DIR,'CalibrateNo')
        else:
            self.filepath = filepath
            
    def run(self, dko=5):
        "Repeat experiment several times"
        for seed in range(1000, 1000+self.N*100, 100):
            self.run1(dko, seed)

    def run1(self, dko, seed):
        "Calculate results 1 experiment"
        HMM, xx, yy = run_model(1000, dko, seed)
        _, _, _ = run_model(1000, dko, seed+50)
        
        datas = xr.Dataset()
        
        def run_xp(xp, No):
            xp.HMM = HMM
            plotCRPS = plots.EnsStatsPlots(FIG_DIR)
            plotCRPS.add_truth(HMM, xx)
            xp.assimilate(HMM, xx, yy, liveplots=False)
            plotCRPS.add_xp(xp)
            
            crps, rmse = plotCRPS.calculate_crps(), plotCRPS.calculate_rmse()
            data = xr.merge([crps, rmse])
            data = data.expand_dims({'seed':1,'N_innovations':1})
            data = data.assign_coords(seed=('seed',[seed]),
                                      N_innovations=('N_innovations',[No]))
            return data
        
        N_inno = 2**np.arange(0,5) * Nens
        for No in N_inno:
            xp = self.factory.build(Nens, 'ETKF_D', No=No, name=f'ETKF_D')
            data = run_xp(xp, No)
            
            if self.data is None:
                self.data = data
            else:
                self.data = self.data.merge(data)
            
        xp = self.factory.build(Nens, 'Sqrt svd', name='ETKF', rot=False )   
        data = run_xp(xp, N_inno[0])
        for No in N_inno:
            data['N_innovations'] = [No] 
            self.data = self.data.merge(data)
             
        xp = self.factory.build(Nens, 'Sqrt svd', name='rotated ETKF', rot=True)   
        data = run_xp(xp, N_inno[0])
        for No in N_inno:
            data['N_innovations'] = [No] 
            self.data = self.data.merge(data)  
        
            
exp = CalibrateNo(N=100)
exp.run(dko=1)
plot = plots.ConfidencePlots(FIG_DIR)
plot.set_axes_labels('N_innovations','experiment','seed')
plot.plot_rms(exp.data['rmse'].sel({'variable':'position'}))
plot.save()

#%% Generate climatology

class ClimaExperiment(VaeExperiment):
    
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
                                verbose=False, shuffle=True)
            self.save(self.filepath)
        
        self.seed += 100
        return self
                
    @property 
    def filepath(self):
        A = int(self.amplitude*100)
        seed = int(self.seed)
        return os.path.join(MODEL_PATH, f'clima_{A:02d}_{seed:04d}.pkl')
                
    def create_model(self, seed):
        """ Run the climatology and train VAE."""

        # Generate climatology
        self.HMM, self.xx, _ = run_model(10000, 1, seed, amplitude=self.amplitude)

        # Create model
        self.hypermodel = vae.DenseVae()
        self.hp = self.hypermodel.build_hp(no_layers=6, no_nodes=32, use_rotation=False,
                                           batch_size=32, latent_dim=1, mc_samples=1)
        
        self.seed = seed
        reset_random_seeds(seed)
        self.model = self.hypermodel.build(self.hp)

    def save(self, filepath):
        with open(filepath, 'wb') as stream:
            dill.dump(self.model.get_weights(), stream)
        
    def load(self, filepath):
        with open(filepath,'rb') as stream:
            weights = dill.load(stream)
        self.model.set_weights(weights)

    def plot_clima(self):
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
        plotReconstruction.plot()
        plotReconstruction.save()
        
climas = ClimaExperiment(0.0)
        
#%%

class StaticExperiment(VaeExperiment):
    
    def __init__(self, N, dko):
        self.dko = dko 
        self.Nclima = max(1,int(np.sqrt(N)))
        self.N = int(N / self.Nclima)
        self.climas = iter(ClimaExperiment(0.0))
        self.No = Nens * 4
        self.factory = eda.EndaFactory() 
        
        self.crps = xr.Dataset()
        self.histograms = xr.Dataset()
        self.rmse = xr.Dataset()
        self.fails = []
        self.run_time = 100
        
        
    def create_xps(self, HMM, clima):
        hp, hypermodel, model = clima.hp, clima.hypermodel, clima.model
        
        #Create different 
        xps = []
        #No DA
        xps.append(eda.EnDa(Nens, [], name='no DA'))
        #Classic ETKF
        xps.append(self.factory.build(Nens, 'Sqrt svd', name='ETKF', rot=False))  
        
        #Use transfer from climatology without 2nd autoencoder
        bkg_trans = eda.BackgroundVaeTransform(hypermodel, hp, model)
        xps.append(self.factory.build(Nens, 'ETKF_D', No=self.No, name='single-transfer',
                                  VaeTransforms=[bkg_trans]))
        #Use transfer from climatology with 2nd autoencoder
        bkg_trans = eda.BackgroundVaeTransform(hypermodel, hp, model)
        inno_trans = eda.InnoVaeTransform(hypermodel, hp, model, self.No, 
                                          HMM.Obs(0).noise.add_sample)
        xps.append(self.factory.build(Nens, 'ETKF_D', No=self.No, name='double-transfer',
                                  VaeTransforms=[inno_trans, bkg_trans]))
        ##Train 1st VAE from previous. 
        #cycle_trans = eda.CyclingVaeTransform(hypermodel, hp, None)
        #xps.append(self.factory.build(Nens, 'ETKF_D', No=self.No, name='double-cycle',
        #                              VaeTransforms=[cycle_trans]))
        #Use clima for 1st autoencoder and a 2nd for observations. 
        vae_trans = eda.VaeTransform(hypermodel, hp, model)
        inno_trans = eda.InnoVaeTransform(hypermodel, hp, model, self.No, 
                                          HMM.Obs(0).noise.add_sample)
        xps.append(self.factory.build(Nens, 'ETKF_D', No=self.No, name='double-clima',
                                  VaeTransforms=[inno_trans, vae_trans]))
        return xps
    
    def run(self):
        #Run all models repeatedly
        climas = (next(self.climas) for _ in range(0,self.Nclima))
        for clima in climas:
            for n in range(0,self.N):
                
                self.seed = clima.seed + n * 7
                print(f'Running seed {self.seed}')
                self.run1(clima)
                
            #Save output 
            self.save()
    
    def run1(self, clima):
        #Create new run. 
        reset_random_seeds(self.seed-100)
        HMM, xx, yy = run_model(self.run_time, self.dko, self.seed-100)
        
        #Create experiments
        self.xx, self.yy = xx, yy
        self.xps = self.create_xps(HMM, clima)
        
        for xp in self.xps:
            #Run the DA experiment
            reset_random_seeds(self.seed)
            #Needs to be here to prevent running with same see. 
            _, _, _ = run_model(1000, self.dko, self.seed)
            xp.HMM = HMM
            try:
                xp.assimilate(HMM, xx, yy, liveplots=False)
            except:
                self.fails.append((clima,xp,self.seed))
                continue
            
            #Calculate CRPS and save in Xarray. 
            crps = plots.calculate_stat(plots.CRPS, xp, xx, seed=self.seed)
            histograms = plots.calculate_stat(plots.Histogram, xp, xx, seed=self.seed)
            rmse = plots.calculate_stat(plots.EnsError, xp, xx, seed=self.seed)
            self.crps = xr.merge([self.crps, crps])
            self.rmse = xr.merge([self.rmse, rmse])
            self.histograms = xr.merge([self.histograms, histograms])
            
    def add_stats(self, xp, HMM, xx, yy):
        plotCRPS = plots.EnsStatsPlots(FIG_DIR)
        plotCRPS.add_truth(HMM, xx)
        plotCRPS.add_xp(xp)
            
        crps, rmse = plotCRPS.calculate_crps(), plotCRPS.calculate_rmse()
        data = xr.merge([crps, rmse])
        data = data.expand_dims({'seed':1})
        data = data.assign_coords(seed=('seed',[self.seed]))
        
        return data
    
    def add_histogram(self, xp, HMM, xx, yy):
        plotHist = plots.ProbDensityPlots(FIG_DIR)
        plotHist.add_truth(HMM, xx)
        plotHist.add_xp(xp)
        
        data = plotHist.calculate_histogram(xp)
        data = data.expand_dims({'seed':1})
        data = data.assign_coords(seed=('seed',[self.seed]))
        return data
    
    @property
    def filepath(self):
        return os.path.join(MODEL_PATH,'static.pkl')
    
    def save(self):
        with open(self.filepath,'wb') as stream:
            dill.dump((self.crps, self.histograms, self.rmse), stream)
            
    def load(self):
        with open(self.filepath, 'rb') as stream:
            self.data, self.histograms, self.rmse = dill.load(stream)
            
    def delete(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            
#Run the experiment.         
exp = StaticExperiment(49, 5)
exp.run()

#%% Plot the conditional pro

#Plot prob density for different variables. 
from dapper.mods.ComplexCircle import vae_plots as plots
plotHist = plots.ProbDensityPlots(FIG_DIR, exp.histograms)
#plotHist.plot_scatter_density(exp.histograms)
crps=plots.CrpsPlots(FIG_DIR, exp.crps)
#crps.plot()
#crps.save()
rmse = plots.TaylorPlots(FIG_DIR, exp.rmse)
rmse.plot_taylor()

# %%

def run_exp01():
    HMM, xx, yy = run_model(1000, 5, 2000)
    _, _, _ = run_model(1000, 5, 3000)

    xps = []
    xps.append(eda.EnDa(Nens, [], name='no DA'))
    xps.append(factory.build(Nens, 'Sqrt svd', name='ETKF', rot=False))
    #
    vae_trans = eda.VaeTransform(hypermodel, hp, model)
    xps.append(factory.build(Nens, 'ETKF_D', No=32*Nens, name='single-clima',
                             VaeTransforms=[vae_trans]))
    inno_trans = eda.InnoVaeTransform(
        hypermodel, hp, model, 4096, HMM0.Obs(0).noise.add_sample)
    xps.append(factory.build(Nens, 'ETKF_D', No=32*Nens, name='double-clima',
                             VaeTransforms=[inno_trans, vae_trans]))
    xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='single-cycle',
                             VaeTransforms=[cycle_trans]))
    xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='double-cycle',
                             VaeTransforms=[cycle_trans]))
    xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='single-trans',
                             VaeTransforms=[inno_trans, bkg_trans]))


run_exp01()
