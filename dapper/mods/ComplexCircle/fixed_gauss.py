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
FIG_DIR = '/home/ivo/Figures/vae/exp01'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'
# Number of ensemble member
Nens = 64

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
            
        self.plot_clima()
        
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
        
climas = ClimaExperiment(0.0)
        
#%% Experiment static

from dapper.mods.ComplexCircle import vae_plots as plots

class XpsClass:
    
    def __init__(self, clima, HMM, Nens, No):
        self.names = ['no DA', 'ETKF', 'single-transfer', 'double-transfer',
                      'double-clima']
        self.names = ['no DA','ETKF','single-transfer','single-clima']
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
                                    VaeTransforms=[inno_trans, bkg_trans])
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
                                    VaeTransforms=[inno_trans, vae_trans])
        else:
            raise ValueError(f'{name} not a valid name for experiment.')
        
        return xp 

class StaticExperiment(VaeExperiment):
    """ Experiment in which truth runs over unit circle. """
    
    def __init__(self, N, dko):
        self.dko = dko 
        self.Nclima = max(1,int(np.sqrt(N)))
        self.N = int(N / self.Nclima)
        self.No = Nens * 4 #IP
        self.run_time = 500
        
        self.climas = iter(ClimaExperiment(0.0))
            
        self.fails = []
        self.keys = dict([('crps',plots.CRPS), ('histogram',plots.Histogram),
                          ('rmse',plots.EnsError)])
        self.data_for = dict([(key,xr.Dataset()) for key in self.keys])
        self.data_ana = dict([(key,xr.Dataset()) for key in self.keys])
        
        
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
            del(clima)
            
    def check_done(self, xp):
        rmse = self.data_ana['rmse']
        
        if len(rmse)==0:
            return False

        rmse = rmse['x']
        try:
            x = rmse.sel({'experiment':xp.name,'seed':self.seed})
        except:
            return False
        
        if np.any(np.isnan(x.data)):
            return False
        else:
            return True
        
    
    def run1(self, clima):
        #Create new run. 
        reset_random_seeds(self.seed-100)
        HMM, xx, yy = run_model(self.run_time, self.dko, self.seed-100)
        
        #Create experiments
        self.xx, self.yy = xx, yy
        self.xps = iter(XpsClass(clima, HMM, Nens, self.No))
        
        for xp in self.xps:
            has_done = self.check_done(xp)
            if has_done:
                print('DONE ',xp.name, self.seed)
                del(xp)
                continue
            
            #Run the DA experiment
            reset_random_seeds(self.seed)
            #Needs to be here to prevent running with same see. 
            _, _, _ = run_model(self.run_time, self.dko, self.seed)
            xp.HMM = HMM
            try:
                xp.assimilate(HMM, xx, yy, liveplots=False)
            except:
                self.fails.append((clima,xp,self.seed))
                continue
            
            #Calculate CRPS and save in Xarray. 
            for key,value in self.keys.items():
                kwargs = {'xp':xp,'xx':xx,'seed':self.seed,'stage':'forecast'}
                stat = plots.calculate_stat(value, **kwargs)
                self.data_for[key] = xr.merge([self.data_for[key], stat])
                
                kwargs = {'xp':xp,'xx':xx,'seed':self.seed,'stage':'analysis'}
                stat = plots.calculate_stat(value, **kwargs)
                self.data_ana[key] = xr.merge([self.data_ana[key], stat])
                
            del(xp)
            
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
            dill.dump((self.data_for, self.data_ana), stream)
            
    def load(self):
        with open(self.filepath, 'rb') as stream:
            self.data_for, self.data_ana = dill.load(stream)
            
    def delete(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            
#Run the experiment.         
exp = StaticExperiment(49, 10)
exp.load()
exp.run()

#%% Plot output statistics. 

for stage, data in zip(['forecast','analysis'],[exp.data_for, exp.data_ana]):
    plot_data = data['histogram']
    plotHist = plots.ProbDensityPlots(FIG_DIR, plot_data)
    plotHist.plot_scatter_density('scatter_'+stage)
    plotHist.save()
    
    plot_data = data['crps']
    plotHist = plots.CrpsPlots(FIG_DIR, plot_data)
    plotHist.plot_crps('crps_'+stage)
    plotHist.save()
    
    plot_data = data['rmse']
    plotHist = plots.TaylorPlots(FIG_DIR, plot_data)
    plotHist.plot_taylor('taylor_'+stage)
    plotHist.save()
