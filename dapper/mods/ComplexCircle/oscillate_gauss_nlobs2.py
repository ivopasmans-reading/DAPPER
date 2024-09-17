#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:18:07 2024

Experiment for paper with truth oscillation around the unit circle and 
observational errors using beta distribution. 

@author: ivo
"""

import numpy as np
import xarray as xr
from climate import ClimaExperiment, VaeExperiment, XpsClass, filter_data, clear
from climate import Nens, reset_random_seeds, run_model_default
from dapper.vae import circle_vae as vae
from dapper.mods.ComplexCircle import vae_plots as plots
import os, dill, shutil, sys

# Directory in which the figures will be stored.
FIG_DIR = '/home/ivo/Figures/vae/paper_nlobs'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'

# Copy this file
if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR, 'experiment.py'))

climas = ClimaExperiment(0.0, do_plot=False)
obs_op = lambda e : np.exp(e[0]*(e[0]**2+e[1]**2)-1)
run_model = lambda K, dko, seed, **kwargs : run_model_default(K, dko, seed, 
                                                    amplitude=0.2, 
                                                    obs_func = obs_op,
                                                    **kwargs)

#%% Experiment oscillation with different obs error distribution

class ObsOscillateExperiment(VaeExperiment):
    """ Experiment in which truth runs over unit circle with varying radius. """

    def __init__(self, N, dko, save_name='nlobs_a.pkl'):
        self.dko = dko 
        self.Nclima = max(1,int(np.sqrt(N)))
        self.N = int(N / self.Nclima)
        self.No = Nens * 4 #IP
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
                if not self._in_seed_range(self.seed):
                    continue
                print(f'Running seed {self.seed}')
                self.run1_obs_type(clima, 'normal')
                
    def _in_seed_range(self, seed):
        if len(sys.argv)!=3:
            return True
        elif seed>=int(sys.argv[1]) and seed<int(sys.argv[2]):
            return True 
        else:
            return False

    def check_done(self, xp):
        rmse = self.data_ana['rmse']

        if len(rmse) == 0:
            return False

        rmse = rmse['x']
        try:
            x = rmse.sel({'experiment': xp.name, 'seed': self.seed})
        except:
            return False

        if np.any(np.isnan(x.data)):
            return False
        else:
            return True

    def run1_obs_type(self, clima, obs_type):
        # Create new run.
        reset_random_seeds(self.seed-100)
        HMM, xx, yy = run_model(self.run_time, self.dko, self.seed-100,
                                obs_type=obs_type)

        # Create experiments
        names = ['ETKF','single-transfer','double-transfer']
        #names = [(name+' beta',name+' normal') for name in names]
        names = np.array(names).ravel()
        
        self.xx, self.yy = xx, yy
        xps_iterator = XpsClass(clima, HMM, Nens, self.No,
                                 names=names) 
        self.xps = iter(xps_iterator)

        for xp in self.xps:
            #Test if experiment has been loaded from file. 
            if (xp.name, self.seed) in self.done:
                print('\nDONE ', xp.name, self.seed)
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
            for key, value in self.keys.items():
                kwargs = {'xp': xp, 'xx': xx,
                          'seed': self.seed, 'stage': 'forecast'}
                stat = plots.calculate_stat(value, **kwargs)
                self.data_for[key] = xr.merge([self.data_for[key], stat])

                kwargs = {'xp': xp, 'xx': xx,
                          'seed': self.seed, 'stage': 'analysis'}
                stat = plots.calculate_stat(value, **kwargs)
                self.data_ana[key] = xr.merge([self.data_ana[key], stat])

            self.save()

# Run the experiment.
exp = ObsOscillateExperiment(49, 10)
exp.load()
exp.run()

#Terminate if not called from command line
if len(sys.argv)>1:
    quit()

# %% Plot output statistics.

for stage, data in zip(['forecast', 'analysis'], [exp.data_for, exp.data_ana]):
    
    plot_data = filter_data(data['crps'])
    plotHist = plots.SingleCrpsPlots(FIG_DIR, plot_data)
    plotHist.style = plots.CompoundedStyles()
    plotHist = plots.set_styles(plotHist)
    plotHist.plot_crps('crps_single_'+stage)
    plotHist.save()
    
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

    plot_data = data['rmse']
    plotHist = plots.TaylorPlots(FIG_DIR, plot_data)
    plotHist.plot_taylor('taylor_'+stage)
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
        if xp.name not in experiments:
            continue
        print('XP ', xp.name)
        
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

plot_movie(['ETKF','single-transfer','double-transfer'], 500, 10)
