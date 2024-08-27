#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:18:07 2024

Experiment for paper with truth moving along unit circle and observational 
error from Gaussian. 

@author: ivo
"""

import numpy as np
import xarray as xr
from climate import ClimaExperiment, VaeExperiment, XpsClass
from climate import Nens, reset_random_seeds, run_model_default
from dapper.vae import circle_vae as vae
from dapper.mods.ComplexCircle import vae_plots as plots
import os, dill, shutil

climas = ClimaExperiment(0.0)
# Directory in which the figures will be stored.
FIG_DIR = '/home/ivo/Figures/vae/paper_oscillation'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'

# Copy this file
if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR, 'experiment.py'))
    
climas = ClimaExperiment(0.0, do_plot=False)
run_model = lambda K, dko, seed : run_model_default(K, dko, seed, 
                                                    amplitude=0.2)
    
# %% Experiment oscillation

class OscillateExperiment(VaeExperiment):
    """ Experiment in which truth runs over unit circle with varying radius. """

    def __init__(self, N, dko, save_name='oscillation3a.pkl'):
        self.dko = dko
        self.Nclima = max(1, int(np.sqrt(N)))
        self.N = int(N / self.Nclima)
        self.No = Nens * 4  # IP
        self.run_time = 500
        self.save_name = save_name

        self.climas = iter(ClimaExperiment(0.0))

        self.fails = []
        self.keys = dict([('crps', plots.CRPS), ('histogram', plots.Histogram),
                          ('rmse', plots.EnsError)])
        self.data_for = dict([(key, xr.Dataset()) for key in self.keys])
        self.data_ana = dict([(key, xr.Dataset()) for key in self.keys])

    def run(self):
        # Run all models repeatedly
        climas = (next(self.climas) for _ in range(0, self.Nclima))
        for clima in climas: 
            for n in range(0, self.N):
                self.seed = clima.seed + n * 7
                print(f'Running seed {self.seed}')
                self.run1(clima)

            # Save output
            self.save()
            del(clima)

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

    def run1(self, clima):
        # Create new run.
        reset_random_seeds(self.seed-100)
        HMM, xx, yy = run_model(self.run_time, self.dko, self.seed-100)

        # Create experiments
        self.xx, self.yy = xx, yy
        self.xps = iter(XpsClass(clima, HMM, Nens, self.No))

        for xp in self.xps:
            has_done = self.check_done(xp)
            if has_done:
                print('DONE ', xp.name, self.seed)
                del(xp)
                continue

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

            del(xp)

    @property
    def filepath(self):
        return os.path.join(MODEL_PATH, self.save_name)

    def save(self):
        with open(self.filepath, 'wb') as stream:
            dill.dump((self.data_for, self.data_ana), stream)

    def load(self):
        with open(self.filepath, 'rb') as stream:
            self.data_for, self.data_ana = dill.load(stream)

    def delete(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


# Run the experiment.
exp = OscillateExperiment(49, 10)
exp.load()
exp.run()

#%% Plot output statistics.

#Remove faulty 1200<=seed<1300
def filter_data(data):
    seeds = data.coords['seed']
    seeds = [s for s in seeds if s<1200 or s>=1300]
    return data.sel(seed=seeds) 

for stage, data in zip(['forecast','analysis'],[exp.data_for, exp.data_ana]):
    plot_data = filter_data(data['histogram'])
    plotHist = plots.ProbDensityPlots(FIG_DIR, plot_data)
    plotHist.plot_scatter_density('scatter_'+stage)
    plotHist.save()
    
    plot_data = filter_data(data['crps'])
    plotHist = plots.CrpsPlots(FIG_DIR, plot_data)
    plotHist.plot_crps('crps_'+stage)
    plotHist.save()
    
    plot_data = filter_data(data['rmse'])
    plotHist = plots.TaylorPlots(FIG_DIR, plot_data)
    plotHist = plots.set_styles(plotHist)
    plotHist.plot_taylor('taylor_'+stage)

    plotHist.save()
    
    plot_data = filter_data(data['crps'])
    plotHist = plots.SingleCrpsPlots(FIG_DIR, plot_data)
    plotHist = plots.set_styles(plotHist)
    plotHist.plot_crps('crps_single_'+stage)
    plotHist.save()
    
    plot_data = filter_data(data['crps'])
    plotHist = plots.SingleCrpsPlots(FIG_DIR, plot_data)
    plotHist = plots.set_styles(plotHist)
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
