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
from climate import ClimaExperiment, VaeExperiment, XpsClass, filter_data
from climate import Nens, reset_random_seeds, run_model_default
from dapper.vae import circle_vae as vae
from dapper.mods.ComplexCircle import vae_plots as plots
import os, dill, shutil, sys

# Directory in which the figures will be stored.
FIG_DIR = '/home/ivo/Figures/vae/paper_static'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle'

# Copy this file
if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR, 'experiment.py'))
    
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

climas = ClimaExperiment(0.0, do_plot=False)
run_model = run_model_default
        
#%% Experiment static

class StaticExperiment(VaeExperiment):
    """ Experiment in which truth runs over unit circle. """
    
    def __init__(self, N, dko, save_name='static.pkl'):
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
        #Run all models repeatedly
        climas = (next(self.climas) for _ in range(0,self.Nclima))
        for clima in climas:
            for n in range(0,self.N):
                self.seed = clima.seed + n * 7
                if not self._in_seed_range(self.seed):
                    continue
                print(f'Running seed {self.seed}')
                self.run1(clima)
                
            #Save output 
            del(clima)
        
    
    def run1(self, clima):
        #Create new run. 
        reset_random_seeds(self.seed-100)
        HMM, xx, yy = run_model(self.run_time, self.dko, self.seed-100)
        
        #Create experiments
        self.xx, self.yy = xx, yy
        self.xps = iter(XpsClass(clima, HMM, Nens, self.No))
        
        for xp in self.xps:
            #Test if experiment has been loaded from file. 
            if (xp.name, self.seed) in self.done:
                print('\nDONE ', xp.name, self.seed)
                continue
            else:
                print('\nRUNNING ', xp.name, self.seed)
            
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
                
            #Save output
            self.save()
            self.done += [(xp.name, self.seed)]
            
#Run the experiment.         
exp = StaticExperiment(49, 10)
exp.load()
exp.run()

#Terminate if not called from command line
if len(sys.argv)>1:
    quit()


#%% Plot output statistics.

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

#%% Generate animation 

def plot_movie(experiments, run_time, dko, No=Nens*16):
    climas = iter(ClimaExperiment(0.0))
    for n in range(1):
        clima  = climas.__next__()
    
    #Create new run. 
    reset_random_seeds(clima.seed-100)
    HMM, xx, yy = run_model(run_time, dko, clima.seed-100)
    xps = iter(XpsClass(clima, HMM, Nens, No))
    
    for xp in xps:
        print('XP ',xp.name)
        if xp.name not in experiments:
            continue 
        #Run
        _, _, _ = run_model(run_time, dko, clima.seed)
        xp.HMM = HMM
        xp.assimilate(HMM, xx, yy, liveplots=False)
        
        
        #Plot 
        circle = plots.CirclePlot(FIG_DIR)
        circle.add_track(xp.name, xp.HMM.tseq.tt, xx)
        circle.add_obs(xp.HMM.tseq.tto, yy)
        circle.add_ens_for(xp.name, xp.HMM.tseq.tto, xp.stats.E.f) 
        circle.add_ens_ana(xp.name, xp.HMM.tseq.tto, xp.stats.E.a)
        #circle.add_latent_for(xp.name, xp.HMM.tseq.tto, xp.stats.Elatent['f'])
        #circle.add_latent_ana(xp.name, xp.HMM.tseq.tto, xp.stats.Elatent['a'])
        
       
        circle.animate_time(xp.HMM.tseq.tto, fig_name='movie_'+xp.name)
    
plot_movie(['ETKF','single-clima','single-transfer'], 500, 10)
#plot_movie(['double-transfer'], 100, 10)