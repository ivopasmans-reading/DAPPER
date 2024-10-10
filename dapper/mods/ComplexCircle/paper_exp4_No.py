#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 4: non-linear observation operator. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR, collect_files
from vae_plots import plot_exp, calculate_output, MoviePlots, TimePlots, SeriesPlots, set_styles 
import os, sys, re, dill
import numpy as np
import xarray as xr

def obs_func(e):
    return np.exp(-e[0]-e[1]**2)

#Create the settings for the experiment
exp_name = "paper_exp4"
xp_parameters = {'names':['no DA','ETKF','double-clima','double-transfer']}
clima_parameters = {}

exps = []
for No in np.array([2,4,6,8,12,16])*64:
    exp = DaExperiment(exp_name+'_No{:04d}'.format(int(No)), Nruns=2, Nclima=1,
                       da_model=DapperModel(obs_func=obs_func),
                       xp_parameters={**xp_parameters, 'No':int(No)}, 
                       clima_parameters=clima_parameters)
    exps.append(exp)
    
def create_No_plot(exps):
    datas = []
    plot = SeriesPlots('No',FIG_DIR)
    pattern = re.compile(".*_No([0-9]+).pkl")
    for exp in exps:
        with open(os.path.join(exp.filepath),'rb') as stream:
            data_for, data_ana = dill.load(stream)
            datas.append(data_for)
            
        No = int(re.match(pattern, exp.filepath)[1])
        plot.add_exp(data_for, No)
        
    plot = set_styles(plot)
    plot.plot()
    
    return datas

def create_plots(exp):
    """ Plot the experiment. """
    
    exp.load()
    fig_dir = os.path.join(FIG_DIR, exp.save_name)
    
    #Create plots of statistics 
    plot_exp(exp, fig_dir)
    #Create movie
    output = xr.open_dataset(exp.filepath_output)
    plotter=MoviePlots(FIG_DIR ,output, experiments=['ETKF','double-clima','double-transfer'], 
                       obs_func=exp.da_model.obs_func)
    plotter.plot_movie_frames('movie_'+exp.save_name)
    #Create time series. 
    output_timeseries = output.sel(seed=1000, 
                                   experiment=['ETKF','double-clima','double-transfer'])
    plotter = TimePlots(fig_dir, output_timeseries)
    plotter.plot_prob_series('prob_series')
    plotter.save()
    
    plotter.close()
    
#%% 

if __name__=='__main__':
    #Run experiment
    iexp = int(sys.argv[2])
    if iexp<len(exps):
        run_exp(exps[iexp])
