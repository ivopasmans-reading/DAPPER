#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 3: skewed distribution distribution. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR, collect_files
from vae_plots import plot_exp, calculate_output, MoviePlots, TimePlots, SeriesPlots, set_styles
import matplotlib.pyplot as plt
import dill
import os, sys, re
import xarray as xr
import numpy as np

#Create the settings for the experiment
exp_name = "paper_exp3_test"
xp_parameters = {'names':['no DA','ETKF','single-clima','double-clima'],
                 'No':64*8}
clima_parameters = {}

exps = []
for skew in [-12,-8,-4,-2,-1,0,1,2,4,8,12]:
    exp = DaExperiment(exp_name+"_{:+04d}".format(int(10*skew)), Nruns=8, Nclima=8,
                       da_model=DapperModel(obs_type=('skewedNormal',skew)),
                       xp_parameters=xp_parameters, 
                       clima_parameters=clima_parameters)
    exp.skew = skew
    exps.append(exp)

def create_plots(exp):
    """ Plot the experiment. """
    exp.load()
    fig_dir = os.path.join(FIG_DIR, exp.save_name)
    
    #Create plots of statistics 
    plot_exp(exp, fig_dir)
    #Create movie
    output = xr.open_dataset(exp.filepath_output)
    plotter=MoviePlots(FIG_DIR ,output, experiments=['ETKF','single-clima','double-clima'], 
                       obs_func=exp.da_model.obs_func)
    plotter.plot_movie_frames('movie_'+exp.save_name)
    #Create time series. 
    output_timeseries = output.sel(seed=1000, 
                                   experiment=['ETKF','single-clima','double-clima',
                                               'single-transfer','double-transfer'])
    plotter = TimePlots(fig_dir, output_timeseries)
    plotter.plot_prob_series('prob_series')
    plotter.save()
    
    plotter.close()
        
            

def create_skew_plot(exps):
    plot = SeriesPlots('skew',FIG_DIR)
    pattern = re.compile(".*_([+-][0-9]+).pkl")
    for exp in exps:
        with open(os.path.join(exp.filepath),'rb') as stream:
            data_for, data_ana = dill.load(stream)
            
        skew = int(re.match(pattern, exp.filepath)[1])/10.
        print('SKEW',skew,exp.filepath)
        plot.add_exp(data_for, skew)
        
    plot = set_styles(plot)
    plot.plot()
    
    

# create_skew_plot(exps)
# for exp in exps:
#     if exp.skew == 0:
#         create_plots(exp)
        

#%%

if __name__=='__main__':
    iexp = int(sys.argv[2])
    if iexp<len(exps):
        run_exp(exps[iexp])
    
       
        
