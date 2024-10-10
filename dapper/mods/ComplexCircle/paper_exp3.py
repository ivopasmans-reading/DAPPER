#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 3: beta distribution. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR, collect_files
from vae_plots import plot_exp, calculate_output, MoviePlots, TimePlots, SkewPlots
import os
import xarray as xr

#Create the settings for the experiment
exp_name = "paper_exp3_beta"
xp_parameters = {}
clima_parameters = {}

exp = DaExperiment(exp_name, Nruns=7, Nclima=7,
                   da_model=DapperModel(obs_type='beta'),
                   xp_parameters=xp_parameters, 
                   clima_parameters=clima_parameters)

exp_xps = DaExperiment(exp_name+'_xps', Nruns=1, Nclima=1,
                   da_model=DapperModel(obs_type='beta'),
                   xp_parameters=xp_parameters, 
                   clima_parameters=clima_parameters)

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
    
def plot_skew():
    pl = SkewPlots(FIG_DIR, [0,-2,-4,-12])
    pl.plot()


#%%

if __name__=='__main__':
    run_exp(exp) 
    
       
        
