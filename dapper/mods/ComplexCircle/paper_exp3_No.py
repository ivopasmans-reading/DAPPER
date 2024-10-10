#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 3: beta distribution. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR, collect_files
from vae_plots import plot_exp, calculate_output, MoviePlots, TimePlots
import os
import xarray as xr

#Create the settings for the experiment
exp_name = "paper_exp3_beta_No"
xp_parameters = {'No':8*64, 'names':['double-clima','double-transfer','single-clima','single-transfer']}
clima_parameters = {}

exp = DaExperiment(exp_name, Nruns=7, Nclima=7,
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
    plotter=MoviePlots(FIG_DIR ,output.sel(seed=1000), 
                       experiments=xp_parameters['names'], 
                       obs_func=exp.da_model.obs_func)
    plotter.plot_movie_frames('movie_'+exp.save_name)
    #Create time series. 
    output_timeseries = output.sel(seed=1000, 
                                   experiment=xp_parameters['names'])
    plotter = TimePlots(fig_dir, output_timeseries)
    plotter.plot_prob_series('prob_series')
    plotter.save()
    
    plotter.close()


#%%

if __name__=='__main__':
    run_exp(exp) 
    
       
        
