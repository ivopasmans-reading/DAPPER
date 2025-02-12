#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 1 in the paper. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR
from vae_plots import plot_exp, MoviePlots, TimePlots, SeriesPlots, set_styles
import os, sys, re, dill
import xarray as xr

#Create the settings for the experiment
exp_name = "P2"
xp_parameters = {'names':['no DA','ETKF','single-clima','single-transfer']}
clima_parameters = {}

exps = []
for amplitude in [0,.1,.2,.3,.4,.6]:
    exp = DaExperiment(exp_name+"_{:02d}".format(int(10*amplitude)), Nruns=8, Nclima=8,
                       da_model=DapperModel(amplitude=amplitude),
                       xp_parameters=xp_parameters, 
                       clima_parameters=clima_parameters)
    exp.amplitude = amplitude
    exps.append(exp)
    
for amplitude in [.2]:
    exp = DaExperiment(exp_name, Nruns=8, Nclima=8,
                       da_model=DapperModel(amplitude=amplitude),
                       xp_parameters={}, 
                       clima_parameters=clima_parameters)
    exp.amplitude = amplitude
    exps.append(exp)

def create_plots(exp):
    """ Plot the experiment. """
    exp.load()
    fig_dir = os.path.join(FIG_DIR, exp.save_name)
    
    #Create plots of statistics 
    plot_exp(exp, fig_dir)
    #Create movie
    output = xr.open_dataset(exp.filepath_output)
    plotter=MoviePlots(FIG_DIR ,output.sel(seed=1000), 
                       experiments=['ETKF','single-clima','single-transfer'], 
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
    return output


def create_amplitude_plot(exps):
    datas = []
    plot = SeriesPlots('amplitude',FIG_DIR)
    pattern = re.compile(".*_([0-9]+).pkl") 
    for exp in exps:
        with open(os.path.join(exp.filepath),'rb') as stream:
            data_for, data_ana = dill.load(stream)
            datas.append(data_for)
            
        plot.add_exp(data_for, exp.amplitude)
        
    plot = set_styles(plot)
    plot.plot()
    
    return datas

datas=create_amplitude_plot(exps[:-1])
create_plots(exps[-1])

#%% Uncomment when using runner2.sh
  
# if __name__=='__main__':
#     #Run experiment
#     iexp = int(sys.argv[2])
#     if iexp<len(exps):
#         run_exp(exps[iexp])
    

    
       
        
