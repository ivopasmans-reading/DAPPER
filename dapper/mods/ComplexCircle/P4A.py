#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 4: non-linear observation operator. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR
from vae_plots import plot_exp, MoviePlots, TimePlots, SeriesPlots, set_styles 
import os, sys, re, dill
import numpy as np
import xarray as xr

def obs_factory(alpha):
    """Nonlinear function."""
    eps = 1.e-2
    
    def obs_func(e):
        x, y = e
        return x * (1+np.abs(y))**(2*alpha)
        
    def obs_Dfunc(e):
        x, y = e 
        dedx = (1+np.abs(y))**(2*alpha)
        dedy = x*0. if np.isclose(alpha,0.) else 2.*alpha*x*(1+np.abs(y))*(2*alpha-1.)*np.sign(y)
        return np.array([dedx, dedy])
    
    return obs_func, obs_Dfunc

#Create the settings for the experiment
exp_name = "P4A"
clima_parameters = {}
exps = []

xp_parameters = {'names':['ETKF linear']}
for nl_parameter in np.linspace(0,1,6):
    obs_func, obs_Dfunc = obs_factory(nl_parameter) 
    exp = DaExperiment(exp_name+'_lin{:03d}'.format(int(nl_parameter*100)), 
                       Nruns=8, Nclima=8,
                       da_model=DapperModel(obs_func=obs_func, obs_Dfunc=obs_Dfunc),
                       xp_parameters=xp_parameters,
                       clima_parameters=clima_parameters)
    exp.nl_parameter = nl_parameter
    exps.append(exp)
    
xp_parameters = {'names':['no DA','ETKF','single-clima','double-clima']}
for nl_parameter in np.linspace(0,1,6):
    obs_func, obs_Dfunc = obs_factory(nl_parameter) 
    exp = DaExperiment(exp_name+'_nl{:03d}'.format(int(nl_parameter*100)), 
                       Nruns=8, Nclima=8,
                       da_model=DapperModel(obs_func=obs_func),
                       xp_parameters=xp_parameters,
                       clima_parameters=clima_parameters)
    exp.nl_parameter = nl_parameter
    exps.append(exp)
    
xp_parameters = {}
for nl_parameter in [1.]:
    obs_func, obs_Dfunc = obs_factory(nl_parameter) 
    exp = DaExperiment(exp_name,
                       Nruns=8, Nclima=8,
                       da_model=DapperModel(obs_func=obs_func),
                       xp_parameters=xp_parameters,
                       clima_parameters=clima_parameters)
    exp.nl_parameter = nl_parameter
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
                                   experiment=['ETKF','single-clima','double-clima'])
    plotter = TimePlots(fig_dir, output_timeseries)
    plotter.plot_prob_series('prob_series')
    plotter.save()
    
    plotter.close()
    
def create_nl_plot(exps):
    datas = []
    plot = SeriesPlots('nonlinear', FIG_DIR)
    pattern = re.compile(".*_([0-9]+).pkl") 
    for exp in exps:
        with open(os.path.join(exp.filepath),'rb') as stream:
            data_for, data_ana = dill.load(stream)
            datas.append(data_for)
            
        nl_parameter = exp.nl_parameter
        plot.add_exp(data_for, nl_parameter)
        
    plot = set_styles(plot)
    plot.plot()
    
    return datas

#create_nl_plot(exps[6:])
    
#%% 

if __name__=='__main__':
    #Run experiment
    iexp = int(sys.argv[2])
    if iexp<len(exps):
        run_exp(exps[iexp])
