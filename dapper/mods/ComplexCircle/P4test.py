#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 4: non-linear observation operator. 

@author: ivo
"""

import xarray as xr
import os, sys, re, dill
import numpy as np
from climate import DapperModel, DaExperiment, run_exp, FIG_DIR
from vae_plots import plot_exp, MoviePlots, TimePlots, SeriesPlots, set_styles 

def obs_factory(alpha):
    """Nonlinear function."""
    eps = 1.e-2
    
    def obs_func(e):
        x, y = e
        r = np.hypot(x,y)
        if (y<0. and np.abs(x)<1e-4) or np.isclose(r+y,0.):
            return np.ones_like(x)*10.
        elif np.isclose(r+y,0.):
            print('RY',r+y,x,y,r)
            raise Exception('DIVIDE')
        else:
            q = np.sign(x) * min(10., np.abs(x) * (r + y)**(-alpha))
            return q
        
    def obs_Dfunc(e):
        x, y = e 
        r = np.hypot(x,y)
        if y<0. and np.abs(x)<1e-4:
            return np.zeros_like(x)
        
        drdx, drdy = x/r, y/r 
        dedx = (r+y)**(-alpha) - alpha * x * (r+y)**(-alpha-1) * drdx 
        dedy = -alpha * x * (r+y)**(-alpha-1) * (drdy + 1)
        return np.array([dedx, dedy])
    
    return obs_func, obs_Dfunc

#Create the settings for the experiment
exp_name = "P4T"
clima_parameters = {}
exps = []
    
xp_parameters = {'names':['ETKF','double-transfer']}
for nl_parameter in [1.]:
    obs_func, obs_Dfunc = obs_factory(nl_parameter) 
    exp = DaExperiment(exp_name,run_time=20,
                       Nruns=2, Nclima=1,
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

#datas=create_nl_plot(exps)
#import netCDF4 as nc 
#output = nc.Dataset('/home/ivo/dpr_data/vae/circle/P4_nl100/1000_1000_output.nc')
   
#%% 

run_exp(exps[0])


nc = xr.open_dataset('/home/ivo/dpr_data/vae/circle/P4T_output.nc')

#%% Plot 

import matplotlib.pyplot as plt
plt.close('all')
options = {'seed':1000,'time':20}
plt.figure()
for exp in np.array(nc['experiment']):
    data = nc['ensemble'].sel(**{'experiment':exp,'stage':'forecast',**options})
    Y = [obs_func(e) for e in data.data]
    plt.plot(Y, label=exp+' for', linestyle='-')
    data = nc['ensemble'].sel(**{'experiment':exp,'stage':'analysis',**options})
    Y = [obs_func(e) for e in data.data]
    plt.plot(Y, label=exp+' ana', linestyle='--')
xx = nc['truth'].sel(**{'experiment':'ETKF',**options})
plt.plot([0,63],obs_func(xx.data)*np.array([1,1]),'k--',label='truth')
plt.legend(loc='lower right')

plt.figure()
for exp in np.array(nc['experiment']):
    if 'latent_ensemble' not in nc:
        continue
    data = nc['latent_ensemble'].sel(**{'experiment':exp,'stage':'forecast',**options})
    plt.plot(data.data, label=exp+' for', linestyle='-')
    data = nc['latent_ensemble'].sel(**{'experiment':exp,'stage':'analysis',**options})
    plt.plot(data.data, label=exp+' ana', linestyle='--')
    xx = nc['latent_truth'].sel(**{'experiment':exp,**options})
    plt.plot([0,63],xx.data*np.array([1,1]),'k--',label='truth')
plt.legend(loc='lower right')

# if __name__=='__main__':
#     #Run experiment
#     iexp = int(sys.argv[2])
#     if iexp<len(exps):
#         run_exp(exps[iexp])


