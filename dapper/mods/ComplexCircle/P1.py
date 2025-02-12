#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 1 in the paper. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, FIG_DIR
from vae_plots import plot_exp,  MoviePlots, TimePlots, SeriesPlots, set_styles
import os, sys, re, dill
import xarray as xr
import numpy as np

#Create the settings for the experiment
exp_name = "P1"
xp_parameters = {}
clima_parameters = {}

exps = []
for rotation_rate in [.01, .02, .05, .1, .15, .2]:
    exp = DaExperiment(exp_name+"_rate{:02d}".format(int(rotation_rate*100)), 
                       Nruns=8, Nclima=8,
                       da_model=DapperModel(rotation_rate=rotation_rate),
                       xp_parameters=xp_parameters, 
                       clima_parameters=clima_parameters)
    exp.rotation_rate = rotation_rate
    exps.append(exp)

def create_plots(exp):
    """ Plot the experiment. """
    exp.load()
    fig_dir = os.path.join(FIG_DIR, exp.save_name)
    
    #Create plots of statistics 
    plot_exp(exp, fig_dir)
    #Create movie
    output = xr.open_dataset(exp.filepath_output)
    plotter=MoviePlots(FIG_DIR ,output, experiments=['ETKF','single-clima','single-transfer'], 
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

def create_rate_plot(exps):
    datas = []
    plot = SeriesPlots('angular velocity', FIG_DIR)
    pattern = re.compile(".*_([0-9]+).pkl") 
    for exp in exps:
        with open(os.path.join(exp.filepath),'rb') as stream:
            data_for, data_ana = dill.load(stream)
            datas.append(data_for)
            
        plot.add_exp(data_for, exp.rotation_rate)
        
    plot = set_styles(plot)
    plot.plot()
    
    return datas

def plot_correlation_xy(exp):
    from matplotlib import pyplot as plt
    from matplotlib import colors as mcolors
    
    experiment='ETKF'
    nc = xr.open_dataset(exp.filepath_output)
    seeds = nc['seed'].data
    times = nc['time'].data
    Efor = nc['ensemble'].sel(stage='forecast',seed=seeds[0],experiment=experiment).data
    Eana = nc['ensemble'].sel(stage='analysis',seed=seeds[0],experiment=experiment).data
    Lfor = nc['latent_ensemble'].sel(stage='forecast',seed=seeds[0],experiment=experiment).data
    Lana = nc['latent_ensemble'].sel(stage='analysis',seed=seeds[0],experiment=experiment).data
    
    plt.close('all')
    fig = plt.figure(figsize=(10,6))
    axes = np.array(fig.subplots(1,2)).reshape((1,2))
    
    normalize = lambda x, axis=0 : (x - np.mean(x,axis=axis,keepdims=True)) / np.std(x,keepdims=True,axis=axis,ddof=1)
    
    handles = []
    for it, color in zip(np.arange(0,6), mcolors.TABLEAU_COLORS):
        x = normalize(Efor[it,:,0])
        y = normalize(Efor[it,:,1])
        z = normalize(Lfor[it,:,0])
        
        cyx = np.dot(x,y)/np.dot(x,x)
        ry = np.mean((y-cyx*x)**2)
        czx = np.dot(x,z)/np.dot(x,x)
        rz = np.mean((z-czx*x)**2)
        
        handle, = axes[0,0].plot(x,y,'o',color=color, 
                                 label="t={:2d} ({:.2f},{:.2f})".format(times[it],ry,rz))
        axes[0,1].plot(x,z,'o',color=color)
        handles.append(handle)
   
    for ax in axes.ravel():
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_aspect(1)
        ax.set_xlabel("x'")
    axes[0,0].set_ylabel("y'")
    axes[0,1].set_ylabel("z'")
    
    lax = fig.add_axes((.15,.8,.7,.1))
    lax.axis('off')
    lax.legend(handles=handles,ncols=3,loc='lower center')
    
    nc.close()
    
    filepath = os.path.join(FIG_DIR,'correlation_'+exp.save_name+'_'+experiment+'.png')
    fig.savefig(filepath,dpi=300)

#%% Uncomment when using runner2.sh
  
# if __name__=='__main__':
#     #Run experiment
#     iexp = int(sys.argv[2])
#     if iexp<len(exps):
#         run_exp(exps[iexp])
    

    
       
        
