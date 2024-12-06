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
exp_name = "P42d"
clima_parameters = {'hp_parameters':{'latent_dim':2}, 'filename':'clima2d'}
xp_parameters = {'names':['ETKF','single-clima','single-transfer']}
exps = []
    

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
    plotter=MoviePlots(FIG_DIR ,output, experiments=['ETKF','single-clima','single-transfer'], 
                       obs_func=exp.da_model.obs_func)
    plotter.plot_movie_frames('movie_'+exp.save_name)
    #Create time series. 
    output_timeseries = output.sel(seed=1000, 
                                   experiment=['ETKF','single-clima','single-transfer'])
    plotter = TimePlots(fig_dir, output_timeseries)
    plotter.plot_prob_series('prob_series')
    plotter.save()
    
    plotter.close()
    
def plot_latent(it,exp):
    from matplotlib import pyplot as plt
    experiment='single-transfer'
    nc = xr.open_dataset(exp.filepath_output)
    seeds = nc['seed'].data
    times = nc['time'].data
    Efor = nc['ensemble'].sel(stage='forecast',seed=seeds[0],experiment=experiment).data
    Eana = nc['ensemble'].sel(stage='analysis',seed=seeds[0],experiment=experiment).data
    Lfor = nc['latent_ensemble'].sel(stage='forecast',seed=seeds[0],experiment=experiment).data
    Lana = nc['latent_ensemble'].sel(stage='analysis',seed=seeds[0],experiment=experiment).data
    truth = nc['latent_truth'].sel(seed=seeds[0],experiment=experiment).data
    
    plt.close('all')
    fig = plt.figure(figsize=(6,6))
    ax = fig.subplots(1,1)
    ax.plot(Efor[it,:,0],Efor[it,:,1],'bx',alpha=.7)
    ax.plot(Eana[it,:,0],Eana[it,:,1],'gx',alpha=.7)
    ax.plot(Lfor[it,:,0],Lfor[it,:,1],'bo',alpha=.7)
    ax.plot(Lana[it,:,0],Lana[it,:,1],'go',alpha=.7)
    ax.plot(truth[it,0],truth[it,1],'ko')
    ax.set_xlabel(r'$z_{0}$')
    ax.set_ylabel(r'$z_{1}$')
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-2., 2.)
    ax.set_aspect(1)
   
    nc.close()
    
    
def create_latent_plot(exps):
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

#if __name__=='__main__':
#    #Run experiment
#    iexp = int(sys.argv[2])
#    if iexp<len(exps):
#        run_exp(exps[iexp])
        
#%% 

#run_exp(exps[0])
#nc = xr.open_dataset('/home/ivo/dpr_data/vae/circle/P4T_output.nc')

# import matplotlib.pyplot as plt
# plt.close('all')
# options = {'seed':1000,'time':20}
# plt.figure()
# for exp in np.array(nc['experiment']):
#     data = nc['ensemble'].sel(**{'experiment':exp,'stage':'forecast',**options})
#     Y = [obs_func(e) for e in data.data]
#     plt.plot(Y, label=exp+' for', linestyle='-')
#     data = nc['ensemble'].sel(**{'experiment':exp,'stage':'analysis',**options})
#     Y = [obs_func(e) for e in data.data]
#     plt.plot(Y, label=exp+' ana', linestyle='--')
# xx = nc['truth'].sel(**{'experiment':'ETKF',**options})
# plt.plot([0,63],obs_func(xx.data)*np.array([1,1]),'k--',label='truth')
# plt.legend(loc='lower right')

# plt.figure()
# for exp in np.array(nc['experiment']):
#     if 'latent_ensemble' not in nc:
#         continue
#     data = nc['latent_ensemble'].sel(**{'experiment':exp,'stage':'forecast',**options})
#     plt.plot(data.data, label=exp+' for', linestyle='-')
#     data = nc['latent_ensemble'].sel(**{'experiment':exp,'stage':'analysis',**options})
#     plt.plot(data.data, label=exp+' ana', linestyle='--')
#     xx = nc['latent_truth'].sel(**{'experiment':exp,**options})
#     plt.plot([0,63],xx.data*np.array([1,1]),'k--',label='truth')
# plt.legend(loc='lower right')




