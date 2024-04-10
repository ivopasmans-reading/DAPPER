# -*- coding: utf-8 -*-

""" 
As ref_forcing, but now temperature and salinity are assimilated.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy
import os
import pickle as pkl
from dapper.mods.Stommel import hadley
import shutil
 

dict_DIR = os.path.join(stommel.DIR,'paramdict.pkl')
#This pkl stores RMSE and spread data for error plots
#It can be generated with reset_pickle.py
#The error plots can be generated via pklPlot.py once
#all four twin experiments are run
with open(dict_DIR, 'rb') as handle:
    error_dict = pkl.load(handle)

#Experiment names toggle the warming and data assimilation settings.
# - Warming vs noWarming is self-explanatory
# - NoDa means no assimilation
# - SynthDA means synthetic data assimilated
# - DA means assimilation using Hadley observations

#First four names make up the twin experiments
#experiment_name = 'noWarmingNoDA'
#experiment_name = 'WarmingNoDA'
#experiment_name = 'WarmingSynthDA'
#experiment_name = 'noWarmingSynthDA'
#experiment_name = 'noWarmingDA'
experiment_name = 'WarmingDA'


fig_dir = os.path.join(stommel.fig_dir, f'{experiment_name}')
#fig_dir = stommel.fig_dir
shutil.copy(__file__, fig_dir)


#Length of experiment
LENGTH = 100
#Time period for observations 
kko = np.arange(1, len(hadley['yy'][1:]))
#Number of months of observations 
tseq = modelling.Chronology(stommel.year/12, kko=kko,
                            T=LENGTH*stommel.year)  # 1 observation/month
T0 = np.max(tseq.tto)


def exp_ref_forcing_da(N=100, seed=1800, with_da=False):
    np.random.seed(seed)
    # Timestepping. Timesteps of 1 day, running for 200 year.
    if with_da:
        kko = np.arange(1, len(hadley['yy'][1:]))
    else:
        kko = np.array([])
    tseq = modelling.Chronology(stommel.year/12, kko=kko,
                                T=LENGTH*stommel.year)  # 1 observation/month
    
    def clima_T(t):
        """ Warming surface due climate change in K. """
        if t<T0:
            #No warming over DA period
            return np.array([0,0])
        else:
            #.06/.03 C warming per year over pole/equator
            return np.array([.06,.03]) * (t-T0) / stommel.year
        
    def clima_S(t):
        """ Freshening due to melt Greenland. """
        volume = 2.9e15 #m3
        melt_period = 10000 * stommel.year 
        A = model.dx[0,:] * model.dy[0,:] #area stommel boxes 

        if t<T0:
            return np.array([0,0])
        else:
            return np.array([-volume/melt_period,0]) / A
        
    # Create default Stommel model
    model = stommel.StommelModel()
    # Switch on heat exchange with atmosphere.
    # Start with default stationary surface temperature and salinity.
    default_temps = stommel.hadley_air_temp(N)
    temp_forcings = default_temps #temporary line

    #Add surface salt forcing
    default_salts = stommel.hadley_air_salt(N)
    model.fluxes.append(stommel.SaltAirFlux(default_salts))
    
    if experiment_name in ['WarmingDA', 'WarmingNoDA', 'WarmingSynthDA']:
        temp_forcings = [stommel.add_functions(f, clima_T) for f in default_temps]
        #Add melt
        melt_rates = [clima_S for _ in range(N)]
        model.fluxes.append(stommel.EPFlux(melt_rates))
        print('Using Warming')
    else:
        print('No Warming')
    model.fluxes.append(stommel.TempAirFlux(temp_forcings))   
    # Initial conditions
    x0 = model.x0
    # Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
    # salt diffusion parameter, transport parameter
    B = stommel.State().zero()
    B.temp += hadley['R'][:2]  # C2
    B.salt += hadley['R'][2:]  # ppt2
    B.temp_diff += np.log(1.3)**2  # (0.0*model.init_state.temp_diff)**2
    B.salt_diff += np.log(1.3)**2  # (0.0*model.init_state.salt_diff)**2
    B.gamma += np.log(1.3)**2  # (0.0*model.init_state.gamma)**

    # Transform modus value in x0 to mean value.
    x0[4] += B.temp_diff
    x0[5] += B.salt_diff
    x0[6] += B.gamma

    #Create sampler for initial conditions. 
    X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
    # Dynamisch model. All model error is assumed to be in forcing.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0
           }
    # Observation
    Obs = model.obs_hadley()
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt', N, infl=1.0)
    return xp, HMM, model

if __name__ == '__main__':
    with_da = experiment_name in ['noWarmingDA','WarmingDA','noWarmingSynthDA', 'WarmingSynthDA']
    xp, HMM, model = exp_ref_forcing_da(seed=1800,with_da=with_da)
    # Run
    xx, yy = HMM.simulate()

    if experiment_name in ['noWarmingDA', 'WarmingDA']:
        yy = hadley['yy'][HMM.tseq.kko]
        print('Using Hadley')
    elif experiment_name in ['noWarmingSynthDA', 'WarmingSynthDA']:
        print('Synthetic Data')
    else:
        print('No Data')
        
    Efor, Eana = xp.assimilate(HMM, xx, yy)

    # Calculate etas
    states = stommel.array2states(stommel.ens_modus(Efor), HMM.tseq.times)
    model.ens_member = 0
    etas = np.array([(model.eta1(s), model.eta2(s), model.eta3(s))
                    for s in states])
    trans = np.array([model.fluxes[0].transport(
        s)*np.mean(model.dx*model.dz) for s in states]).flatten()

    # Plot
    fig, ax = stommel.time_figure_with_phase(HMM.tseq)
    for n in range(np.size(Efor, 1)):
        stommel.plot_truth_with_phase(ax, HMM, model, Efor[:, n, :], yy, 'member')
    
    if experiment_name in ['noWarmingSynthDA', 'WarmingSynthDA','noWarmingNoDA', 'WarmingNoDA']:
        stommel.plot_truth_with_phase(ax, HMM, model, stommel.ens_modus(Efor), yy, 'modus')
        stommel.plot_truth_with_phase(ax, HMM, model, xx, yy, 'truth')
    if with_da:
        stommel.plot_data(ax, HMM, model, yy,experiment_name)
    
    fig.savefig(os.path.join(fig_dir,'truth_with_phase.png'),format='png',dpi=600)


    # Plot spread
    fig, ax = stommel.time_figure(HMM.tseq)
    
    def plot_relative_spread_here(axes, tseq, E, yy):
        names = ['temp_pole','temp_eq','salt_pole','salt_eq',
                 'temp_diff', 'salt_diff','adv']
        E = E.T
        yy = yy.T

        spread_for_dict = []
        ctimes = set(tseq.times).intersection(tseq.otimes)
        mask  = [t1 in ctimes for t1 in tseq.times ]
        masko = [t1 in ctimes for t1 in tseq.otimes]
        
        nyy = np.size(yy,0)
        for name, field, obs in zip(names[:nyy], E[:nyy], yy):
            std = np.std(field, axis=0, ddof=1)
            mu  = np.mean(field, axis=0)
            #h = axes[0].plot(tseq.times/stommel.year, std/std[0], label=name)
            h = axes[0].plot(tseq.times/stommel.year, std, label=name)
            #spread_for_dict.append(std/std[0])
            spread_for_dict.append(std)
            
        for name, field in zip(names[nyy:], E[nyy:]):
            std = np.std(field, axis=0, ddof=1)
            #h = axes[1].plot(tseq.times/stommel.year, std/std[0], label=name)
            h = axes[1].plot(tseq.times/stommel.year, std, label=name)
            #spread_for_dict.append(std/std[0])
            spread_for_dict.append(std)
            
        error_dict[experiment_name]['Spread'] = spread_for_dict
        
        for ax in axes:
            ax.legend(loc='upper left', ncol=2)
            ax.set_ylabel('Relative spread')
            ax.grid(which='major', color=(.7,.7,.7), linestyle='-')
    if experiment_name in ['noWarmingSynthDA', 'WarmingSynthDA','noWarmingNoDA', 'WarmingNoDA']:
        plot_relative_spread_here(ax, HMM.tseq, Efor, yy)
        fig.savefig(os.path.join(fig_dir,'relative_spread.png'),format='png',dpi=600)
    
    #plot RMSE
    def error_figure():
    	plt.close('all')
    	fig = plt.figure(figsize=(11,6))
    	ax = fig.subplots(2,3)
    	fig.subplots_adjust(left=.1,right=.98,wspace=.3)
    
    	for ax1 in np.reshape(ax,(-1)):
    	    ax1.grid()
        
    	return fig, ax
    
    def plot_error(ax, errors):
        errors_for_dict = []
        draws = []
        ax = np.reshape(ax,(-1))
        
        errors = np.array(errors)
        temp = errors[:,:,1]-errors[:,:,0]
        salt = errors[:,:,3]-errors[:,:,2]
            
        x=np.linspace(0,LENGTH, np.size(temp,1))
        ax[0].plot(x, np.sqrt(np.mean(temp**2,axis=0)))
        for n in range(4):
            errors_for_dict.append(np.sqrt(np.mean(errors[:,:,n]**2,axis=0)))
        
        draw, = ax[1].plot(x, np.sqrt(np.mean(salt**2,axis=0)))
        draws.append(draw)
            
        for n,ax1 in enumerate(ax[3:]):
            rmse = np.sqrt(np.mean(errors[:,:,n+4]**2,axis=0))
            ax1.plot(x, rmse)
            errors_for_dict.append(rmse)
            
        ax[0].set_ylabel('RMSE temp diff. [C]')
        ax[1].set_ylabel('RMSE salt diff. [ppt]')
        ax[3].set_ylabel('RMSE temp_diff [mms-1]')
        ax[4].set_ylabel('RMSE salt_diff [mms-1]')
        ax[5].set_ylabel('RMSE gamma [ms-1]')
        for ax1 in ax:
            ax1.set_xlabel('Time [year]')
        
        error_dict[experiment_name]['RMSE'] = errors_for_dict
        
        return draws
    
    if experiment_name in ['noWarmingSynthDA', 'WarmingSynthDA','noWarmingNoDA', 'WarmingNoDA']:
        errors = []
        errors.append(np.mean(Efor,axis=1)-xx)
        fig, ax = error_figure()
        plot_error(ax, errors)
        fig.savefig(os.path.join(fig_dir, 'rmse_time.png'),
                    format='png', dpi=600)

#plot etas
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    for n, eta in enumerate(etas.T):
        ax.plot(HMM.tseq.times/stommel.year+2004, eta, label='eta'+str(n+1))
    ax.grid()
    ax.set_xlabel('Time [year]')
    plt.axvline(x=2023, linestyle = '--', color='k')
    plt.legend()
    fig.savefig(os.path.join(fig_dir,'etas.png'),format='png',dpi=600)

    #Plot transport
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    ax.plot(HMM.tseq.times/stommel.year+2004, trans/1e6)
    ax.plot(HMM.tseq.times/stommel.year+2004, np.ones_like(HMM.tseq.times)
            * stommel.Q_overturning/1e6, 'k--')
    ax.grid()
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('Transport [Sv]')
    ax.axvline(x=2023, linestyle = '--', color='k')
    fig.savefig(os.path.join(fig_dir,'transport.png'),format='png',dpi=600)
    
    #Save 1st and last analysis
    if len(Eana)==0:
        with open(os.path.join(fig_dir,'ana_stats.pkl'),'wb') as stream:
            stats = {'mean0':np.mean(Efor[0],axis=0),
                     'mean1':np.mean(Efor[-1],axis=0),
                     'var0':np.var(Efor[0],axis=0,ddof=1),
                     'var1':np.var(Efor[-1],axis=0,ddof=1),
                     'times':[HMM.tseq.times[0],HMM.tseq.times[-1]],
                     }
            pkl.dump(stats,stream)
    else:
        with open(os.path.join(fig_dir,'ana_stats.pkl'),'wb') as stream:
            stats = {'mean0':np.mean(Eana[0],axis=0),
                     'mean1':np.mean(Eana[-1],axis=0),
                     'var0':np.var(Eana[0],axis=0,ddof=1),
                     'var1':np.var(Eana[-1],axis=0,ddof=1),
                     'times':[HMM.tseq.otimes[0],HMM.tseq.otimes[-1]],
                     }
            pkl.dump(stats,stream)
            
        
#Plot parameters
    fig = plt.figure(figsize=(14, 4))
    ax = fig.subplots(1,3)
    
    for ax1 in ax:
        ax1.grid()
        ax1.set_xlabel('Time [year]')
        
    ax[0].set_ylabel('Surface Temperature Flux Coefficient')
    ax[1].set_ylabel('Surface Salinity Flux Coefficient')
    ax[2].set_ylabel('Advective Transport Flux Coefficient')
    
    for n in range(np.size(Efor, 1)):
        stommel.plot_more_spreads(ax, HMM, model, Efor[:, n, :], yy)
    
    #plot modus
    states = stommel.array2states(stommel.ens_modus(Efor), HMM.tseq.times)
    model.ens_member = 0
    temp_diff_arr = np.reshape([np.exp(s.temp_diff) for s in states], (-1))
    salt_diff_arr = np.reshape([np.exp(s.salt_diff) for s in states], (-1))
    gamma_arr = np.reshape([np.exp(s.gamma) for s in states], (-1))
    datime = HMM.tseq.times / stommel.year + 2004
    stop = len(HMM.tseq.tto)
    ax[0].plot(datime[0:stop], temp_diff_arr[0:stop], c='r')
    ax[1].plot(datime[0:stop], salt_diff_arr[0:stop], c='r')
    ax[2].plot(datime[0:stop], gamma_arr[0:stop], c='r')
    #plot truth
    if experiment_name in ['noWarmingSynthDA', 'WarmingSynthDA','noWarmingNoDA', 'WarmingNoDA']:
        states = stommel.array2states(xx, HMM.tseq.times)
        temp_diff_arr = np.reshape([np.exp(s.temp_diff) for s in states], (-1))
        salt_diff_arr = np.reshape([np.exp(s.salt_diff) for s in states], (-1))
        gamma_arr = np.reshape([np.exp(s.gamma) for s in states], (-1))
        ax[0].plot(datime[0:stop], temp_diff_arr[0:stop], c='g')
        ax[1].plot(datime[0:stop], salt_diff_arr[0:stop], c='g')
        ax[2].plot(datime[0:stop], gamma_arr[0:stop], c='g')
        
    fig.savefig(os.path.join(fig_dir,'params.png'),format='png',dpi=600)
    
    error_dict['time'] = HMM.tseq.times/stommel.year + 2004
    with open(dict_DIR, 'wb') as handle:
        pkl.dump(error_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
