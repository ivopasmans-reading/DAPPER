# -*- coding: utf-8 -*-

""" 
Used to generate contourplot by running WarmingDA many times with varying warming parameters
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

import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path

fig_dir = stommel.fig_dir
shutil.copy(__file__, fig_dir) 

#Time period for observations 
LENGTH = 100
kko = np.arange(1, len(hadley['yy'][1:]))
tseq = modelling.Chronology(stommel.year/12, kko=kko,
                            T=LENGTH*stommel.year)  # 1 observation/month
T0 = np.max(tseq.tto)


#Number of months of observations 
def exp_ref_forcing_da(melt,t_diff,N=100, seed=1000, with_da=False):
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
        nonlocal t_diff
        if t<T0:
            #No warming over DA period
            return np.array([0,0])
        else:
            #.06/.03 C warming per year over pole/equator
            #return np.array([.06,.03]) * (t-T0) / stommel.year
            return np.array([2*t_diff,t_diff]) * (t-T0) / stommel.year
             
    def clima_S(t):
        """ Freshening due to melt Greenland. """
        nonlocal melt
        volume = 2.9e15 #m3
        melt_period = melt * stommel.year 
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
    temp_forcings = [stommel.add_functions(f, clima_T) for f in default_temps]
    model.fluxes.append(stommel.TempAirFlux(temp_forcings))
    #Add surface salt forcing
    default_salts = stommel.hadley_air_salt(N)
    model.fluxes.append(stommel.SaltAirFlux(default_salts))
    #Add melt
    melt_rates = [clima_S for _ in range(N)]
    model.fluxes.append(stommel.EPFlux(melt_rates))
    
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
#Edit melts and t_diffs to change contourmap mesh
    melts = np.arange(400, 4001, 100)
    t_diffs = np.arange(0,.071,.01)
    #melts = np.arange(100, 500, 200)
    #t_diffs = np.arange(.01,.03,.01)
    mm,tt = np.meshgrid(melts,t_diffs)
    probs = np.zeros(mm.shape)
    runs = probs.size
    count = 1
    
    for (i,melt) in zip(range(len(melts)),melts):
        for (j,t_diff) in zip(range(len(t_diffs)),t_diffs):
            print(f'Run {count} of {runs}')
            count = count +1
            xp, HMM, model = exp_ref_forcing_da(melt,t_diff,with_da=True)
        
            # Run
            xx, yy = HMM.simulate()
            yy = hadley['yy'][HMM.tseq.kko]
            Efor, Eana = xp.assimilate(HMM, xx, yy)
            
            probs[j,i] = stommel.prob_change(Efor)*100
            """
            fig, ax = stommel.time_figure_with_phase(HMM.tseq)
            for n in range(np.size(Efor, 1)):
                stommel.plot_truth_with_phase(ax, HMM, model, Efor[:, n, :], yy,xx)
            fig.savefig(os.path.join(fig_dir,f'truth_with_phase_{i}_{j}.png'),format='png',dpi=300)
            """
    fig, ax = plt.subplots()
    ax.set_xlabel("Melt Period (years)")
    ax.set_ylabel("Yearly Temperature Warming (C)")
    cont1 = ax.contourf(mm,tt,probs, levels = np.arange(0,101,1),norm=cm.colors.Normalize(vmax=100, vmin=0), cmap = cm.bwr)
    clb = fig.colorbar(cont1, ax=ax, orientation='vertical', fraction=.1)
    clb.set_ticks(np.arange(0, 101, 20))
    clb.set_label('Percent of Ensemble Flipping',labelpad = 10, rotation = 270)
    fig.savefig(os.path.join(fig_dir,'melt_clima_probs'),format='png',dpi=300)
    
    
    melt_values = {'mm': mm, 'tt':tt, 'probs': probs}
    with open(os.path.join(fig_dir,'melt_values.pkl'), 'wb') as handle:
    	pkl.dump(melt_values, handle, protocol=pkl.HIGHEST_PROTOCOL)
