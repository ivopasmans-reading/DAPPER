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

plt.close('all')

fig_dir = stommel.fig_dir+'/default'
shutil.copy(__file__, fig_dir) 

def exp_ref_forcing_da(N=100, seed=1000):
    np.random.seed(seed)
    # Timestepping. Timesteps of 1 day, running for 200 year.
    kko = np.arange(1, len(hadley['yy'][1:]))
    tseq = modelling.Chronology(stommel.year/12, kko=kko,
                                T=19*stommel.year)  # 1 observation/month
    
    
    # Switch on heat exchange with atmosphere.
    # Start with default stationary surface temperature and salinity.
    default_temps = stommel.hadley_air_temp(N)    
    default_salts = stommel.hadley_air_salt(N)
    # Initial conditions
    model_ref = stommel.StommelModel(fluxes=[stommel.TempAirFlux(default_temps),
                                             stommel.SaltAirFlux(default_salts)])
    x0 = model_ref.x0
    
    # Add additional periodic forcing
    temp_forcings, salt_forcings = stommel.budd_forcing(model_ref, model_ref.init_state, 86., 0.0,
                                                        stommel.Bhat(0.0, 0.0), 0.00) 
    temp_forcings, salt_forcings = temp_forcings * N, salt_forcings * N
    temp_forcings = [stommel.add_functions(f0, f1) for f0, f1 
                     in zip(default_temps, temp_forcings)]
    salt_forcings = [stommel.add_functions(f0, f1) for f0, f1 
                     in zip(default_salts, salt_forcings)]
    
    model = stommel.StommelModel(fluxes=[stommel.TempAirFlux(default_temps),
                                         stommel.SaltAirFlux(default_salts)])
    
   
    # Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
    # salt diffusion parameter, transport parameter
    B = stommel.State().zero()
    B.temp += hadley['R'][:2]  # C2
    B.salt += hadley['R'][2:]  # ppt2
    B.temp_diff += np.log(1.3)**2  # (0.0*model.init_state.temp_diff)**2
    B.salt_diff += np.log(1.3)**2  # (0.0*model.init_state.salt_diff)**2
    B.gamma += np.log(1.3)**2  # (0.0*model.init_state.gamma)**

    print('ETA ',model.eta1(model.init_state),model.eta2(model.init_state),
          model.eta3(model.init_state))

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
    Obs = model.obs_hadley(factor=np.array([1.,1.,1.,1.]))
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt', N, infl=[1.02]*4+[1.01]*3)
    return xp, HMM, model

if __name__ == '__main__':
    xp, HMM, model = exp_ref_forcing_da()

    # Run
    xx, yy = HMM.simulate()
    yy = hadley['yy'][HMM.tseq.kko]
    Efor, Eana = xp.assimilate(HMM, xx, yy)
         
    
    # Plot
    fig, ax = stommel.time_figure_with_phase(HMM.tseq)
    for n in range(np.size(Efor, 1)):
        stommel.plot_truth_with_phase(ax, HMM, model, Efor[:, n, :], yy,classification='')
    fig.savefig(os.path.join(fig_dir,'truth_with_phase.png'),format='png',dpi=300)
   

    # Calculate etas
    states = stommel.array2states(stommel.ens_modus(Efor), HMM.tseq.times)
    model.ens_member = 0
    etas = np.array([(model.eta1(s), model.eta2(s), model.eta3(s))
                    for s in states])
    trans = np.array([model.fluxes[0].transport(
        s)*np.mean(model.dx*model.dz) for s in states]).flatten()

    fig = plt.figure()
    ax = fig.subplots(1, 1)
    for n, eta in enumerate(etas.T):
        ax.plot(HMM.tseq.times/stommel.year+2004, eta, label='eta'+str(n+1))
    ax.grid()
    ax.set_xlabel('Time [year]')
    plt.axvline(x=2023, linestyle = '--', color='k')
    plt.legend()
    fig.savefig(os.path.join(fig_dir,'etas.png'),format='png',dpi=600)

    # Plot spread
    fig, ax = stommel.time_figure(HMM.tseq)
    stommel.plot_relative_spread(ax, HMM.tseq, Efor, yy)
    fig.savefig(os.path.join(fig_dir,'relative_spread.png'),format='png',dpi=300)

    fig = plt.figure()
    ax = fig.subplots(1, 1)
    for n, eta in enumerate(etas.T):
        ax.plot(HMM.tseq.times/stommel.year, eta, label='eta'+str(n+1))
    ax.grid()
    ax.set_xlabel('Time [year]')
    plt.legend()
    fig.savefig(os.path.join(fig_dir,'etas.png'),format='png',dpi=300)

    #Plot transport
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    ax.plot(HMM.tseq.times/stommel.year, trans/1e6)
    ax.plot(HMM.tseq.times/stommel.year, np.ones_like(HMM.tseq.times)
            * stommel.Q_overturning/1e6, 'k--')
    ax.grid()
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('Transport [Sv]')
    fig.savefig(os.path.join(fig_dir,'transport.png'),format='png',dpi=300)
    
    #Plot output
    fig, ax = plt.subplots(4,2,figsize=(8,11))
    fig.subplots_adjust(hspace=.3,wspace=.4)
    ax = np.array(ax).ravel()
    labels=['Temperature pole [C]','Temperature equator [C]',
            'Salinity pole [ppt]','Salinity equator [ppt]',
            'Temp. diffusivity [ms-1]', 'Salt. diffusivity [ms-1]',
            'Advection parameter [ms-1]']
    for e in Efor.transpose((1,2,0)):
        for ax1, e1, label1 in zip(ax,e,labels):
            ax1.set_xlabel('Time [yr]')
            ax1.set_ylabel(label1)
            ax1.grid('on')
            if ('[ms-1]' in label1):
                e1 = np.exp(e1)
            ax1.plot(HMM.tseq.kk, e1, 'b-')
            
                
        for ax1, y1 in zip(ax,hadley['yy'].transpose((1,0))):
            ax1.plot(HMM.tseq.kko, y1[1:-1], 'k.')
    fig.savefig(os.path.join(fig_dir,'states.png'),format='png',dpi=300)
    
    #Save 1st and last analysis
    with open(os.path.join(fig_dir,'ana_stats.pkl'),'wb') as stream:
        stats = {'mean0':np.mean(Eana[0],axis=0),
                 'mean1':np.mean(Eana[-1],axis=0),
                 'var0':np.var(Eana[0],axis=0,ddof=1),
                 'var1':np.var(Eana[-1],axis=0,ddof=1),
                 'times':[HMM.tseq.otimes[0],HMM.tseq.otimes[-1]],
                 }
        pkl.dump(stats,stream)
        
        
#%% Print best estimates and 95 confidence interval
from scipy.stats import norm 

mu, sig = np.mean(Eana[-1],0)[4:], np.std(Eana[-1],0,ddof=1)[4:]
best = np.exp(stommel.ens_modus(Eana)[-1,4:])

for b1, mu1, sig1 in zip(best, mu, sig):
    norm1 = norm(loc=mu1, scale=sig1)
    print('(best, confidence 90)',b1, np.exp(norm1.ppf(.05)), np.exp(norm1.ppf(.95)))
    
A_Tair = np.sqrt(np.sum(np.diff(hadley['surface_temperature_harmonic'],axis=1)[1:]**2))
A_Sair = np.sqrt(np.sum(np.diff(hadley['surface_salinity_harmonic'],axis=1)[1:]**2))
dys = [hadley['geo_eq']['dy'],hadley['geo_pole']['dy']]
dybar = np.prod(dys)/np.sum(dys)
print('Omega dimensionaless ',2*np.pi / stommel.year * (hadley['geo_eq']['dz']/best[0]))
print('Tair amplitude', A_Tair, A_Tair * (model.eos.alpha_T * best[-1] / best[0]) * hadley['geo_eq']['dz']/ dybar)
print('Sair amplitude', A_Sair, A_Sair * (best[1]/best[0]) * (model.eos.alpha_S * best[-1] / best[0]) * hadley['geo_eq']['dz'] / dybar)