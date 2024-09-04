# -*- coding: utf-8 -*-

""" As clime_forcing but now with assimilation of ocean temperature and salinity. """
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy 
import os
import pickle as pkl
from dapper.mods.Stommel import hadley

#Import HADLEY data 
#DIR = "/home/ggorblin/Downloads/analysis/EN.4.2.2.analyses.g10.2019"
#with open(os.path.join(DIR,'yy.pkl'), 'rb') as stream:
#    hadley = pkl.load(stream)

def exp_clima_forcing_da(N=100, seed=1000):
    #Time period for DA
    Tda = 20 * stommel.year #time period over which DA takes place. 
    kko = [] #np.arange(1, len(hadley['yy'])+1)
    tseq = modelling.Chronology(stommel.year/12, kko=kko, 
                                T=10*stommel.year, BurnIn=0)  # 1 observation/month
    T_warming=100*stommel.year
    # Timestepping. Timesteps of 1 day, running for 200 year.
    #Create model
    model = stommel.StommelModel()
    #Heat air fluxes 
    functions = stommel.default_air_temp(N)
    #Add linear warming with 6C/T_warming over the pole and 3C/T_warming over the equatior. 
    trend = interp1d(np.array([0.,T_warming]), np.array([[0.,6.],[0.,3.]]), 
                     fill_value='extrapolate', axis=1)
    trended = [stommel.add_functions(func, trend) for func in functions]
    #Add random temperature perturbations with std dev. of 2C
    noised = [stommel.add_noise(func, seed=seed+n*20+1, sig=np.array([2.,2.])) 
              for n,func in enumerate(trended)]
    default_temps = [stommel.merge_functions(Tda, noised[0], func2) 
                 for func2 in noised]
    #For time<Tda all ensemble member n uses noised[0] after that noised[n]
    functions = [stommel.merge_functions(Tda, noised[0], func) 
                 for func in noised]
    #Activate surface heat flux. functions[n] contains atm. temperature for ensemble member n. 
    model.fluxes.append(stommel.TempAirFlux(functions))
    #Salinity air fluxes 
    functions = stommel.default_air_salt(N)
    #Add random salinity perturbations with std dev. of 0.2ppt
    noised = [stommel.add_noise(func, seed=seed+n*20+2, sig=np.array([.2,.2])) 
              for n,func in enumerate(functions)]
    default_salts = [stommel.merge_functions(Tda, noised[0], func2) 
                 for func2 in noised]
    #For time<Tda all ensemble member n uses noised[0] after that noised[n]
    functions = [stommel.merge_functions(Tda, noised[0], func) 
                 for func in noised]
    #Activate surface salinity flux. 
    model.fluxes.append(stommel.SaltAirFlux(functions))
    #Melt flux 
    melt_rate = -stommel.V_ice * np.array([1.0/(model.dx[0,0]*model.dy[0,0]), 0.0]) / T_warming #ms-1
    #Default evaporation-percipitation flux (=0)
    functions = stommel.default_air_ep(N)
    #Add effect Greenland melt with annual rate melt_rate
    functions = [stommel.merge_functions(T_warming, lambda t:func(t)+melt_rate, func)
                 for func in functions]
    #Activate EP flux. 
    model.fluxes.append(stommel.EPFlux(functions))
    #Default initial conditions
    x0 = model.x0
    temp_forcings, salt_forcings = stommel.budd_forcing(model, model.init_state, 10., 5.0, 
                                                        stommel.Bhat(4.0,5.0), 0.01)
    temp_forcings = [stommel.add_functions(f0,f1) for f0,f1 in zip(default_temps,temp_forcings)]
    salt_forcings = [stommel.add_functions(f0,f1) for f0,f1 in zip(default_salts,salt_forcings)]
    model.fluxes.append(stommel.TempAirFlux(temp_forcings))
    model.fluxes.append(stommel.SaltAirFlux(salt_forcings))
    #Variance in initial conditions and parameters.
    B = stommel.State().zero()
    B.temp += np.mean(hadley['R'][:2]) #C2
    B.salt += np.mean(hadley['R'][2:]) #ppt2
    B.temp_diff += (0.3*model.init_state.temp_diff)**2
    B.salt_diff += (0.3*model.init_state.salt_diff)**2
    B.gamma += (0.3*model.init_state.gamma)**2
    X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
    # Dynamisch model. All model error is assumed to be in forcing.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0
           }
    #Default observations. 
    Obs = model.obs_ocean()
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt',N)
    
    return xp, HMM, model

if __name__=='__main__':
    xp, HMM, model = exp_clima_forcing_da()
    
    #Run
    xx, yy = HMM.simulate()
    yy = hadley['yy']
    Efor, Eana = xp.assimilate(HMM, xx, yy)
    
    #Plot
    fig, ax = stommel.time_figure_with_phase(HMM.tseq)    
    for n in range(np.size(Efor,1)):
        stommel.plot_truth_with_phase(ax, model,Efor[:,n,:], yy)
        
    #Add equilibrium based on unperturbed initial conditions. 
    model.ens_member=0
    stommel.plot_all_eq(ax, HMM.tseq, model, xx, stommel.prob_change(Efor) * 100.)
    stommel.plot_eq(ax, HMM.tseq, model, stommel.prob_change(Efor) * 100.)
    
    #Save figure 
    fig.savefig(os.path.join(stommel.fig_dir, 
                             'clima_forcing_da.png'),format='png',dpi=500)
    
    

    
    
    

