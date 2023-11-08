# -*- coding: utf-8 -*-

""" 
Run Stommel model with perturbations in initial conditions, model parameters
and model forcing. For t<=Tda truth and ensemble members experience the same
forcing.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy
import os


def exp_ref_forcing(N=100, seed=1000):
    Tda = 20 * stommel.year  # time period over which DA takes place.
    # Timestepping. Timesteps of 1 day, running for 200 year.
    tseq = modelling.Chronology(stommel.year/12, kko=np.array([], dtype=int),
                                T=50*stommel.year, BurnIn=0)  # no DA
    # Create default Stommel model
    model = stommel.StommelModel()
    # Switch on heat exchange with atmosphere.
    # Start with default stationary atm. temperature.
    functions = stommel.default_air_temp(N)
    # Add white noise with std dev of 2C over both pole and equator basin separately.
    noised = [stommel.add_noise(func, seed=seed+n*20+1, sig=np.array([2., 2.]))
              for n, func in enumerate(functions)]
    default_temps =  [stommel.merge_functions(Tda, noised[0], func2)
                      for func2 in noised]
    # Switch on salinity exchange with atmosphere.
    # Start with default stationary atm. salinity.
    functions= stommel.default_air_salt(N)
    # Add white with std dev. of .2 ppt.
    noised = [stommel.add_noise(func, seed=seed+n*20+2, sig=np.array([.2, .2]))
              for n, func in enumerate(functions)]
    default_salts = [stommel.merge_functions(Tda, noised[0], func2)
                 for func2 in noised]
    # Switch on salinity fluxes.
    model.fluxes.append(stommel.SaltAirFlux(functions))
    # Default initial conditions
    x0 = model.x0
    #Add additional periodic forcing 
    temp_forcings, salt_forcings = stommel.budd_forcing(model, model.init_state, 10., 5.0, 
                                                        stommel.Bhat(4.0,5.0), 0.01)
    temp_forcings = [stommel.add_functions(f0,f1) for f0,f1 in zip(default_temps,temp_forcings)]
    salt_forcings = [stommel.add_functions(f0,f1) for f0,f1 in zip(default_salts,salt_forcings)]
    model.fluxes.append(stommel.TempAirFlux(temp_forcings))
    model.fluxes.append(stommel.SaltAirFlux(salt_forcings))
    # Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
    # salt diffusion parameter, transport parameter
    B = stommel.State().zero()
    B.temp += 0.5**2  # C2
    B.salt += 0.05**2  # ppt2
    B.temp_diff += (0.3*model.init_state.temp_diff)**2
    B.salt_diff += (0.3*model.init_state.salt_diff)**2
    B.gamma += (0.3*model.init_state.gamma)**2
    X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
    # Dynamisch model. All model error is assumed to be in forcing.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0
           }
    # Default observation/
    Obs = model.obs_ocean()
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt', N)

    return xp, HMM, model


if __name__ == '__main__':
    xp, HMM, model = exp_ref_forcing()

    # Run
    xx, yy = HMM.simulate()
    Efor, Eana = xp.assimilate(HMM, xx, yy)

    # Plot
    fig, ax = stommel.time_figure_with_phase(HMM.tseq)
    for n in range(np.size(Efor, 1)):
        stommel.plot_truth_with_phase(ax, model, Efor[:, n, :], yy)

    # Add equilibrium based on unperturbed initial conditions.
    model.ens_member = 0
    stommel.plot_all_eq(ax, HMM.tseq, model, xx, stommel.prob_change(Efor) * 100.)
    stommel.plot_eq(ax, HMM.tseq, model, stommel.prob_change(Efor) * 100.)

    # Save figure
    fig.savefig(os.path.join(stommel.fig_dir, 'forcing_noise.png'),
                format='png', dpi=500)