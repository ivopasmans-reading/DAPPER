# -*- coding: utf-8 -*-

""" 
Run model over a range of climate parameters. 
"""

import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
from dapper.mods.Stommel import hadley
from scipy.interpolate import interp1d
import dill, os
import multiprocessing
import concurrent.futures

def climate_temp_forcing(N, T_da, T_warming=0.0, A_polar=0.0):
    """ Create surface forcing for temperatur flux due to global warming. """
    #Heat air fluxes 
    functions = stommel.hadley_air_temp(N)
    #Add linear warming with 6C/T_warming over the pole and 3C/T_warming over the equatior. 
    trend = interp1d(np.array([0.,T_da,T_warming+T_da]), 
                     np.array([[0.,0.,A_polar],[0.,0.,0.5*A_polar]]), 
                     fill_value='extrapolate', axis=1)
    trended = [stommel.add_functions(func, trend) for func in functions]
    #For time<Tda all ensemble member n uses noised[0] after that 
    #functions = [stommel.merge_functions(T_da, func, trend) 
    #             for func,trend in zip(functions,trended)]
    return trended

def climate_ep_forcing(N, model, T_da, T_melt):
    """ Create evaporation-precipitation forcing due to Greenland ice sheet melt."""
    #Melt flux 
    melt_rate = -stommel.V_ice * np.array([1.0/(model.dx[0,0]*model.dy[0,0]), 0.0]) / T_melt #ms-1
    #Default evaporation-percipitation flux (=0)
    functions = stommel.default_air_ep(N)
    #Add effect Greenland melt with annual rate melt_rate
    def add_melting(func):
        def func_with_melt(t):
            if t<T_da:
                return func(t)
            elif t<T_da+T_melt:
                return func(t)+melt_rate
            else:
                return func(t)
        return func_with_melt
    
    functions = [add_melting(func) for func in functions]
    return functions


def experiment(N=100, seed=1100, do_da=True, 
               A_polar=0.0, T_warming = np.inf, T_melt=np.inf):
    #Set seed
    np.random.seed(seed)
    # Timestepping. Timesteps of 1 day, running for 200 year.
    dt = stommel.year/12
    kko = np.arange(1, len(hadley['yy'][1:])+1) if do_da else np.array([])    
    tseq = modelling.Chronology(dt, kko=kko, T=100*stommel.year)  # 1 observation/month
    T_da = np.size(hadley['yy'],0) * dt
    
    #Activate surface heat flux. functions[n] contains atm. temperature for ensemble member n. 
    temp_forcings = climate_temp_forcing(N, T_da, T_warming, A_polar)
    
    #Salinity air fluxes 
    salt_forcings = stommel.hadley_air_salt(N)
    
    #Activate EP flux. 
    ref_model = stommel.StommelModel(fluxes=[stommel.TempAirFlux(temp_forcings),
                                         stommel.SaltAirFlux(salt_forcings)])
    ep_forcings = climate_ep_forcing(N, ref_model, T_da, T_melt)
    
    #Create model
    model = stommel.StommelModel(fluxes=[stommel.TempAirFlux(temp_forcings),
                                         stommel.SaltAirFlux(salt_forcings),
                                         stommel.EPFlux(ep_forcings)])

    
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
    # Observation. If factor=1 no correction to obs. error std. dev. is applied. 
    Obs = model.obs_hadley(factor=np.array([1.,1.,1.,1.]))
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA with inflation
    xp = EnKF('Sqrt', N, infl=[1.0]*4+[1.01]*3)
    return xp, HMM, model

def flip_probability(HMM,E):
    """ Calculate probability of flip at some time. """
    times = HMM.tseq.times
    N_flips = 0
    end_da = np.max(HMM.tseq.kko)
    for e in E.transpose((1,0,2)):
        states = stommel.array2states(e[end_da:], times[end_da:])
        SA = np.array([s.regime=='SA' for s in states], dtype=bool)
        N_flips += np.any(SA)
    return float(N_flips) / float(np.size(E,1))

#%% Run the experiment

def run_list(filename, observations=[None], T_melts = [np.inf], 
             A_polars=[0.0], labels=['nCC-DA']):
    """
    Run all experiments in sequence. 
    """   
    filepath = os.path.join(stommel.fig_dir, filename+'.pkl')
    if os.path.exists(filepath):
        with open(filepath,'rb') as stream:
            data = dill.load(stream)
    else:
        data = []
    
    for T_melt, A_polar, obs, label in zip(T_melts, A_polars, observations, labels):
        labels = [data1['label'] for data1 in data]
        if label in labels:
            continue
        
        print('Running ',label)
        #Create experiment
        do_da = len(obs)>0
        xp, HMM, model = experiment(T_melt=T_melt, A_polar=A_polar, 
                                    T_warming=100.*stommel.year, do_da=do_da)
        #Run
        xx, yy = HMM.simulate()
        if do_da and str(obs[0])!='twin':
            assert len(yy)==len(obs)
            yy = obs
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        
        data1 = {'model':model,'xp':xp, 'HMM':HMM,
                 'yy': yy, 'Efor':Efor, 'Eana':Eana,
                 'T_melt':T_melt, 'A_polar':A_polar, 'label':label}
        
        if len(obs)==0 or str(obs[0])=='twin':
            data1 = {**data1, 'xx':xx}

        data.append(data1)
    
    with open(filepath,'wb') as stream:
        dill.dump(data, stream)
            
    return data 

def experiment1(param_in):
    from time import sleep
    T_melt, A_polar = param_in
    #Create experiment
    xp, HMM, model = experiment(T_melt=T_melt, A_polar=A_polar,
                                T_warming=100.*stommel.year)
    # #Run
    xx, yy = HMM.simulate()
    yy = hadley['yy'][HMM.tseq.kko]
    Efor, Eana = xp.assimilate(HMM, xx, yy)
    #Calculate flip probility
    return flip_probability(HMM, Efor)


def run_hypercube(filename, T_melts = [np.inf], 
                  A_polars=[0.0]):
    """ 
    Run all combinations of settings. 
    """ 
    
    filepath = os.path.join(stommel.fig_dir, filename+'.pkl')
    climates = [(T_melt, A_polar) for T_melt in T_melts for A_polar in A_polars]
    
    pool = multiprocessing.Pool(processes = 8)
    probs = pool.map(experiment1, climates)
    print('PROBS ',probs)
      
    data = {'climas':climates, 'probabilities':probs}
    with open(filepath,'wb') as stream:
        dill.dump(data, stream)
        
    return data
    
        

