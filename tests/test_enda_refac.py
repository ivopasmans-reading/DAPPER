#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:46:55 2023

Test whether the new approach using EnDa yields the same DA results as obtained 
with old classes. 

@author: ivo
"""

import numpy as np
import dapper.mods as modelling
import dapper.da_methods as da
from dapper.mods.Lorenz63 import dstep_dx, step, x0
from dapper.tools.seeding import set_seed
from itertools import product
from datetime import datetime, timedelta

#%%  Lorentz 63

tseq = modelling.Chronology(0.01, dko=5, Ko=100)

Nx = len(x0)

Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(C=2, mu=x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 2  # modelling.GaussRV(C=CovMat(2*eye(Nx)))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
xx, yy = HMM.simulate()

#%% 

N=16
factory = da.EndaFactory()

def test_enkf():
    filters  = ['Sqrt','Sqrt explicit','Sqrt svd','Sqrt sS']
    filters += ['Serial','Serial Stoch','Serial Stoch Var1']
    filters += ['DEnKF']
    rots = (True, False)
    inflations =(1.0, 1.02)
    times = []
    
    for options in product(filters, rots, inflations):
        old = da.EnKF(options[0], N, rot=options[1], infl=options[2])
        new = factory.build(N, filter=options[0], rot=options[1], 
                      infl=options[2])
        
        times = np.append(times,datetime.now())
        set_seed(1000)
        old.assimilate(HMM, xx, yy)
        times = np.append(times,datetime.now())
        set_seed(1000)
        new.assimilate(HMM, xx, yy)
        times = np.append(times,datetime.now())
        
        error = old.stats.E.a[-1] - new.stats.E.a[-1]
        value = old.stats.E.a[-1] + new.stats.E.a[-1]
        rtol = np.sqrt(2*np.mean(error**2) / np.mean(value**2))
        atol = np.max(np.abs(error))
        trat = (times[-1]-times[-2])/(times[-2]-times[-3])
        
        print(options)
        print('rel RMSE, abs E, rel dt:',rtol,atol,trat)
        
        if not np.all(np.isclose(error,0.)):
            print('Old E ',old.stats.E.a[-1])
            print('New E ',new.stats.E.a[-1])
            raise Exception('Test failed.')
        
    print('Sqrt test passed.')
    
def test_smoother():
    old = da.EnKS('Sqrt', N, 100)
    new = factory.build(N, filter='Sqrt', smoother='EnKS', lag=100)
    times = []
    
    times = np.append(times,datetime.now())
    set_seed(1000)
    old.assimilate(HMM, xx, yy)
    times = np.append(times,datetime.now())
    set_seed(1000)
    new.assimilate(HMM, xx, yy)
    times = np.append(times,datetime.now())
    
    error = old.stats.E.a[-1] - new.stats.E.a[-1]
    value = old.stats.E.a[-1] + new.stats.E.a[-1]
    rtol = np.sqrt(2*np.mean(error**2) / np.mean(value**2))
    atol = np.max(np.abs(error))
    trat = (times[-1]-times[-2])/(times[-2]-times[-3])
    
    print('EnKS')
    print('rel RMSE, abs E, rel dt:',rtol,atol,trat)
    
    old = da.EnRTS('Sqrt', N, DeCorr=.1)
    new = factory.build(N, filter='Sqrt', smoother='EnRTS', lag=100,
                        decorr=.1)
    times = []
    
    times = np.append(times,datetime.now())
    set_seed(1000)
    old.assimilate(HMM, xx, yy)
    times = np.append(times,datetime.now())
    set_seed(1000)
    new.assimilate(HMM, xx, yy)
    times = np.append(times,datetime.now())
    
    error = old.stats.E.a[-1] - new.stats.E.a[-1]
    value = old.stats.E.a[-1] + new.stats.E.a[-1]
    rtol = np.sqrt(2*np.mean(error**2) / np.mean(value**2))
    atol = np.max(np.abs(error))
    trat = (times[-1]-times[-2])/(times[-2]-times[-3])
    
    print('EnRTS')
    print('rel RMSE, abs E, rel dt:',rtol,atol,trat)
    

#test_enkf()
test_smoother()
    