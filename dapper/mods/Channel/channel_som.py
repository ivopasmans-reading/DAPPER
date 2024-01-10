#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:41:12 2023

Channel flow 

@author: ivo
"""

import numpy as np
import dapper.mods.Channel as modelling
import dapper.da_methods as da 
import matplotlib.pyplot as plt
from dapper.tools.randvars import RV_with_mean_and_cov

dt = 10
N = 10

class Acceleration: 
    
    def __init__(self, dt, L, amplitudes, seed=1000):
        self.seed, self.L = seed, L
        self.A = np.array(amplitudes) 
        self.dt = dt 
        
    def __call__(self, t, x, y):
        x, y = np.array(x), np.array(y)
        s = np.shape(y)
        y = y.reshape((-1,1)) / self.L
        
        np.random.seed(int(t*13)+self.seed)
        A = (self.A * complex(1,0) * np.random.normal(size=np.shape(self.A)) +
             self.A * complex(0,1) * np.random.normal(size=np.shape(self.A)))
        A = np.sqrt(.5)*A.reshape((1,-1))
        
        k = np.pi / np.arange(1,len(self.A)+1)
        k = k.reshape((1,-1))
        
        F = A*np.exp(complex(0,1)*k*y)
        F = np.sum(np.real(F), axis=1)
        F = np.reshape(F, s) / np.sqrt(self.dt)
        
        return F 
    
def obs_coords(t):
    xy =[(x,y) for x in np.arange(1e3,20e3,1e3) for y in [1e3,3e3]]
    return {'tracer': np.array(xy)}
    
    
geometry = modelling.Geometry((20,100),200.,200.)
accelerations = [Acceleration(dt, geometry.L[1], .01/np.array([1,2,3]), 10*n) for n in range(N)]
boundary = modelling.BoundaryFlux(geometry, [lambda t,x,y: 2.0])
surface = modelling.SurfaceFlux(geometry, accelerations)
model = modelling.ChannelModel(geometry, [boundary, surface])

state = modelling.State(geometry)
states = [modelling.State(geometry).initial() for _ in range(N)]
x = modelling.states2array(states)
    
tseq = modelling.Chronology(dt, dko=10, T=8000)
Dyn = {'M': model.M,'model': model.step,'noise': 0}
Obs0 = modelling.PointObserver(geometry,obs_coords,.1)
Obs = {'time_dependent':Obs0}
X0=RV_with_mean_and_cov(M=state.M, mu=states[0].to_vector())
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

xx,yy=HMM.simulate()