""" 
Model for along channel flow. 
"""
import dataclasses
import numpy as np
from abc import ABC, abstractmethod
from dapper.mods import *
from scipy.interpolate.tests.test_ndgriddata import TestNearestNDInterpolator

def states2array(states):
    return np.array([state.to_vector() for state in states])

def array2states(array, times=None):
    array = np.array(array)
    if not hasattr(times,'__iter__'):
        times = [times for _ in np.size(array,0)]
    states = [State().from_vector(x,time) for x,time in zip(array,times)]
    return states
    

#Dynamical model.
def step(x,t,dt):
    """ Forward in time. """ 
    
class Geometry:
    """ Model grid. """
    
    def __init__(self, shape, dx, dy):
        #Number of grid cells
        self.shape = np.array(shape, dtype=int)
        
        #Size of grid cells
        self.dx = np.ones(shape) * dx 
        self.dy = np.ones(shape) * dy
        self.dA = self.dx*self.dy
        
        #Position grid cell centers. 
        self.x = np.cumsum(self.dx, 1) - .5*dx 
        self.y = np.cumsum(self.dy, 0) - .5*dy 
        
        #Total grid size
        self.L = (np.sum(self.dx[0,:]),np.sum(self.dy[:,0]))
        
class InitialState:
    
    def sample(self, N):
        x = [State().initial(member=n).to_vector() for n in range(N)]
        return np.array(x)
        
class State:
    """ Values in grid cells. """
    
    def __init__(self, geometry):
        self.geometry = geometry 
        self.time   = None
        self.member = 0
        
        self.fields = {}
        for field in ['tracer','u','v']:
            self.fields[field] = np.zeros(self.geometry.shape)
            
    @property 
    def M(self):
        return sum([np.size(field) for field in self.fields.values()])
        
    def reset(self):
        self.time = None
        for field in self.fields:
            self.fields[field] *= 0.0
        return self
        
    def initial(self, member=0):
        self.reset()
        self.member = member
        self.time = 0.0 
        self.fields['tracer'] += 1.0
        self.fields['u'] += 1.0
        return self
        
    def to_vector(self):
        vector = np.array([],dtype=float)
        for field in self.fields.values():
            vector = np.append(vector, field.reshape((-1,)))
        return vector 
    
    def from_vector(self, vector, time=None, member=0):
        self.member = member
        for key,field in self.fields.items():
            self.fields[key] = vector[:np.size(field)].reshape(np.shape(field))
            vector = vector[np.size(field):]
        
        if time is not None:
            self.time = time
            
        return self
            
    def __add__(self, other):
        for key in self.fields:
            self.fields[key] += other.fields[key] 
        return self
    
    def __sub__(self, other):
        for key in self.fields:
            self.fields[key] -= other.fields[key] 
        return self
    
    def __mul__(self, other):
        for key in self.fields:
            if isinstance(other, State):
                self.fields[key] *= other.fields[key]
            else:
                self.fields[key] *= other 
            
        return self
            
class Flux:
    """Class representing fluxes in model."""
    
    def __init__(self, geometry):
        self.geometry = geometry 
        self.left, self.right = State(self.geometry), State(self.geometry)
        self.top, self.bottom = State(self.geometry), State(self.geometry)
        self.center = State(self.geometry)
    
    @abstractmethod 
    def update(self, state):
        pass
            
class UpwindFlux(Flux):
    
    def __init__(self, geometry):
        super().__init__(geometry)
        
    def update(self, state):
        #Velocity at interfaces
        C = state.fields['tracer']
        U = state.fields['u'] * self.geometry.dy 
        V = state.fields['v'] * self.geometry.dx 
        
        U = .5*U[:,1:]+.5*U[:,:-1]
        V = .5*V[1:,:]+.5*V[:-1,:]

        self.left.fields['tracer'][:,1:] = np.where(U>0.0, C[:,:-1], C[:,1:]) * U 
        self.right.fields['tracer'][:,:-1] = np.where(U>0.0, C[:,:-1], C[:,1:]) * U 
        self.bottom.fields['tracer'][1:,:] = np.where(V>0.0, C[:-1,:], C[1:,:]) * V 
        self.top.fields['tracer'][:-1,:] = np.where(V>0.0, C[:-1,:], C[1:,:]) * V
        
class SurfaceFlux(Flux):
    
    def __init__(self, geometry, functions):
        super().__init__(geometry)
        self.functions = functions
        
    def update(self, state):
        n = np.mod(state.member, len(self.functions))
        self.center.fields['u'] = self.geometry.dA * self.functions[n](state.time,self.geometry.x,self.geometry.y)
        
class BoundaryFlux(Flux):
    
    def __init__(self, geometry, functions):
        super().__init__(geometry)
        self.functions = functions
        
    def update(self, state):
        n  = np.mod(state.member, len(self.functions))
        C0 = state.fields['tracer']
        
        C  = self.functions[n](state.time, self.geometry.x[:,0], self.geometry.y[:,0])
        U  = state.fields['u'][:,0] * self.geometry.dy[:,0]
        self.left.fields['tracer'][:,0] = np.where(U>0.0, C, C0[:,0]) * U 
        
        C  = self.functions[n](state.time, self.geometry.x[:,-1], self.geometry.y[:,-1])
        U  = state.fields['u'][:,-1] * self.geometry.dy[:,-1]
        self.right.fields['tracer'][:,-1] = np.where(U<0.0, C, C0[:,-1]) * U
        
        C  = self.functions[n](state.time, self.geometry.x[0,:], self.geometry.y[0,:])
        U  = state.fields['v'][0,:] * self.geometry.dx[0,:]
        self.bottom.fields['tracer'][0,:] = np.where(U>0.0, C, C0[0,:]) * U
        
        C  = self.functions[n](state.time, self.geometry.x[-1,:], self.geometry.y[-1,:])
        U  = state.fields['v'][-1,:] * self.geometry.dx[-1,:]
        self.top.fields['tracer'][-1,:] = np.where(U<0.0, C, C0[-1,:]) * U
    
class ChannelModel: 
    """ Advection of traces with a given along-channel flow. """
    
    def __init__(self, geometry, fluxes=[]):
        self.geometry = geometry
        self.states = [State(self.geometry) for _ in range(1)]
        self.dxdt = [State(self.geometry) for _ in range(1)]
        self.fluxes = [UpwindFlux(self.geometry)] + fluxes
        
    def step(self, x, t, dt):
        ndim = np.ndim(x)
        if ndim==1:
            x = x[None,...]
            
        for n,x1 in enumerate(x):
            self.states[0].from_vector(x[n],t,n)
            for state, dxdt in zip(self.states, self.dxdt):
                self.update_dxdt(state, dxdt)
            self.euler_forward(dt)
            x[n] = self.states[0].to_vector()
            
        if ndim==1:
            x = x[0]
            
        return x
        
    def update_dxdt(self, state, dxdt):
        dxdt.reset()
        for flux in self.fluxes:
            flux.update(state)
            dxdt += flux.left 
            dxdt -= flux.right 
            dxdt += flux.bottom 
            dxdt -= flux.top 
            dxdt += flux.center
        dxdt *= (1/self.geometry.dA)
            
    def euler_forward(self, dt):
        self.states[0] += self.dxdt[0] * dt 
        
    @property 
    def M(self):
        return self.states[0].M
        
class PointObserver:
    
    def __init__(self, geometry, coord_function, sigo):
        self.geometry = geometry 
        self.grid = [(x1,y1) for x1,y1 in zip(self.geometry.x.ravel(),self.geometry.y.ravel())]
        self.coord_function = coord_function
        self.sigo = sigo 

        
    def interp(self, x):
        ndim = np.ndim(x)
        if ndim==1:
            x = x[None,...]
            
        y = np.array([self.interp1(x1) for x1 in x])
        
        if ndim==1:
            y = y[0]
            
        return y 
        
    def interp1(self, x):
        from scipy.interpolate import NearestNDInterpolator
        
        y = np.array([])
        state = State(self.geometry).from_vector(x)
        for key,value in self.coords.items():
            I = NearestNDInterpolator(self.grid, state.fields[key].ravel())
            y = np.append(y, I(value))
            
        return y 
        
    def __call__(self, t):
        self.coords = self.coord_function(t)
        M = sum([len(field) for field in self.coords.values()])
        Obs = {'M':M,
               'model': self.interp,
               'linear': self.interp,
               'noise': GaussRV(C=self.sigo**2, mu=0, M=M)
               }
        return Operator(**Obs)
    
    