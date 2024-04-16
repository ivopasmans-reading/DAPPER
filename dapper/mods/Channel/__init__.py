"""
This model will advect a tracer. 
"""

import numpy as np
import scipy
import dapper.mods as modelling
import dapper.tools as tools
import pandas as pd
from dapper.tools.seeding import set_seed
import dataclasses

class Grid:
    
    def __init__(self, x, y):
        self.x, self.y = np.meshgrid(x, y)
        self.dx = np.diff(self.x, axis=1)
        self.dy = np.diff(self.y, axis=0)
        #self.dA = self.dx * self.dy
        self.shape = np.shape(self.dA)
        self.size = np.size(self.dA)
        
@dataclasses.dataclass 
class State:
    """Class containing all attributes that make up a physical state."""

    #Grid 
    grid: Grid 
    #Array 
    tracer: np.ndarray = np.array([0])
    #Ensemble member 
    member: int = 0
    #Time associated with state
    time: float = 0.0 
    
    def post_init(self):
        for key in self.keys:
            setattr(self, key, np.empty(self.grid.shape))
    
    @property 
    def keys(self):
        keys = set(dataclasses.asdict(self).keys())
        keys = keys - set(['member','time'])
        return keys 
    
    def to_vector(self):
        vector = [getattr(self, key).flatten() for key in self.keys]
        vector = np.ndarray(v, dtype=float).flatten()
        return vector 
    
    def from_vector(self, vector):
        vector = np.reshape(vector, (-1,) + self.grid.shape)
        for key, values in zip(self.keys, vector):
            setattr(self, key, values)
        
        
        
    
    
class Flux:
    """Class representing fluxes in model."""
    
    def __init__(self):
        self.ens_member = 0
    
    def left(self, state):
        """Return flux flowing into cell from left."""
        return State().zero()
    
    def right(self, state):
        """Return flux exiting cell to right."""
        return State().zero()
    
    def top(self, state):
        """Return flux exiting cell via top."""
        return State().zero()
    
    def bottom(self, state):
        """Return flux entering cell from bottom."""
        return State().zero()
    
class Model:
    
    def tendency(self, state):
        """Calculate tendency (d/dt) for the model."""
        
        #Empty tendency
        tendency = State().zero()
        
        #Convergence vertical fluxes
        for flux in self.fluxes:
            
            
            
            top, bottom = flux.top(state), flux.bottom(state)
            tendency.temp -= (top.temp - bottom.temp) / self.dz 
            tendency.salt -= (top.salt - bottom.salt) / self.dz
    
        #Cross-section cells 
        Aleft = self.dz[:,:-1] * self.dx[:,:-1] 
        Aright = self.dz[:,1:] * self.dx[:,1:] 
        
        
        #Convergence horizontal fluxes. 
        for flux in self.fluxes:
            left, right = flux.left(state), flux.right(state)
            tendency.temp -= (right.temp * Aright - left.temp * Aleft ) / self.V  
            tendency.salt -= (right.salt * Aright - left.salt * Aleft ) / self.V 
            
        return tendency