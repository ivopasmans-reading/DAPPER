"""
This model at each time step rotates a point in the complex plain. Good for testing 
simple systems with multimodal distributions.
"""

import numpy as np
import scipy
import dapper.mods as modelling
import dapper.tools as tools
import pandas as pd
from dapper.tools.seeding import set_seed

#complex number 
I = complex(0,1)

#Rate that influences rotation
rate = 0.1

def rotate2d(theta, data, axis=-1):
    theta = np.arcsin(theta)
    x = np.take(data, 0, axis=axis)
    y = np.take(data, 1, axis=axis)
    xrot = np.cos(theta) * x - np.sin(theta) * y
    yrot = np.sin(theta) * x + np.cos(theta) * y 
    return np.stack((xrot,yrot), axis=axis)
                    

#Dynamical model.
def step_factory(amplitude=0.0, period=50):
    omega = 2*np.pi/period
    dr = lambda t : amplitude*omega*np.cos(omega*t)
    
    def step(x,t,dt):
        """ Step function if Cartesian coordinates are used. """
        shape = np.shape(x)
        x = np.reshape(x,(-1,2))
        #Convert
        polar = cartesian2polar(x)
        #Scale
        scale = 1.0 + dr(t) * dt / polar[:,0]
        #Rotate
        dtheta = rate * polar[:,1]
        for n, dtheta1 in enumerate(dtheta):
            x[n,:] =  rotation(dtheta1) @ (x[n,:] * scale[n])
   
        #Output
        return np.reshape(x, shape)
    
    return step 

#Initial conditions
class PolarRotation(tools.randvars.RV):
    """
    Initial with uniform rotation when polar coordinates are used. 
    """
    
    def __init__(self, radius=1, stats=None, seed=1000):
        self.seed = seed
        self.radius = radius
        if stats is None:
            self.stats = scipy.uniform(-np.pi,np.pi)
        else:
            self.stats = stats

    def sample(self, N):
        samples = np.ones((N,2)) * self.radius #radius
        samples[:,1] = self.stats.rvs(size=N, random_state=self.seed) #angles
        return samples
    
class Rotation(tools.randvars.RV):
    """
    Initial with uniform rotation when Cartesian coordinates are used.  
    """
    
    def __init__(self, radius=1, stats=None, seed=1000):
        self.seed = seed
        self.radius = radius
        if stats is None:
            self.stats = scipy.uniform(-np.pi,np.pi)
        else:
            self.stats = stats

    def sample(self, N):
        samples = np.empty((N,2)) 
        theta = self.stats.rvs(size=N, random_state=self.seed) #angles
        samples[:,0] = self.radius * np.cos(theta)
        samples[:,1] = self.radius * np.sin(theta)
        return samples

#Initial conditions
X0 = Rotation(radius=1, stats=scipy.stats.uniform(-.1*np.pi,.1*np.pi))
 
#Observation real part
def create_obs_factory(ind, sig, distribution):  
    M = len(ind)
    
    def create_obs(ko):
        """ Create time-dependent observation operator. """
        
        def sample(E):
            if np.ndim(E)==1:
                return np.reshape(E[ind], (M,))
            else:
                return np.reshape(E[:,ind], (-1,M))
        
        if distribution=='normal':
            C = sig**2 * np.ones((M,))
            noise = tools.randvars.GaussRV(mu=0,C=C,M=M)
        elif distribution=='beta':
            noise = tools.randvars.RV_beta(sig**2, lbounds=-1, ubounds=1, M=M)
        
        Obs = {'M':M, 'model':sample, 'linear':sample,
               'noise':noise}
    
        return modelling.Operator(**Obs)
    
    return create_obs

def create_obs_xy(ko):
    """ Create time-dependent observation operator. """
    
    def polar_sample(E):
        if np.ndim(E)==1:
            return np.reshape(E[0] * np.cos(E[1]), (1,))
        else:
            return np.reshape(E[:,0] * np.cos(E[:,1]), (-1,1))
        
    def sample(E):
        if np.ndim(E)==1:
            return np.reshape(E, (2,))
        else:
            return np.reshape(E, (-1,2))
        
    Obs = {'M':2, 'model':sample, 'linear':sample,
           'noise':tools.randvars.GaussRV(mu=0,C=0.1,M=1)}
    
    return modelling.Operator(**Obs)


def cartesian2polar(coord, axis=-1):
    """ Convert Cartesian to polar coordinates. """
    
    shape = np.shape(coord)
    if np.size(coord, axis)!=2 or np.ndim(coord)==0:
        raise ValueError("Cartesian coordinates must be 2D.")
    elif np.ndim(coord)==1:
        coord = np.reshape(coord, (1,-1))
    
    #Cartesian coordinates
    x = np.take(coord, 0, axis)
    y = np.take(coord, 1, axis)
    
    #Polar coordinates
    logx = np.log(x + I * y)
    r = np.exp(np.real(logx))
    theta = np.mod(np.imag( logx ), 2*np.pi)
    
    return np.reshape( np.stack((r,theta), axis=axis), shape)

def polar2cartesian(coord, axis=-1):
    """ Convert Cartesian to polar coordinates. """
    
    shape = np.shape(coord)
    if np.size(coord, axis)!=2 or np.ndim(coord)==0:
        raise ValueError("Cartesian coordinates must be 2D.")
    elif np.ndim(coord)==1:
        coord = np.reshape(coord, (1,-1))
    
    #Polar coordinates
    r = np.take(coord, 0, axis)
    theta = np.take(coord, 1, axis)
    
    #Polar coordinates
    x = r * np.cos(theta) 
    y = r * np.sin(theta)
    
    return np.reshape( np.stack((r,theta), axis=axis), shape)

def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def data2pandas(xx):
    data = {'x':xx[:,0], 'y':xx[:,1]}
    xx = cartesian2polar(xx)
    data = {**data, 'radius':xx[:,0], 'angle':np.rad2deg(xx[:,1])}
    return pd.DataFrame(data=data, columns=('x','y','radius','angle'))
    
    
        
    
