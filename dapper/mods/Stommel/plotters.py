#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots from ensemble.

@author: ivo
"""
import os
import numpy as np
import dapper.mods.Stommel as stommel
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap, cm
EARTH_RADIUS = 6375e3

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 12

#Directory to save files. 
FIG_DIR = stommel.fig_dir+'/climate3'

def tt2year(t):
    """ Convert model time to absolute year. """
    return t/stommel.year + 2004

class Alphabet:
    """ Cycles through alphabet to generate plot labels. """
    
    def __iter__(self):
        self.index = 96
        return self

    def __next__(self):
        self.index += 1
        return chr(self.index)
    

class BasePlot(ABC):
    """ 
    Base class to draw all plots. 
    """
    
    def create_figure(self, nrow, ncol):
        fig_size = (4*ncol, 3*nrow)
        self.fig, self.axes= plt.subplots(nrow, ncol, figsize=fig_size)
        self.axes = np.array(self.axes).reshape(nrow, ncol)
    
    def save(self, filename):
        self.fig.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    
    def add_end_da(self, ax):
        if not hasattr(self, 'timeso') or self.timeso is None: 
            return 
        
        #End DA 
        ylim = ax.get_ylim()
        ax.plot(np.max(tt2year(self.timeso))*np.array([1,1]),
                ax.get_ylim(),'k--')
        ax.set_ylim(ylim)
        
        ylim = ax.get_ylim()
        ax.plot(np.max(tt2year(self.timeso))*np.array([1,1]),
                ax.get_ylim(),'k--')
        ax.set_ylim(ylim)

    def add_subplot_labels(self, loc=(0.01, 1.0)):
        for ax, letter in zip(self.axes.ravel(), iter(Alphabet())):
            ax.annotate(letter+')', loc, xycoords='axes fraction',
                        ha='left',va='top')
            
    def add_subplot_labels2d(self, loc=(0.01, 1.0)):
        for ax, letter in zip(self.axes, iter(Alphabet())):
            for ax1,numeral in zip(ax,['I','II','III','IV','V','VI','VII']):
                ax1.annotate(letter+'.'+numeral+')', loc, xycoords='axes fraction',
                            ha='left',va='top')

class DiffPhase(BasePlot):
    """ 
    Create plots with temp differences, salt differences and phase portrait. 
    """
    
    def __init__(self, data):
        self.datas = data
        
    def load_data(self, data):
        self.HMM = data['HMM']
        self.model = data['model']
        self.times = self.HMM.tseq.times
        
        E = data['Efor']
        #Most likely value
        self.mode = stommel.ens_modus(E)
        #members, times, variables
        self.E = E.transpose((1,0,2))
        
        if 'xx' in data:
            self.truth = data['xx']
        else:
            self.truth = None
            
        if 'yy' in data and len(data['yy'])>0:
            self.yy = data['yy']
            self.R = self.HMM.Obs.noise.C.diag
            self.timeso = self.times[self.HMM.tseq.kko]
        else:
            self.yy, self.R, self.timeso = None, None, None 
            
    def plot_trajectory(self, axes, xx, **kwargs):
        times = tt2year(self.times)        
        
        states = stommel.array2states(xx, self.times)
        TH = np.array([s.regime=='TH' for s in states], dtype=bool)
        temp = np.reshape(np.diff([s.temp[0] for s in states],axis=1), (-1))
        salt = np.reshape(np.diff([s.salt[0] for s in states],axis=1), (-1))
        
        #Rescale the salt and heat
        temp_divisors = np.array([self.model.temp_scale(state) for state in states])
        salt_divisors = np.array([self.model.salt_scale(state) for state in states])
        scaled_temp = np.divide(temp ,temp_divisors)
        scaled_salt = np.divide(salt ,salt_divisors)
        
        #TH
        mask = np.where(~TH, np.nan, 1.)
        plot_params = {'color':'b','linestyle':'-','linewidth':2,'alpha':0.7}
        plot_params = {**plot_params, **kwargs}
        axes[0].plot(times, temp * mask, **plot_params)
        axes[1].plot(times, salt * mask, **plot_params)
        axes[2].plot(scaled_salt * mask, scaled_temp * mask, **plot_params)
        
        #SA
        mask = np.where(TH, np.nan, 1.)
        plot_params = {'color':'r','linestyle':'-','linewidth':2,'alpha':0.7}
        plot_params = {**plot_params, **kwargs}
        axes[0].plot(times, temp * mask, **plot_params)
        axes[1].plot(times, salt * mask, **plot_params)
        axes[2].plot(scaled_salt * mask, scaled_temp * mask, **plot_params)
        
        #accentuate initial conditions in phase portrait
        color = kwargs['color'] if 'color' in kwargs else 'green'
        axes[2].plot(scaled_salt[0], scaled_temp[0], 'o', color=color,
                     markersize=2)
        
        self.add_end_da(axes[0])
        self.add_end_da(axes[1])
        
    def plot_obs(self, ax, yy, R):
        times = tt2year(self.timeso) 
        ax.errorbar(times, np.squeeze(yy), np.sqrt(R), color='yellow', alpha=.085)
        ax.plot(times, np.squeeze(yy), color='orange', markersize=3, marker='o',
                markerfacecolor='orange', linestyle='None')
        
    def create_plot(self, nrows, lims):
        self.create_figure(nrows, 3)
        fig_size = (10,nrows*4.+.5)
        self.fig.set_size_inches(fig_size)
        self.fig.subplots_adjust(left=.1, right=.98, wspace=.4,
                                 top=.98, bottom=.5/fig_size[1])

        #Axes layout
        for ax in self.axes.ravel():
            ax.grid()
        for ax in self.axes[-1,:-1]:
            ax.set_xlabel('Time [year]')
            ax.set_xlim(np.min(tt2year(self.times)), np.max(tt2year(self.times)))
        for ax in self.axes[:,0]:
            ax.set_ylabel('Temperature\n difference [C]')
            ax.set_ylim(lims[0])
        for ax in self.axes[:,1]:
            ax.set_ylabel('Salinity\n difference [ppt]')
            ax.set_ylim(lims[1])
        for ax in self.axes[:,2]:
            ax.set_ylabel('Dimensionless\n temperature')
        for ax in self.axes[:,:2].ravel():
            ax.set_xlim(np.min(tt2year(self.times)),np.max(tt2year(self.times)))
        self.axes[-1,-1].set_xlabel('Dimensionless salinity')  
        
        self.add_subplot_labels(loc=(.01,.88))
        
    def plot(self, filename=None, lims=[(2,8),(.1,1)]):
        self.load_data(self.datas[0])
        self.create_plot(nrows=len(self.datas),lims=lims)
        for axes,data in zip(self.axes, self.datas):
            self.load_data(data)
            self.plot1(axes)    
            
        #Save
        if filename is not None:
            self.save(filename)
            
        
    def plot1(self, axes):        
        #Plot individual trajectories. 
        for e in self.E:
            self.plot_trajectory(axes, e)
            
        #Plot truth 
        if self.truth is not None:
            self.plot_trajectory(axes, self.truth, alpha=1.0, color='k')
            
        #Plot best-estimate 
        if self.mode is not None:
            self.plot_trajectory(axes, self.mode, alpha=1.0, color="#FFA500")
            
        #observations
        if self.yy is not None:
            self.plot_obs(axes[0], np.diff(self.yy[:,:2], axis=1),
                          np.sum(self.R[:2])) 
            self.plot_obs(axes[1], np.diff(self.yy[:,2:], axis=1),
                          np.sum(self.R[2:])) 
            
        #Plot T=S line for reference
        axes[2].set_aspect(1)
        axes[2].set_xlim(0,11)
        axes[2].set_ylim(0,11)
        axlim = axes[2].get_xlim()
        axes[2].plot(np.linspace(0,axlim),np.linspace(0,axlim), 
                        color="m", linestyle=':')
        axes[2].annotate('TH', (7.,8.5))
        axes[2].annotate('SA', (8.,6.5))
        
        self.add_end_da(axes[0])
        self.add_end_da(axes[1])
        
    def plot_flip(self, filename=None, lims=[(2,11),(0,3)]):
        self.load_data(self.datas[0])
        self.create_plot(nrows=len(self.datas),lims=lims)
        for axes,data in zip(self.axes, self.datas):
            self.load_data(data)
            self.plot1_flip(axes)    
            
        #Save
        if filename is not None:
            self.save(filename)
            
        
    def plot1_flip(self, axes):        
        #Plot individual trajectories. 
        for e in self.E:
            self.plot_trajectory(axes, e)
            
        #Plot truth 
        if self.truth is not None:
            self.plot_trajectory(axes, self.truth, alpha=1.0, color='k')
            
        #Plot best-estimate 
        if self.mode is not None:
            self.plot_trajectory(axes, self.mode, alpha=1.0, color="#FFA500")
            
        #observations
        if self.yy is not None:
            self.plot_obs(axes[0], np.diff(self.yy[:,:2], axis=1),
                          np.sum(self.R[:2])) 
            self.plot_obs(axes[1], np.diff(self.yy[:,2:], axis=1),
                          np.sum(self.R[2:])) 
            
        #Plot T=S line for reference
        axes[2].set_aspect(1)
        axes[2].set_xlim(0,20)
        axes[2].set_ylim(0,20)
        axlim = axes[2].get_xlim()
        axes[2].plot(np.linspace(0,axlim),np.linspace(0,axlim), 
                        color="m", linestyle=':')
        axes[2].annotate('TH', (17.,18.5))
        axes[2].annotate('SA', (18.,16.5))
        
        self.add_end_da(axes[0])
        self.add_end_da(axes[1])
        
        
    
class EtaPlot(BasePlot):
    """ Plot etas used in the non-dimensional equations. """
    
    def __init__(self, HMM, model, **kwargs):
        self.HMM = HMM
        self.model = model
        self.times = HMM.tseq.times 
        self.timeso = self.times[HMM.tseq.kko]
        
        if 'Efor' in kwargs:
            self.add_ensemble(kwargs['Efor'])
    
    def add_ensemble(self, E):
        #Most likely value
        self.mode = stommel.ens_modus(E)
        #members, times, variables
        self.E = E.transpose((1,0,2))

    def create_plot(self):
        plt.close('all')
        self.create_figure(1, 1)
        self.fig.subplots_adjust(bottom=.16)
        self.ax = self.axes[0,0]
        
        #Axes layout
        self.ax.grid()
        self.ax.set_xlabel('Time [year]')
        times = tt2year(self.times)
        self.ax.set_xlim(np.min(times), np.max(times))
        
        
    def plot(self, filename=None):
        self.create_plot()
        times = tt2year(self.times)
        
        #Plot eta1 
        states = stommel.array2states(self.mode, self.times)
        etas = np.array([self.model.eta1(state) for state in states])
        print('ETA1',etas[0],etas[-1])
        self.ax.plot(times, etas, '-', label=r'$\eta_{1}$', linewidth=.8)
        
        #Plot eta2
        states = stommel.array2states(self.mode, self.times)
        etas = np.array([self.model.eta2(state) for state in states])
        print('ETA2',etas[0],etas[-1])
        self.ax.plot(times, etas, '-', label=r'$\eta_{2}$', linewidth=.8)
        
        #Plot eta3 
        states = stommel.array2states(self.mode, self.times)
        etas = np.array([self.model.eta3(state) for state in states])
        print('ETA3',etas[0],etas[-1])
        self.ax.plot(times, etas, '-', label=r'$\eta_{3}$', linewidth=.8)
        
        #Add marker end DA
        self.ax.set_ylim(0,18)
        self.add_end_da(self.ax)
        
        #Add legend 
        self.ax.legend(loc='upper right', framealpha=1., ncol=2)
        
        #Save
        if filename is not None:
            self.save(filename)
            
class ParameterPlot(BasePlot):
    """ Plot model parameters as function of time. """
    
    def __init__(self, HMM, model, **kwargs):
        self.HMM = HMM
        self.model = model
        self.times = HMM.tseq.times 
        
        if 'Efor' in kwargs:
            self.add_ensemble(kwargs['Efor'])
        if 'xx' in kwargs:
            self.add_truth(kwargs['xx'])
            
    def add_ensemble(self, E):
        #Most likely value
        self.mode = stommel.ens_modus(E)
        self.mode = self.mode.transpose()
        #members, times, variables
        self.E = E.transpose((2,1,0))
        
    def add_truth(self, xx):
        self.xx = xx.T
        
    def create_plot(self):
        plt.close('all')
        self.create_figure(1, 3)
        self.fig.subplots_adjust(bottom=.18, wspace=.3)
        self.axes = np.ravel(self.axes)
        
        self.axes[0].set_ylabel('Surface temperature\n flux coefficient [ms-1]')
        self.axes[1].set_ylabel('Surface salinity\n flux coefficient [ms-1]')
        self.axes[2].set_ylabel('Advection flux coefficient [ms-1]')
        
        #Axes layout
        times = tt2year(self.times) 
        times = times[:len(stommel.hadley)]
        for ax in self.axes:
            ax.grid()
            ax.set_xlabel('Time [year]')
            ax.set_xlim(2004,2022)
            
        self.add_subplot_labels()
            
    def plot_series(self, ax, xx, **kwargs):
        plot_params = {**{'color':'b','linewidth':2}, **kwargs}
        
        times = tt2year(self.times) 
        ax.plot(times, np.exp(xx), **plot_params)
        
        
            
    def plot(self, filename=None):
        self.create_plot()
        
        for ax, params, mode in zip(self.axes, self.E[4:], self.mode[4:]):
            for series in params:
                self.plot_series(ax, series)
            self.plot_series(ax, mode, color='r')
            
        if hasattr(self, 'xx'):
            for ax, x in zip(self.axes, self.xx[4:]):
                self.plot_series(ax, x, color='g', linewidth=2)
            
        #Use 0 as lower bound
        for ax in self.axes:
            ax.set_ylim(np.maximum(ax.get_ylim(),0.0))
        
        #Save
        if filename is not None:
            self.save(filename)
    
class TransportPlot(BasePlot):
    """ Plot transport as function of time. """
    
    def __init__(self, HMM, model, **kwargs):
        self.HMM = HMM
        self.model = model
        self.times = HMM.tseq.times 
        self.timeso = self.times[HMM.tseq.kko]
        
        if 'Efor' in kwargs:
            self.add_ensemble(kwargs['Efor'])
            
    def add_ensemble(self, E):
        #Most likely value
        self.mode = stommel.ens_modus(E)
        
    def create_plot(self):
        plt.close('all')
        self.create_figure(1, 1)
        self.fig.subplots_adjust(bottom=.18, left=.16)
        self.ax = self.axes[0,0]
        
        #Axes layout
        self.ax.grid()
        
        self.ax.set_xlabel('Time [year]')
        self.ax.set_xlabel('Time [year]')
        self.ax.set_ylabel('Transport [Sv]')
        
    def plot(self, filename=None):
        self.create_plot()
        times = tt2year(self.times)
        
        states = stommel.array2states(self.mode)
        trans = np.array([self.model.fluxes[0].transport(
            s)*np.mean(self.model.dx*self.model.dz) for s in states]).flatten()
        
        self.ax.plot(times, 18*np.ones_like(times), 'k--')
        self.ax.plot(times, trans*1e-6)
        self.ax.set_ylim(0,18.5)
        self.ax.set_xlim(np.min(times), np.max(times))
        self.add_end_da(self.ax)
        
        #Save
        if filename is not None:
            self.save(filename)
            
class RmseStdPlot(BasePlot):
    """ Plot plots with RMSEs and ensemble std. dev. as function of time."""
    
    def __init__(self, datas):
        self.HMM = datas[0]['HMM']
        self.model = datas[0]['model']
        self.times = self.HMM.tseq.times
        self.datas = datas
        
        for data in self.datas:
            if len(data['HMM'].tseq.kko)>0:
                self.timeso = self.times[data['HMM'].tseq.kko]
            
    def create_plot(self):
        self.create_figure(7, len(self.datas))
        self.fig.set_size_inches((10,14))
        self.fig.set_size_inches(10, 18)
        self.fig.subplots_adjust(left=.1, right=.98, bottom=.08, wspace=.1)
        
        times = tt2year(self.times)
        for ax in self.axes.ravel():
            ax.grid()
            ax.set_xlim(np.min(times), np.max(times))
        for ax in self.axes[:-1,:].ravel():
            ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        for ax in self.axes[-1,:]:
            ax.set_xlabel('Time [year]')
        
        self.vars = [r'$T_{pole}$ [C]', r'$T_{eq}$ [C]',
                     r'$S_{pole}$ [ppt]', r'$S_{eq}$ [ppt]',
                     r'$log \kappa_{T}$', r'$log \kappa_{S}$ ',r'$log \gamma$']
        self.ylims = [(0,1),(0,1),(0,.1),(0,.1),(0,1-5),(0,1e-6),(0,1)]
        for ax, label, lim in zip(self.axes[:,0], self.vars, self.ylims):
            ax.set_ylabel(label)
        
        for ax in self.axes[0:2,:].ravel():
            ax.set_ylim(0,1.4)
        for ax in self.axes[2:4,:].ravel():
            ax.set_ylim(0,.3)
        for ax in self.axes[4:,:].ravel():
             ax.set_ylim(0,.4)
        for ax in self.axes[:,1:].ravel():
            ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
            
        self.add_subplot_labels2d()
        
    def plot(self, filename=None):
        self.create_plot()
        times = tt2year(self.times)
        
        for axes, data in zip(self.axes.T, self.datas):
            axes[0].set_title(data['label'])
        
            #Calculate ensemble standard deviation
            E = data['Efor'] + 0.0
            #E[:,:,4:] = np.exp(E[:,:,4:])
            stds = np.std(E, axis=1, ddof=1) 
            
            #Calculate forecast
            modes = stommel.ens_modus(data['Efor']) + 0.0
            #modes[:,4:] = np.exp(modes[:,4:])
            
            #Calculate error. 
            truth = data['xx'] + 0.0
            #truth[:,4:] = np.exp(truth[:,4:])
            errors = modes - truth 
            
            for ax, error, std in zip(axes, errors.T, stds.T):
                ax.plot(times[::6], np.abs(error[::6]),'k-')
                ax.plot(times[::6], std[::6], 'b+')
        
        for ax in self.axes.ravel():
            self.add_end_da(ax)        

        #Save
        if filename is not None:
            self.save(filename)
    
        
class TippingPlot(BasePlot):
    """ Plot plot with the probability of a flip under different climas."""
    
    def __init__(self, data):
        self.data = np.array([(*clima,prob) for clima,prob in 
                              zip(data['climas'],data['probabilities'])])
    
    def plot(self, filename=None):
        self.create_figure(1,1)
        self.fig.subplots_adjust(bottom=.22,left=.18)
        
        #Reshape
        S0 = len(np.unique(self.data[:,0]))
        T_melts = np.reshape(self.data[:,0], (S0,-1)) / stommel.year
        A_polars = np.reshape(self.data[:,1], (S0,-1)) * 1e-2
        probs = np.reshape(self.data[:,2], (S0,-1)) * 1e2
        print(probs)
        
        ax=self.axes[0,0]
        handle = ax.pcolormesh(T_melts, A_polars, probs, shading='gouraud',
                               cmap='bwr',vmin=0,vmax=100)
        ax.set_xlabel('Melt period [year]')
        ax.set_ylabel('Annual temperature rate [C]')
        ax.set_xticks(np.unique(T_melts)[::2])
        ax.set_yticks(np.unique(A_polars)[::2])
        
        self.cbar = plt.colorbar(handle, orientation='vertical',
                                 label='Percent of ensemble flipping',
                                 ticks=np.arange(0,101,20))
        
        #Save
        if filename is not None:
            self.save(filename)
        
class BoxesPlot(BasePlot):
    """ Plot the clusters on a world map. """

    def __init__(self, clusters, en4):
        self.clusters = clusters 
        self.lon, self.lat = en4['lon'], en4['lat']
        
    def plot(self, filename=None):
        self.create_figure(1, 1)
        
        lat0, lon0 = 50, -50
        width = np.deg2rad(90)*np.cos(lat0)*EARTH_RADIUS
        height = np.deg2rad(90)*EARTH_RADIUS
        self.map = Basemap(projection='aeqd',lat_0=lat0, lon_0=lon0,
                           width=width, height=height)
        self.map.drawparallels(np.arange(0,90,10), 
                               labels = [True,True,False,False])
        self.map.drawmeridians(np.arange(-90,91,30),
                               labels = [False,False,False,True])
        self.map.drawcoastlines(linewidth=0.5)

        for indices, color in zip(self.clusters, 
                                  mpl.colormaps['tab10'](np.arange(0,10))):
            lon, lat = self.get_profiles_locs(indices)
            if len(lon)==0:
                continue
            
            x, y = self.map(lon, lat)
            self.map.scatter(x, y, marker='o', color=color,
                             s=np.ones_like(x)*.5)
            
        #Save
        if filename is not None:
            self.save(filename)
            
    def get_profiles_locs(self, indices):
        clusters = [ind[1:] for ind in indices if ind[0]==0]
        lats = [self.lat[ind[0]] for ind in clusters]
        lons = [self.lon[ind[1]] for ind in clusters]
        return lons, lats
    
            
