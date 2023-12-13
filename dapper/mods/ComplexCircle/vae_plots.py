#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing plotting routines for VAE. 

@author: Ivo Pasmans
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
from dapper.mods import ComplexCircle as circle
from scipy.stats import norm
import os, shutil
from abc import ABC, abstractmethod

#Default settings for layout. 
mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'

#%% 

def ens2ml(E, p=0.5):
    """ 
    Estimate most-likely value for ensemble by thinning 
    ensemble members until the ensemble follows Gaussian 
    distribution and mean can be used to determined 
    most-likely estimate. 
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import shapiro
    
    D = squareform(pdist(E))
    d = np.unique(D[...])[::-1]
    
    for d1 in d:
        #Check if ensemble is Gaussian
        _, pvalue = shapiro(E)
        if pvalue>p:
            break
        
        #Eliminate the outlier. 
        n = np.sum(D<d1, axis=1)
        if np.all(n==np.min(n)):
            continue
        else:
            mask = n>np.min(n)     
            
        E = E[mask]
        D = D[mask,:]
        D = D[:,mask]
        
    mu = np.mean(E, axis=0)
    return mu
    

#%% Abstract classes for plotting. 

class StepGenerator(object):
    """ Returns possible tick distances in decreasing order. """
    
    def __init__(self, factor=1):
        self.steps = np.array([1,2,2.5,5], dtype=float) * factor
        self.n = len(self.steps)
    def __iter__(self):
        return self 
    def __next__(self):
        self.n = self.n - 1
        if self.n<0:
            self.steps *= 0.1
            self.n = len(self.steps) - 1
        return self.steps[self.n]

class BasePlots: 
    """ 
    Abstract class that is used as template for figures that generate 
    plots based on series of data. 
    
    Parameters
    ----------
    fig_dir : str 
        Directory to which figure will be saved. 
        
    """
    
    def __init__(self, fig_dir):
        self.fig_dir = fig_dir
        self.labels = []
    
    @staticmethod
    def adaptable_bins(x, alpha=0.05):
        """ 
        Create bins of different size with equal amount of samples
        per bin. 
        
        Parameters:
        x : 1D numpy array 
            Samples used to create bins. 
        alpha : float>0
            Value used to calculate number of bins. The closer to zero, the
            fewer bins. 
            
        """
        #Number of bins 
        n = len(x)
        k = int( 4*(2*n**2 / norm.isf(alpha))**.2 )
        nbin = int(n / k)+1
        #Sort in ascending error
        x = np.sort(x)
        #Create bins
        bins, i = np.array([x[0]]), nbin
        while i<n-1:
            bins = np.append(bins,.5*x[i-1]+.5*x[i])
            i = i + nbin
        bins  = np.append(bins, x[-1])
        bins += np.arange(len(bins)) * 1e-6 * max(1, x[-1]-x[0])
        
        return bins
    
    @staticmethod 
    def nice_ticks(lims, max_ticks=12, symmetric=False, include=[],
                   minlim=-np.inf, maxlim=np.inf):
        """ 
        Calculate nice locations for the ticks. 
        
        Parameters:
        -----------
        lims : 1D-array 
            1D numpy-array used to determine lower and upper limits. 
        max_ticks : int>0
            Maximum number of ticks. 
        symmetric : boolean
            Indicate whether ticks should be symmetric around zero. 
        include : list 
            Values that must lie within axis limits. 
            
        """
        
        lims = np.append(include, lims)
        lims = np.array([np.min(lims), np.max(lims)])  
        lims = np.maximum(minlim, lims)
        lims = np.minimum(maxlim, lims)
        if symmetric:
            lims = np.array([-1,1]) * np.max(np.abs(lims))
        
        dlim = np.log10( max(1e-6, np.diff(lims)[0]) )
        order, r = 10**np.floor(dlim), 10**np.mod(dlim, 1.)
        
        #Find right step
        for step in StepGenerator(order):
            nticks = np.ceil(r*order / step)
            if nticks > max_ticks:
                break 
            else:
                step0 = step
                
        lbound = np.floor(lims[0] / step0) * step0
        ubound = np.ceil(lims[-1] / step0) * step0
        ticks  = np.arange(lbound, ubound+step0, step0)
        
        return ticks
        
    @staticmethod
    def set_nice_xlim(ax, lims=None, **kwargs):
        """ 
        Use self.nice_ticks to set ticks and limits of x-axis. 
        
        Parameters:
        lims : (min,max)-tuple
            Limits x-axis. 
        kwargs : 
            See parameters self.nice_ticks
        
        """
        flatten = False
        if not hasattr(ax, '__iter__'):
            ax = [ax]
            flatten = True
        
        if lims is None:
            mm = np.array([ax1.get_xlim() for ax1 in ax])
            lims = [np.min(mm), np.max(mm)]
        ticks = BasePlots.nice_ticks(lims, **kwargs)
        
        for ax1 in ax:
            ax1.set_xticks(ticks)
            ax1.set_xlim(ticks[0], ticks[-1])
        
        if flatten:
            ax = ax[0]
        
        return ax
    
    @staticmethod
    def set_nice_ylim(ax, lims=None, **kwargs):
        """ Use nice_ticks to set ticks and limit of y-axis. 
        
        Parameters:
        lims : (min,max)-tuple
            Limits y-axis. 
        kwargs : 
            See parameter self.nice_ticks
        
        """
        flatten = False
        if not hasattr(ax, '__iter__'):
            ax = [ax]
            flatten = True
        
        if lims is None:
            mm = np.array([ax1.get_ylim() for ax1 in ax])
            lims = [np.min(mm), np.max(mm)]
        ticks = BasePlots.nice_ticks(lims, **kwargs)
        
        for ax1 in ax:
            ax1.set_yticks(ticks)
            ax1.set_ylim(ticks[0], ticks[-1])
        
        if flatten:
            ax = ax[0]
        
        return ax
   
    def styles(self):
        """ Generate 9 different combinations of color and linestyle. """
        from matplotlib import colors as mcolors
        colors = mcolors.TABLEAU_COLORS.keys()
        styles = ['-', (0,(1,1)), (5,(10,3)), 
                  (0,(3,1,1,1)), (0,(3,10,1,10,1,10)), (5,(10,3)),
                  (0,(3,1,1,1,1,1)), (0,(3,5,1,5,1,5)), (0,(5,5))]
        return zip(self.labels,colors,styles)
    
    @property 
    def fig_path(self):
        if self.fig_dir is None:
            return None 
        else:
            return os.path.join(self.fig_dir, self.fig_name)
    
    def save(self):
        """ Save figure to file path in self.fig_path. """
        if self.fig_dir is not None:
            if not os.path.exists(self.fig_dir):
                os.mkdir(self.fig_dir)
            self.fig.savefig(self.fig_path, dpi=400, format='png')
        
#%% 

class EnsStatsPlots(BasePlots):
    """ 
    Class that plots statistics of ensemble DA. 
    """
    
    def __init__(self, fig_dir):
        super().__init__(fig_dir)
        self.xps = []
        self.labels = []
        self.truth = []
        self.p_level = .9
        
    def rmse(self, a, **kwargs):
        """ Calculate root-mean square error."""
        return np.sqrt(np.mean(a**2, **kwargs))
        
    def add_xp(self, xp):
        """ Add experiment for processing. """
        self.labels.append(xp.name)
        self.xps.append(xp)
        self.xps[-1].stats.besta = np.array([ens2ml(E) for E in xp.stats.E.a])
        self.xps[-1].stats.bestf = np.array([ens2ml(E) for E in xp.stats.E.f])
        
    def add_truth(self, HMM, truth):
        """ Add truth for calculating errors. """ 
        self.HMM = HMM
        self.truth = truth
        
    def plot_rmse_std(self, fig_name='rmse'):
        """
        Plot RMSE and ensemble standard deviation. 
        """
        
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []
        
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            rmse = self.truth[times] - xp.stats.bestf
            rmse  = self.rmse(rmse, axis=1)
            h0, = self.axes[0,0].plot(times, rmse, label=style[0], color=style[1],
                                      linestyle=style[2])
            
            rmse = self.truth[times] - xp.stats.besta
            rmse  = self.rmse(rmse, axis=1)
            h0, = self.axes[0,1].plot(times, rmse, label=style[0], color=style[1],
                                      linestyle=style[2])
            
            E = xp.stats.E.f
            std = np.hypot(np.std(E[:,:,0], axis=1, ddof=1), 
                           np.std(E[:,:,1], axis=1, ddof=1))
            self.axes[1,0].plot(times, std, label=style[0], color=style[1],
                                linestyle=style[2])
            
            E = xp.stats.E.a
            std = np.hypot(np.std(E[:,:,0], axis=1, ddof=1), 
                           np.std(E[:,:,1], axis=1, ddof=1))
            self.axes[1,1].plot(times, std, label=style[0], color=style[1],
                                linestyle=style[2])
            
            self.handles.append(h0)
            
        self.axes[0,0].set_ylabel('forecast RMSE')
        self.axes[0,1].set_ylabel('analysis RMSE')
        self.axes[1,0].set_ylabel('forecast ensemble std. dev.')
        self.axes[1,1].set_ylabel('analysis ensemble std. dev.')
        self.axes[0,0].legend(handles=self.handles, loc='upper right')
        
        self.set_nice_ylim(self.axes[0], include=[0], minlim=0)
        self.set_nice_ylim(self.axes[1], include=[0], minlim=0)
        
        for ax in self.axes.flatten():
            self.set_nice_xlim(ax, (0,np.max(times)), max_ticks=10)
            ax.set_xlabel('Time')
            ax.grid()
            
    def plot_rmse_std_variables(self, fig_name='rmse_variables', mode='f'):
        """
        Plot RMSE and ensemble standard deviation. 
        """
        
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 4, figsize=(11,8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []
        
        for xp, style in zip(self.xps, self.styles()):
            print('XP ',xp.name)
            if mode=='f':
                E, best, mode_label = xp.stats.E.f, xp.stats.bestf, 'Forecast'
            elif mode=='a':
                E, best, mode_label = xp.stats.E.a, xp.stats.besta, 'Analysis'
                
            for ax,func in zip(self.axes.T, self.best_funcs()):
                times = xp.HMM.tseq.kko
                rmse  = func(self.truth[times]) - func(best)
                if ax[0]==self.axes[0,-1]:
                    rmse = np.mod(rmse+180.,360.)-180.
                
                rmse  = self.rmse(rmse.reshape((-1,1)), axis=1)
                h0, = ax[0].plot(times, rmse, label=style[0], color=style[1],
                                 linestyle=style[2])
            
                E1 = func(E)
                if ax[1]==self.axes[1,-1]:
                    j  = complex(0,1)
                    E1  = np.exp(j*np.deg2rad(E1))
                    mu = np.prod(E1, axis=1)**(1/np.size(E1,1))
                    E1  = E1 / mu.reshape((-1,1)) 
                    E1  = np.rad2deg(np.imag(np.log(E1)))
                std = np.std(E1, axis=1, ddof=1)
                ax[1].plot(times, std, label=style[0], color=style[1],
                           linestyle=style[2])
            
            self.handles.append(h0)
            
        self.axes[0,0].set_ylabel(mode_label+' RMSE')
        self.axes[1,0].set_ylabel(mode_label+' ensemble std. dev.')
        for ax,title in zip(self.axes[0],['x','y','radius','angle']):
            ax.set_title(title)

        for ax in self.axes[1]:
            ax.set_xlabel('Time')
            
        for ax in self.axes.flatten():
            self.set_nice_ylim(ax, include=[0], minlim=0)
            self.set_nice_xlim(ax, (0,np.max(times)), max_ticks=6)
            ax.grid()
            
        self.axes[0,0].legend(handles=self.handles, loc='upper right')
            
    def plot_best1(self, times, xy, style):     
        """ Plot best estimate. """
        for ax, func in zip(self.axes.flatten(), self.best_funcs()):
            h0, = ax.plot(times, func(xy), label=style[0], color=style[1],
                          linestyle=style[2])
        
        return h0
    
    def plot_best1_range(self, times, xy, E, style, p=.65):
        """ Plot range closest to to best estimate. """
        e = E - xy.reshape((np.size(E,0),1,np.size(E,-1)))
        e = np.linalg.norm(e, axis=-1)
        ind = np.argsort(e, axis=-1)

        Eind = []
        for E1,ind1 in zip(E, ind):
            Eind.append(E1[ind1[:int(p*len(ind1))],:])
        Eind = np.array(Eind)
        
        for ax, func in zip(self.axes.flatten()[:-1], self.best_funcs()):
            Eind1 = func(Eind)
            h0 = ax.fill_between(times, np.min(Eind1,axis=-1), np.max(Eind1,axis=-1),
                                 label=style[0], color=style[1], linestyle=style[2], alpha=.3)
        
        return h0
            
    def best_funcs(self):
        """ Subtract different variables from ensemble. """
        xfunc = lambda E: E[...,0]
        yfunc = lambda E: E[...,1]
        rfunc = lambda E: np.linalg.norm(E, axis=-1)
        tfunc = lambda E: np.mod(np.rad2deg(np.arctan2(yfunc(E), xfunc(E))), 360)
        return [xfunc, yfunc, rfunc, tfunc]
            
    def plot_best(self, fig_name='best'):
        """ Plot best estimates. """
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.95)
        self.handles = []
        
        times = self.HMM.tseq.kk
        h0 = self.plot_best1(times, self.truth[times], ('truth','k','-'))
        self.handles.append(h0)
        
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            _  = self.plot_best1_range(times, xp.stats.besta, xp.stats.E.a, style)
            h0 = self.plot_best1(times, xp.stats.besta, style)
            self.handles.append(h0)
            
        self.axes[0,0].legend(handles=self.handles, loc='upper right')
        self.axes[0,0].set_ylabel('x')
        self.set_nice_ylim(self.axes[0,0], symmetric=True)
        self.axes[0,1].set_ylabel('y')
        self.set_nice_ylim(self.axes[0,1], symmetric=True)
        self.axes[1,0].set_ylabel('radius')
        self.set_nice_ylim(self.axes[1,0], include=[0])
        self.axes[1,1].set_ylabel('angle')
        self.axes[1,1].set_yticks(np.arange(0,361,45))
        self.axes[1,1].set_ylim(0,360)
        
        for ax in self.axes[1]:
            ax.set_xlabel('Time')
        for ax in self.axes.flatten():
            self.set_nice_xlim(ax, lims=(0,np.max(times)), max_ticks=12)
            ax.grid()
            
    def plot_taylor(self, fig_name='taylor'):
        """ Plot Taylor diagram. """
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 8),
                                           subplot_kw={'projection':'polar'})
        self.fig.subplots_adjust(wspace=.25, hspace=.35, bottom=.1,
                                 left=.1, right=.9)
        self.handles = []
        
        obs = self.xps[0].HMM.Obs(0)
        H = obs(np.eye(2))
        std = H@np.diag(np.sqrt(obs.noise.C.diag))
        std[std==0] = np.nan
        
        ax = self.axes[0,0]
        ax.set_title('Forecast x')
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            self.plot_taylor1(ax, self.truth[times,0], xp.stats.bestf[:,0], 
                              style, std=std[0,0])
            
        ax = self.axes[0,1]
        ax.set_title('Forecast y')
        for xp, style in zip(self.xps, self.styles()):
           times = xp.HMM.tseq.kko
           h0 = self.plot_taylor1(ax, self.truth[times,1], xp.stats.bestf[:,1], 
                                  style, std=std[1,-1])
           
        ax = self.axes[1,0]
        ax.set_title('Analysis x')
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            self.plot_taylor1(ax, self.truth[times,0], xp.stats.besta[:,0], 
                              style, std=std[0,0])
            
        ax = self.axes[1,1]
        ax.set_title('Analysis y')
        for xp, style in zip(self.xps, self.styles()):
           times = xp.HMM.tseq.kko
           h0 = self.plot_taylor1(ax, self.truth[times,1], xp.stats.besta[:,1], 
                                  style, std=std[1,-1])
           self.handles.append(h0)
           
        self.axes[0,0].legend(handles=self.handles, ncols=1,
                              bbox_to_anchor=(1.4, 1.05))
        
        for ax in self.axes.flatten():
            ax.set_xlim((0,.5*np.pi))
            ax.set_xticks(np.linspace(0,.5*np.pi,11))
            ax.xaxis.set_major_formatter(lambda x, pos: np.round(1.-2*x/np.pi,1))
            ax.set_ylabel('Correlation')
            
            ticks = self.nice_ticks(ax.get_ylim(), include=[0], max_ticks=8)
            ax.set_rticks(ticks)
            ax.set_xlabel('RMSE')
            ax.xaxis.set_label_coords(.5, -0.1)
        
    def plot_taylor1(self, ax, x1, x2, style, std=None):
        """ Add point to Taylor diagram. """
        from scipy.stats import bootstrap 
        
        corr2rad = lambda c: .5*np.pi*(1-c)
        
        rmse = self.rmse(x1 - x2)
        rmse_range = bootstrap((x1-x2,), self.rmse, method='percentile',
                               confidence_level=self.p_level) 
        rmse_range = np.array(rmse_range.confidence_interval).reshape((2,1))
        
        corr = correlation(np.array([x1,x2]), axis=-1)
        corr_range = bootstrap((np.array([x1,x2]),), correlation, 
                               axis=-1, vectorized=True, method='percentile',
                               confidence_level=self.p_level)
        corr_range = np.array(corr_range.confidence_interval).reshape((2,1))
        
        corr_range, corr = corr2rad(corr_range), corr2rad(corr)
        corr_range = np.abs( corr_range  - corr )
        rmse_range = np.abs(rmse_range-rmse)
        ax.errorbar(corr, rmse, corr_range, rmse_range, color=style[1])
        h0, = ax.plot(corr, rmse, 'o', label=style[0], markersize=6, 
                     color=style[1])
        
        if std is not None and not np.isnan(std):
            theta = np.linspace(0,.5*np.pi,100)
            ax.plot(theta, std*np.ones_like(theta), 'k-')
        
        return h0
    
def correlation(x, axis=-1):
    """ Calculate correlation between x[0] and x[1]."""
    x   = x - np.mean(x, axis=axis, keepdims=True)
    x2  = np.mean(x**2, axis=axis) 
    x12 = np.mean(x[0]*x[1], axis=axis)
    return x12/np.sqrt(x2[0]*x2[1])
    
 
#%% Plot CDFs.    
class ProbPlots(BasePlots):
    """ 
    Class that is used to plot probability distribution of data in different
    series. 
    
    Parameters
    ----------
    fig_dir : str 
        Directory to which figure will be saved. 
        
    """
    
    def __init__(self, fig_dir):
        super().__init__(fig_dir)
        self.cartesian = []
        self.polar = []
        
    def add_series(self, label, xy):
        """ Add series to plot. 
        
        Parameters
        ----------
        label : str
            Name of series. 
        xy : 2D numpy array
            Array with xy[:,0] x-coordinates and xy[:,1] the y-coordinates. 
        
        """
        self.cartesian.append(xy)
        self.polar.append(circle.cartesian2polar(xy))
        self.labels.append(label)
        
    def plot_cdfs_state(self, fig_name='cdfs_state'):   
        """ 
        Plot cumulative probability distribution of x-coordinates, y-coordinates,
        radius and polar angle. 
        
        Parameters
        ----------
        fig_name : str 
            Name of figure. Used to create figure file path. 
            
        """
        
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []
        
        #Calculate p-value
        self.calc_kolmogorov_smirnov()
        def add_pvalue(key, style):
            style = list(style)
            mask = np.array(self.labels, dtype=str) == style[0]
            p = self.pValue[key][0][mask][0]
            style[0] = "{:s} ({:0.2f})".format(style[0], p)
            return style
        
        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            h00 = self.plot_cdf1(self.axes[0,0], xy[:,0], add_pvalue('x',style))
            h01 = self.plot_cdf1(self.axes[0,1], xy[:,1], add_pvalue('y',style))
            h10 = self.plot_cdf1(self.axes[1,0], rt[:,0], add_pvalue('r',style))
            h11 = self.plot_cdf1(self.axes[1,1], np.rad2deg(rt[:,1]), 
                               add_pvalue('theta',style))
            self.handles.append([h00,h01,h10,h11])
        
        self.handles = np.array(self.handles).T
        for handle, ax in zip(self.handles, self.axes.flatten()):
            ax.legend(handles=list(handle), framealpha=.7)
        self.axes[0,0].set_xlabel('x')
        self.axes[0,1].set_xlabel('y')
        self.axes[1,0].set_xlabel('radius')
        self.axes[1,1].set_xlabel('polar angle')
        
        for ax1 in self.axes[0]:
            self.set_nice_xlim(ax1, symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1,0], include=[0], max_ticks=8)
        self.axes[1,1].set_xlim(0,360)
        self.axes[1,1].set_xticks(np.arange(0,361,45))
        
        for ax in self.axes[:,0]:
            ax.set_ylabel('CDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0,1)
            ax.grid()
            
    def plot_pdfs_state(self, fig_name='pdfs_state'):   
        """ 
        Plot cumulative probability distribution of x-coordinates, y-coordinates,
        radius and polar angle. 
        
        Parameters
        ----------
        fig_name : str 
            Name of figure. Used to create figure file path. 
            
        """
        
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []
        
        #Calculate p-value
        self.calc_kolmogorov_smirnov()
        def add_pvalue(key, style):
            style = list(style)
            mask = np.array(self.labels, dtype=str) == style[0]
            p = self.pValue[key][0][mask][0]
            style[0] = "{:s} ({:0.2f})".format(style[0], p)
            return style
        
        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            h00 = self.plot_pdf1(self.axes[0,0], xy[:,0], add_pvalue('x',style))
            h01 = self.plot_pdf1(self.axes[0,1], xy[:,1], add_pvalue('y',style))
            h10 = self.plot_pdf1(self.axes[1,0], rt[:,0], add_pvalue('r',style))
            h11 = self.plot_pdf1(self.axes[1,1], np.rad2deg(rt[:,1]), 
                               add_pvalue('theta',style))
            self.handles.append([h00,h01,h10,h11])
        
        self.handles = np.array(self.handles).T
        for handle, ax in zip(self.handles, self.axes.flatten()):
            ax.legend(handles=list(handle), framealpha=.7)
        self.axes[0,0].set_xlabel('x')
        self.axes[0,1].set_xlabel('y')
        self.axes[1,0].set_xlabel('radius')
        self.axes[1,1].set_xlabel('polar angle')
        
        for ax1 in self.axes[0]:
            self.set_nice_xlim(ax1, symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1,0], include=[0], max_ticks=8)
        self.axes[1,1].set_xlim(0,360)
        self.axes[1,1].set_xticks(np.arange(0,361,45))
        
        for ax in self.axes[:,0]:
            ax.set_ylabel('PDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0,1)
            ax.grid()
        
    def plot_cdfs_latent(self, fig_name='cdfs_latent'):    
        """ 
        Plot cumulative probability distribution of the two latent variables.
        
        Parameters
        ----------
        fig_name : str 
            Name of figure. Used to create figure file path. 
            
        """
        
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []
        
        #Calculate p-value
        self.calc_kolmogorov_smirnov()
        def add_pvalue(key, style):
            style = list(style)
            mask = np.array(self.labels, dtype=str) == style[0]
            p = self.pValue[key][0][mask][0]
            style[0] = "{:s} ({:0.2f})".format(style[0], p)
            return style
        
        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            h0 = self.plot_cdf1(self.axes[0], xy[:,0], add_pvalue('x',style))
            h1 = self.plot_cdf1(self.axes[1], xy[:,1], add_pvalue('y',style))
            self.handles.append([h0,h1])
        
        self.handles = np.array(self.handles).T
        for handle, ax in zip(self.handles, self.axes.flatten()):
            ax.legend(handles=list(handle), framealpha=.7)
        self.axes[0].set_xlabel('z1')
        self.axes[1].set_xlabel('z2')
        
        self.set_nice_xlim(self.axes[0], symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1], symmetric=True, max_ticks=8)

        self.axes[0].set_ylabel('CDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0,1)
            ax.grid()
        
    def plot_cdf1(self, ax, x, style):
        """ 
        Plot cumulative probability distribution for a single variable and series. 
        
        Parameters 
        ----------
        ax : matplotlib.pyplot.axis object 
            Axes to which to add CDF. 
        x : 1D numpy array 
            Data to plot. 
        style : (label, color, style)-tuplet
            Formatting for line. 
            
        """
        bins = self.adaptable_bins(x)
        centers = .5*bins[1:] + .5*bins[:-1]
        n, _ = np.histogram(x, bins=bins)
        n = np.cumsum(n) / np.sum(n)
        
        x = np.sort(x)
        n = np.arange(1, len(x)+1) / len(x)
        
        handle, = ax.plot(x, n, label=style[0], color=style[1],
                          linestyle=style[2])
        
        return handle
    
    def plot_pdf1(self, ax, x, style):
        """ 
        Plot probability distribution for a single variable and series. 
        
        Parameters 
        ----------
        ax : matplotlib.pyplot.axis object 
            Axes to which to add CDF. 
        x : 1D numpy array 
            Data to plot. 
        style : (label, color, style)-tuplet
            Formatting for line. 
            
        """
        step = 5
        bins = self.adaptable_bins(x)
        centers = .5*bins[1:] + .5*bins[:-1]
        n, _ = np.histogram(x, bins=bins)
        n = np.cumsum(n) / np.sum(n)
        
        x = np.sort(x)
        n = np.arange(1, len(x)+1) / len(x)
        
        n = np.diff(n, n=step) / np.diff(x, n=step)
        x = .5*x[step:]+.5*x[:-step]
        
        handle, = ax.plot(x, n, label=style[0], color=style[1],
                          linestyle=style[2])
        
        return handle
            
    def calc_kolmogorov_smirnov(self):
        """ 
        Calculate Kolmogorov-Smirnov test statistic and p-value to test 
        hypothesis that 2 CDFs are the same. 
        """ 
        from scipy.stats import kstest as test
        self.statistic = {'method':'KS','x':[], 'y':[], 'r':[], 'theta':[]}
        self.pValue = {'method':'KS','x':[], 'y':[], 'r':[], 'theta':[]}
        
        for label1, xy1 in zip(self.labels, self.cartesian):
            for label2, xy2 in zip(self.labels, self.cartesian):
                #X
                res = test(xy1[:,0], xy2[:,0], alternative='two-sided')
                self.statistic['x'].append(res.statistic)
                self.pValue['x'].append(res.pvalue)
                #Y
                res = test(xy1[:,1], xy2[:,1], alternative='two-sided')
                self.statistic['y'].append(res.statistic)
                self.pValue['y'].append(res.pvalue)
                
        for label1, xy1 in zip(self.labels, self.polar):
            for label2, xy2 in zip(self.labels, self.polar):
                #X
                res = test(xy1[:,0], xy2[:,0], alternative='two-sided')
                self.statistic['r'].append(res.statistic)
                self.pValue['r'].append(res.pvalue)
                #Y
                res = test(np.rad2deg(xy1[:,1]), np.rad2deg(xy2[:,1]),
                             alternative='two-sided')
                self.statistic['theta'].append(res.statistic)
                self.pValue['theta'].append(res.pvalue)
                
        for key in self.statistic.keys():
            if key=='method':
                continue
            n = len(self.labels)
            self.statistic[key] = np.array(self.statistic[key]).reshape((n, n))
            self.pValue[key] = np.array(self.pValue[key]).reshape((n, n))
            
    def calc_cramer_vonmises(self):
        """ 
        Calculate Cramer-Von Mises test statistic and p-value to test 
        hypothesis that 2 CDFs are the same. 
        """ 
        from scipy.stats import cramervonmises_2samp as test
        self.statistic = {'method':'cramer','x':[], 'y':[], 'r':[], 'theta':[]}
        self.pValue = {'method':'cramer','x':[], 'y':[], 'r':[], 'theta':[]}
        
        for label1, xy1 in zip(self.labels, self.cartesian):
            for label2, xy2 in zip(self.labels, self.cartesian):
                #X
                res = test(xy1[:,0], xy2[:,0])
                self.statistic['x'].append(res.statistic)
                self.pValue['x'].append(res.pvalue)
                #Y
                res = test(xy1[:,1], xy2[:,1])
                self.statistic['y'].append(res.statistic)
                self.pValue['y'].append(res.pvalue)
                
        for label1, xy1 in zip(self.labels, self.polar):
            for label2, xy2 in zip(self.labels, self.polar):
                #X
                res = test(xy1[:,0], xy2[:,0])
                self.statistic['r'].append(res.statistic)
                self.pValue['r'].append(res.pvalue)
                #Y
                res = test(np.rad2deg(xy1[:,1]), np.rad2deg(xy2[:,1]))
                self.statistic['theta'].append(res.statistic)
                self.pValue['theta'].append(res.pvalue)
                
        for key in self.statistic.keys():
            if key=='method':
                continue
            n = len(self.labels)
            self.statistic[key] = np.array(self.statistic[key]).reshape((n, n))
            self.pValue[key] = np.array(self.pValue[key]).reshape((n, n))
            
            
#%% Plot covariances 

class PrincipalPlots(BasePlots):
    """ 
    Class that is used to plot principal values and principal components 
    of the covariances used in the decoder p(x|z) with x in state space and 
    z in latent space. 
    
    Parameters
    ----------
    fig_dir : str 
        Directory to which figure will be saved. 
        
    """
    
    def __init__(self, fig_dir):
        super().__init__(fig_dir)
        self.xy, self.pc, self.angle = [], [], []
        
    def add_series(self, label, xy, pc, sin):
        """ Add series to plot. 
        
        Parameters
        ----------
        label : str
            Name of series. 
        xy : 2D numpy array
            Array with xy[:,0] x-coordinates and xy[:,1] the y-coordinates. 
        pc : 2D numpy array 
            Array with in each row one of the principal values. 
        sin : 1D numpy array 
            Array with sin(theta) where theta is the polar angle between 
            the 1st principal component and the x-axis. 
        
        """
        self.labels.append(label)
        self.xy.append(xy)
        self.pc.append(pc)
        self.angle.append(np.arcsin(sin))
        
    def plot_pc(self, fig_name='pc'):
        """ Add series to plot. 
        
        Parameters
        ----------
        label : str
            Name of series. 
        xy : 2D numpy array
            Array with xy[:,0] x-coordinates and xy[:,1] the y-coordinates. 
        
        """
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                           figsize=(8,6))
        self.fig.subplots_adjust(wspace=.25,  left=.05, right=.95)
        
        #Plot principal components
        self.handles = []
        for xy, pc, angle, style in zip(self.xy, self.pc, self.angle, self.styles()):
            h = self.plot_pc1(self.axes[0], [xy,pc,angle], style)
            self.plot_ellipses1(self.axes[1], [xy,pc,angle], style)
            self.handles.append(h)
        
        #Complete layout
        self.axes[0].legend(handles=self.handles, loc='upper left')
        self.axes[0].set_title('Principal components')        

    def binner(self, x, y, axis=0, bins=None):
        """ 
        Group data into bins and calculate bin means and standard deviation. 
        
        Parameters
        ----------
        x : 1D numpy array 
            Data used to define bins. 
        y : 1D numpy array 
            Data to be binned. 
        axis : int 
            Axis of y that is reduced by binning. 
        bins : 1D numpy array 
            Optional bin edges used for binning. 
            
        Returns 
        -------
        centres 
            Bin means of x. 
        means
            Bin means of y.
        std 
            Bin standard deviatio of y. 
            
        """
        
        if bins is None:
            bins = self.adaptable_bins(x)
            
        y = np.swapaxes(y, axis, 0)
        centers, mean, std = [], [], []
        for bin0, bin1 in zip(bins[:-1],bins[1:]):
            mask = np.logical_and(x>=bin0, x<=bin1)
            centers.append(np.mean(x[mask]))
            mean.append(np.mean(y[mask], axis=0))
            std.append(np.std(y[mask], ddof=1, axis=0))
        
        centers = np.array(centers)
        mean = np.array(mean).swapaxes(0, axis)
        std = np.array(std).swapaxes(0, axis)
        
        return centers, mean, std
        
    def plot_pc1(self, ax, data, style):
        """ 
        Plot largenst and smallest principal component as function of 
        polar angle for a single series. 
        
        Parameters
        ----------
        data : (xy, pc, angle)-tuplet 
            2D numpy array with x and y coordinates, 2D numpy array with 
            principal component and 1D array with rotation angle covariance. 
        style: (label, color, style)-tuplet
            Layout for the series in plot. 
            
        """
        
        xy, pc, angle = data 
        label, color, linestyle = style 
        
        rt = circle.cartesian2polar(xy)
        theta = rt[:,1]
        pc_max = np.max(pc, axis=1)
        pc_min = np.min(pc, axis=1)
        
        for pc in [pc_max, pc_min]:
            centers, mean, std = self.binner(theta, pc)
            ax.fill_between(centers, mean-std, mean+std, color=color,
                           alpha=0.3)
            handle, = ax.plot(centers, mean, label=label, color=color, 
                              linestyle=linestyle)
            
        return handle
    
    def plot_ellipses1(self, ax, data, style):
        """ 
        Plot ellipses with principal component as major and minor-axis for 
        a single series. 
        
        Parameters
        ----------
        data : (xy, pc, angle)-tuplet 
            2D numpy array with x and y coordinates, 2D numpy array with 
            principal component and 1D array with rotation angle covariance. 
        style: (label, color, style)-tuplet
            Layout for the series in plot. 
        
        
        """ 
        
        xy, pc, angle = data 
        rt = circle.cartesian2polar(xy)
        xy, pc, angle = xy.T, pc, angle.flatten()
        label, color, linestyle = style 
        
        #Calculate covariance matrices 
        def R(theta):
            return np.array([[np.cos(theta),-np.sin(theta)],
                             [np.sin(theta),np.cos(theta)]])
        covs = np.array([R(theta1)@np.diag(pc1)@R(theta1).T for theta1, pc1 in zip(angle, pc)])
       
        #Average covariance matrices by first turning them into vectors
        data = np.reshape(covs, (-1,4))
        data = np.concatenate((data, rt), axis=-1)
        bins = np.linspace(0, 2*np.pi, 25, endpoint=True)
        theta, mean, std = self.binner(rt[:,1], data, axis=0, bins=bins)
        
        #Plot eclipses
        def add_ellipse(r, theta, cov):
            if np.isnan(r):
                return
            
            #Principal axes ecllips
            s, V = np.linalg.eig(cov)
            V = V@np.diag(s)
            
            #Curve ecllips
            t = np.linspace(0, 2*np.pi, 50, endpoint=True).reshape((1,-1))
            eclipse = V[:,:1] * np.cos(t) + V[:,1:] * np.sin(t)
            eclipse[0] += r*np.cos(theta)
            eclipse[1] += r*np.sin(theta)
            
            #Plot ellips in polar coordinates
            ax.plot(np.arctan2(eclipse[1], eclipse[0]), 
                    np.linalg.norm(eclipse, axis=0), color=color)
            
        for data1 in mean:
            r1, theta1 = data1[4], data1[5]
            cov1 = np.reshape(data1[:4],(2,2))
            add_ellipse(r1, theta1, cov1)

#%% 2D plot of circle

class CirclePlot(BasePlots):

    def __init__(self, fig_dir):
        self.fig_dir = fig_dir 
        self.labels = set()
        self.ens_for, self.ens_ana, self.tracks, self.obs = {}, {}, {}, {}

    def add_ens_for(self, label, times, E):        
        #Store data in dict. 
        self.ens_for[label] = {'time': times, 'data': E}
        self.labels = self.labels.union([label])
        
    def add_ens_ana(self, label, times, E):        
        #Store data in dict. 
        self.ens_ana[label] = {'time': times, 'data':E}
        self.labels = self.labels.union([label])
            
    def add_track(self, label, times, x):
        self.tracks[label] = {'time': times, 'data': x}
        self.labels = self.labels.union([label])
        
    def add_obs(self, times, y):
        self.obs = {'time': times, 'data':y}
        
    def plot_circle(self, ax, radius=1):
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), '-',
                color=(.7,.7,.7))
        
    def assign_styles(self):
        styles = self.styles()
        for style in styles:
            if style[0] in self.tracks:
                self.tracks[style[0]]['style'] = style 
            if style[0] in self.ens_for:
                self.ens_for[style[0]]['style'] = style 
            if style[0] in self.ens_ana:
                self.ens_ana[style[0]]['style'] = style
        
    def plot_time(self, time, fig_name='trajectory'):
        
        if not hasattr(self, 'axes') or self.axes is None:
            plt.close('all')
            self.fig, self.axes = plt.subplots(1, 1, figsize=(6, 6))
            self.axes = np.array([self.axes])
            self.assign_styles()
        else:
            for ax in self.axes.flatten():
                ax.clear()
        
        self.fig_name = fig_name
        self.handles = []
        self.plot_circle(self.axes[0])
        
        for key, value in self.ens_for.items():
            mask = value['time'] == time
            
            #Plot forecast ensemble
            self.axes[0].plot(value['data'][mask][0,:,0], 
                              value['data'][mask][0,:,1], 'o',
                              alpha=.2, color = value['style'][1],
                              label = value['style'][0], markeredgewidth=0)
            
            #Plot forecast mean
            m = np.mean(value['data'][mask], axis=1)
            h, = self.axes[0].plot(m[0,0], m[0,1], 'o',
                                   alpha=1., color = value['style'][1], 
                                   label = value['style'][0])
            
            self.handles.append(h)
            
        #Add observation 
        mask = self.obs['time'] == time 
        self.axes[0].plot(np.array([1,1])*self.obs['data'][mask][0], np.array([-2,2]),'k--')
            
        #Add truth
        for key, value in self.tracks.items():
            mask = value['time'] == time
            h, = self.axes[0].plot(value['data'][mask][0,0], value['data'][mask][0,1], 
                                   'o', alpha=1., color = value['style'][1],
                                   label = value['style'][0])
            self.handles.append(h)
            
        for ax in self.axes.flatten():
            ax.grid()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(handles=self.handles, loc='center')
            ax.set_title('Time {:5d}'.format(time))
            
            ax.set_xlim(-1.25,1.25)
            ax.set_ylim(-1.25,1.25)
            ax.set_aspect(1)
            
    def animate_time(self, times, fig_name='movie_time', fps=30):
        """ 
        Create animation of ensemble. 
        """
        fig, ax = plt.subplots(1,1)
        
        if self.fig_dir is not None:
            tmp_dir = os.path.join(self.fig_dir, 'tmp')
            os.mkdir(tmp_dir)
            
            print('Printing figures.')
            for it,t in  enumerate(times):
                fig_name1 = os.path.join('tmp', 'frame_{:04d}'.format(it))
                self.plot_time(t, fig_name=fig_name1)
                self.save()
                
            print('Compiling figures into animation.')
            fmt = os.path.join(tmp_dir,'frame_%04d')
            file_path = os.path.join(self.fig_dir, fig_name+'.mp4')
            cmd = (f'ffmpeg -f image2 -r {fps} -i {fmt} -vcodec libx264 -y '
                   f'-profile:v high444 -refs 16 -crf 0 -preset ultrafast {file_path}')
            os.system(cmd)
            shutil.rmtree(tmp_dir)

#%% Axes

def plot_circle2d(ax, radius=1):
    """ Add circle to plot of complex plane. """
    theta = np.linspace(0,2*np.pi,100,endpoint=True)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-')
    ax.set_aspect(1)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.grid()
      




