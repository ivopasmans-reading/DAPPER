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
import os
from abc import ABC, abstractmethod

#Default settings for layout. 
mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'

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

class SeriesPlots: 
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
    def nice_ticks(lims, max_ticks=12, symmetric=False, origin=False):
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
        origin : boolean
            Indicate whether 0 should be one of the ticks. 
            
        """
        
        lims = np.array(lims)
        if origin:
            lims = np.append(lims, [0])
            
        lims = np.array([np.min(lims), np.max(lims)])  
        if symmetric:
            lims = np.array([-1,1]) * np.max(np.abs(lims))
    
        dlim = np.log10( max(1e-6, np.diff(lims)[0]) )
        order, r = 10**int(dlim), 10**np.mod(dlim, 1.)
        
        #Find right step
        for step in StepGenerator(order):
            nticks = np.ceil(r / step)
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
        
        if lims is None:
            lims = np.array(ax.get_xlim())
        ticks = SeriesPlots.nice_ticks(lims, **kwargs)
        
        ax.set_xticks(ticks)
        ax.set_xlim(ticks[0], ticks[-1])
        
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
        
        if lims is None:
            lims = np.array(ax.get_ylim())
        ticks = SeriesPlots.nice_ticks(lims, **kwargs)
        
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])
        
        return ax
   
    def styles(self):
        """ Generate 9 different combinations of color and linestyle. """
        from matplotlib import colormaps as cm
        colors = [cm.get_cmap('Set1')(i) for i in range(9)]
        styles = ['-', (0,(1,1)), (5,(10,3)), 
                  (0,(3,1,1,1)), (0,(3,10,1,10,1,10)), (5,(10,3)),
                  (0,(3,1,1,1,1,1)), (0,(3,5,1,5,1,5)), (0,(5,5))]
        return zip(self.labels,colors,styles)
    
    @property 
    def fig_path(self):
        if self.fig_path is None:
            return None 
        else:
            return os.path.join(self.fig_dir, self.fig_name)
    
    def save(self):
        """ Save figure to file path in self.fig_path. """
        if self.fig_dir is not None:
            if not os.path.exists(self.fig_dir):
                os.mkdir(self.fig_dir)
            self.fig.savefig(self.fig_path, dpi=400, format='png')
 
#%% Plot CDFs.    
 
class ProbPlots(SeriesPlots):
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
        
        self.set_nice_xlim(self.axes[0,0], symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[0,1], symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1,0], origin=True, max_ticks=8)
        self.axes[1,1].set_xlim(0,360)
        self.axes[1,1].set_xticks(np.arange(0,361,45))
        
        for ax in self.axes[:,0]:
            ax.set_ylabel('CDF')
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
        
        self.set_nice_xlim(self.axes[0], symmetric=True)
        self.set_nice_xlim(self.axes[1], symmetric=True)

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

class PrincipalPlots(SeriesPlots):
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


#%% Axes

def plot_circle2d(ax, radius=1):
    """ Add circle to plot of complex plane. """
    theta = np.linspace(0,2*np.pi,100,endpoint=True)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-')
    ax.set_aspect(1)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.grid()
      




