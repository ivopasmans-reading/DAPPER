#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing plotting routines for VAE

@author: ivo
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
from dapper.mods import ComplexCircle as circle
from scipy.stats import norm
import os
from abc import ABC, abstractmethod

mpl.rcParams['lines.linewidth'] = 2 
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'




#%% Supporting functions

class SeriesPlots: 
    
    def __init__(self, fig_dir):
        self.fig_dir = fig_dir
        self.labels = []
    
    @staticmethod
    def adaptable_bins(x, alpha=0.05):
        """ Create bins of different size. """
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
    
    def styles(self):
        from matplotlib import colormaps as cm
        colors = [cm.get_cmap('Set1')(i) for i in range(9)]
        styles = ['-', (0,(1,1)), (5,(10,3)), 
                  (0,(3,1,1,1)), (0,(3,10,1,10,1,10)), (5,(10,3)),
                  (0,(3,1,1,1,1,1)), (0,(3,5,1,5,1,5)), (0,(5,5))]
        return zip(self.labels,colors,styles)
    
    def save(self):
        self.fig.savefig(self.fig_path, resolution=400, format='png')
    
class ProbPlots(SeriesPlots):
    
    def __init__(self, fig_dir):
        super().__init__(fig_dir)
        self.cartesian = []
        self.polar = []
        
    def add_series(self, label, xy):
        self.cartesian.append(xy)
        self.polar.append(circle.cartesian2polar(xy))
        self.labels.append(label)
     
    def plot_cdf1(self, ax, x, style):
        bins = self.adaptable_bins(x)
        centers = .5*bins[1:] + .5*bins[:-1]
        n, _ = np.histogram(x, bins=bins)
        n = np.cumsum(n) / np.sum(n)
        
        x = np.sort(x)
        n = np.arange(1, len(x)+1) / len(x)
        
        handle, = ax.plot(x, n, label=style[0], color=style[1],
                          linestyle=style[2])
        
        return handle
        
    def plot_cdfs(self, fig_name='cdfs'):
        plt.close('all')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []
        
        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            self.plot_cdf1(self.axes[0,0], xy[:,0], style)
            self.plot_cdf1(self.axes[0,1], xy[:,1], style)
            self.plot_cdf1(self.axes[1,0], rt[:,0], style)
            h = self.plot_cdf1(self.axes[1,1], np.rad2deg(rt[:,1]), style)
            self.handles.append(h)
            
        self.axes[0,0].legend(handles=self.handles)
        self.axes[0,0].set_xlabel('x')
        self.axes[0,1].set_xlabel('y')
        self.axes[1,0].set_xlabel('radius')
        self.axes[1,1].set_xlabel('polar angle')
        
        for ax in self.axes[0,:]:
            ax.set_xlim(-1.2,1.2)
        self.axes[1,0].set_xlim(0,1.2)
        self.axes[1,1].set_xlim(0,360)
        self.axes[1,1].set_xticks(np.arange(0,361,45))
        
        for ax in self.axes[:,0]:
            ax.set_ylabel('CDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0,1)
            ax.grid()
            
        self.fig_path = os.path.join(self.fig_dir, fig_name)
            
    def calc_kolmogorov_smirnov(self):
        from scipy.stats import kstest
        self.KS = {'x':[], 'y':[], 'r':[], 'theta':[]}
        self.pValue = {'x':[], 'y':[], 'r':[], 'theta':[]}
        
        for label1, xy1 in zip(self.labels, self.cartesian):
            for label2, xy2 in zip(self.labels, self.cartesian):
                #X
                res = kstest(xy1[:,0], xy2[:,0], alternative='two-sided')
                self.KS['x'].append(res.statistic)
                self.pValue['x'].append(res.pvalue)
                #Y
                res = kstest(xy1[:,1], xy2[:,1], alternative='two-sided')
                self.KS['y'].append(res.statistic)
                self.pValue['y'].append(res.pvalue)
                
        for label1, xy1 in zip(self.labels, self.polar):
            for label2, xy2 in zip(self.labels, self.polar):
                #X
                res = kstest(xy1[:,0], xy2[:,0], alternative='two-sided')
                self.KS['r'].append(res.statistic)
                self.pValue['r'].append(res.pvalue)
                #Y
                res = kstest(np.rad2deg(xy1[:,1]), np.rad2deg(xy2[:,1]),
                             alternative='two-sided')
                self.KS['theta'].append(res.statistic)
                self.pValue['theta'].append(res.pvalue)
                
        for key in self.KS.keys():
            n = len(self.labels)
            self.KS[key] = np.array(self.KS[key]).reshape((n, n))
            self.pValue[key] = np.array(self.pValue[key]).reshape((n, n))


class PrincipalPlots(SeriesPlots):
    
    def __init__(self, fig_dir):
        super().__init__(fig_dir)
        self.xy, self.pc, self.angle = [], [], []
        
    def add_series(self, label, xy, pc, angle):
        self.labels.append(label)
        self.xy.append(xy)
        self.pc.append(pc)
        self.angle.append(angle)
        
    def plot_pc(self):
        self.fig, self.axes = plt.subplots(2, 1, subplot_kw={'projection': 'polar'})
        
        #Plot principal components
        self.handles = []
        for xy, pc, angle, style in zip(self.xy, self.pc, self.angle, self.styles()):
            h = self.plot_pc1(self.axes[0], [xy,pc,angle], style)
            self.handles.append(h)
            
        self.axes[0].legend(handles=self.handles)

    def binner(self, x, y, axis=0):
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
        xy, pc, angle = data 
        label, color, linestyle = style 
        
        rt = circle.cartesian2polar(xy)
        theta = np.rad2deg(rt[:,1])
        pc_max = np.maximum(pc, axis=1)
        pc_min = np.minimum(pc, axis=1)
        
        for pc in [pc_max, pc_min]:
            centers, mean, std = self.binner(theta, pc)
            ax.fillbetween(centers, mean, mean-std, mean+std, color=color,
                           linestyle=None, alpha=0.3)
            handle, = ax.plot(centers, mean, label=label, color=color, 
                              linestyle=linestyle)
            
        return handle
        
    
        

#%% Axes

def plot_circle2d(ax, radius=1):
    """ Add circle to plot of complex plane. """
    theta = np.linspace(0,2*np.pi,100,endpoint=True)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-')
    ax.set_aspect(1)
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.grid()
      




