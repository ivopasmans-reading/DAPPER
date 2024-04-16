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
import shutil
from abc import ABC, abstractmethod
from scipy.stats import poisson
import scipy

# Default settings for layout.
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'


def smod(x1, x2, *args):
    return np.mod(x1+.5*x2, x2, *args)-.5*x2


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
        # Check if ensemble is Gaussian
        _, pvalue = shapiro(E)
        if pvalue > p:
            break

        # Eliminate the outlier.
        n = np.sum(D < d1, axis=1)
        if np.all(n == np.min(n)):
            continue
        else:
            mask = n > np.min(n)

        E = E[mask]
        D = D[mask, :]
        D = D[:, mask]

    mu = np.mean(E, axis=0)
    return mu


# %% Abstract classes for plotting.

class StepGenerator(object):
    """ Returns possible tick distances in decreasing order. """

    def __init__(self, factor=1):
        self.steps = np.array([1, 2, 2.5, 5], dtype=float) * factor
        self.n = len(self.steps)

    def __iter__(self):
        return self

    def __next__(self):
        self.n = self.n - 1
        if self.n < 0:
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
        # Number of bins
        n = len(x)
        k = int(4*(2*n**2 / norm.isf(alpha))**.2)
        nbin = int(n / k)+1
        # Sort in ascending error
        x = np.sort(x)
        # Create bins
        bins, i = np.array([x[0]]), nbin
        while i < n-1:
            bins = np.append(bins, .5*x[i-1]+.5*x[i])
            i = i + nbin
        bins = np.append(bins, x[-1])
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
            lims = np.array([-1, 1]) * np.max(np.abs(lims))

        dlim = np.log10(max(1e-6, np.diff(lims)[0]))
        order, r = 10**np.floor(dlim), 10**np.mod(dlim, 1.)

        # Find right step
        for step in StepGenerator(order):
            nticks = np.ceil(r*order / step)
            if nticks > max_ticks:
                break
            else:
                step0 = step

        lbound = np.floor(lims[0] / step0) * step0
        ubound = np.ceil(lims[-1] / step0) * step0
        ticks = np.arange(lbound, ubound+step0, step0)

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
        styles = ['-', (0, (1, 1)), (5, (10, 3)),
                  (0, (3, 1, 1, 1)), (0, (3, 10, 1, 10, 1, 10)), (5, (10, 3)),
                  (0, (3, 1, 1, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (5, 5))]
        return zip(self.labels, colors, styles)

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

# %% CPRS

class CRPS:
    """ 
    Class to calculate the Continuous Rank Probability Score. 
    
    See Hersbach (2000) for definitions and explanation. 
    """

    def __init__(self, ens, truth, weights=1.0):
        """ 
        Class constructor. 
        
        ens : TxNxM numpy array 
            Array with N ensemble members at different times T. 
        truth : TxM 
            Array with truth at different times T. 
        weights : float | int | M numpy array 
            Array with weighting of different variables. 
        
        """ 
        
        #Ensemble members along axis=0
        self.ens = np.swapaxes(ens, 1, 0)
        self.ens = np.sort(self.ens, axis=0)
        self.N = np.size(self.ens, 0)
        self.truth = truth
        self.weights = np.reshape(weights+np.zeros((np.size(ens, -1),)), 
                                  (1, 1, -1))
        
    def calculate_widths(self):
        """ 
        Average width of spacing between ensemble members. 
        """
        truth = self.truth[None,...]
        ens = np.concatenate((truth,self.ens,truth), axis=0)
        # Width between ensemble members.
        diff  = np.diff(ens, axis=0)
        #Average width. 
        mask  = diff > 0.0
        diff  = np.nanmean(self.weights * diff, axis=-1, where=mask)
        w     = np.nanmean(self.weights + np.zeros_like(mask), axis=-1, 
                           where=mask)
        diff  = np.where(~np.isnan(w), diff/w, 0.0)

        return diff

    def calculate_indicator(self):
        """
        Average fraction of spacing smaller than the truth. 
        """
        # H(x-x_truth)
        beta  = self.ens[1:] - np.maximum(self.truth[None,...], self.ens[:-1]) 
        beta  = np.nanmean(np.maximum(beta, 0.0) * self.weights, axis=-1)
        beta /= np.nanmean(self.weights, axis=-1)
        
        # Width between ensemble members.
        diff  = np.maximum(self.ens[1:] - self.ens[:-1], np.finfo(float).resolution)
        diff  = np.nanmean(self.weights * diff, axis=-1)
        diff /= np.nanmean(self.weights, -1)
        
        #Probability smaller than ensemble. 
        under  = self.truth[None,...]<self.ens[:1]
        under  = np.mean(under*self.weights, axis=-1) 
        under /= np.mean(self.weights, axis=-1)
        
        #Probability larger than ensemble.
        over  = self.truth[None,...]>self.ens[-1:]
        over  = np.mean(over*self.weights, axis=-1) 
        over /= np.mean(self.weights, axis=-1)
        
        return np.concatenate((under, beta/diff, over), axis=0)

    def calculate_reliability(self, widths, indicators):
        prob = np.arange(0, self.N+1) / self.N
        prob = np.reshape(prob, (-1, 1))
        
        return np.sum(widths * (indicators - prob)**2, axis=0)
    
    def calculate_crps_pot(self, widths, indicators):
        return np.sum(widths * indicators * (1-indicators), axis=0)

    def calculate_uncertainty(self):
        weights = self.weights[0] + np.zeros_like(self.truth)
        weights = np.reshape(weights, (-1,)) 
        truth = np.reshape(self.truth, (-1,))

        indices = np.argsort(truth)
        truth = np.take(truth, indices)
        weights = np.take(weights, indices)

        probs = np.cumsum(weights)[:-1] / np.sum(weights)
        diffs = np.diff(truth)

        return np.sum(probs * (1-probs) * diffs, axis=-1)

    def calculate_resolution(self, uncertainty, crps_pot):
        return uncertainty - crps_pot
    
    def calculate_rank(self, p=.9):
        M = np.size(self.ens, 1)
        bins = np.arange(0, self.N+1)
        ranks = np.sum(self.ens<=self.truth[None,...], axis=0)
        rank_count = np.zeros((self.N+1, np.size(ranks,-1)))
        for rank in bins:
            rank_count[rank,:] = np.sum(ranks==rank, axis=0)
        
        #Calculate confidence interval
        P = poisson(M)
        dp = 0.5*(1-p)
        low, high = P.isf(1-dp), P.isf(dp)
        
        return rank_count / M, (low/M, high/M)
    
    def calculate_crps(self, widths, indicators):
        probs = np.arange(0, self.N+1) / self.N
        probs = np.reshape(probs, (-1,1))
        
        crps  = (1-indicators) * probs**2
        crps += indicators * (1-probs)**2
        crps *= widths 
        
        return np.sum(crps, axis=0)

    def __call__(self):
        """ 
        Calculate the CRPS and its components. 
        """
        widths = self.calculate_widths()
        indicators = self.calculate_indicator()
        reliability = self.calculate_reliability(widths, indicators)
        crps_pot = self.calculate_crps_pot(widths, indicators)
        uncertainty = self.calculate_uncertainty()
        resolution = self.calculate_resolution(uncertainty, crps_pot)
        crps = reliability + crps_pot
        
        error = crps - self.calculate_crps(widths, indicators)
        if not np.isclose(np.linalg.norm(error), 0.0):
            raise Exception('CRPS not consistent.')

        return crps, reliability, resolution, uncertainty

# %% Plot of ensemble statistics. 


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
        self.best_calculated = False

    def rmse(self, a, **kwargs):
        """ Calculate root-mean square error."""
        return np.sqrt(np.mean(a**2, **kwargs))

    def plot_crps1(self, axes, E, xx, style, mode='f'):
        times = self.HMM.tseq.kko
        crps = CRPS(E, xx)
        stats = crps()

        for stat1, ax1 in zip(stats, axes):
            h0, = ax1.plot(times, stat1, label=style[0], color=style[1],
                           linestyle=style[2])

        axes[-1].plot(times, stats[-1]*np.ones_like(times), 'k--',
                      label='uncertainty')

        return h0

    def plot_crps(self, fig_name='cprs'):
        """ 
        Plot cumulative probability ranking statistics
        """

        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(4, 3, figsize=(8, 11))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98, top=.96)
        times = self.HMM.tseq.kko

        for axes, func in zip(self.axes, self.best_funcs()):
            xx = func(self.truth)
            xx = xx[..., None]
            self.handles = []
            for xp, style in zip(self.xps, self.styles()):
                E = func(xp.stats.E.a)[..., None]
                self.handles.append(self.plot_crps1(axes, E, xx[times], style))

        lax = self.fig.add_axes([0.05, .0, .9, .05],)
        lax.set_axis_off()
        lax.legend(handles=self.handles, loc='lower center', ncol=4)

        for ax, title in zip(self.axes[0], ['CRPS', 'Reliability', 'Resolution']):
            ax.set_title(title)
        for ax, title in zip(self.axes[:, 0], ['x', 'y', 'radius', 'angle']):
            ax.set_ylabel(title)

        for ax in self.axes[-1]:
            ax.set_xlabel('Time')
        for ax in self.axes[:, :2].flatten():
            ticks = self.nice_ticks(ax.get_ylim(), max_ticks=6, minlim=0)
            ax.yaxis.ticks = ticks
            ax.set_ylim(np.min(ticks), np.max(ticks))

    def calculate_crps(self, fig_name='stats'):
        stream = open(os.path.join(self.fig_dir,fig_name+'.txt'),'w')
        
        times = self.HMM.tseq.kko
        fmt = "{:10s}" + 4*" {:>12s}"
        stream.write(fmt.format('', "CRPS", "Reliability", "Resolution", "Uncertainty\n"))
        fmt = "{:10s}" + 4*" {:12.3f}" + '\n'
        
        for func, varname in zip(self.best_funcs(), ['x', 'y', 'radius', 'angle']):
            stream.write('Variable '+varname+'\n')
            xx = func(self.truth)[times]
            xx = xx[None, ...]

            for xp in self.xps:
                E = func(xp.stats.E.a)[..., None]
                N = np.size(E, 1)
                E = np.reshape(np.swapaxes(E, 0, 2), (1, N, -1))

                if varname == 'angle':
                    E = smod(E - xx[None, ...], 360)
                    E += xx[None, ...]

                crps = CRPS(E, xx)
                stats = crps()

                stats = np.append(np.array(stats[:-1]).flatten(), stats[-1])
                stats = tuple(stats)
                stream.write(fmt.format(xp.name, *stats))
                
        stream.close()

    def add_xp(self, xp):
        """ Add experiment for processing. """
        self.labels.append(xp.name)
        self.xps.append(xp)

    def calculate_best(self):
        for xp in self.xps:
            xp.stats.besta = np.array([ens2ml(E) for E in xp.stats.E.a])
            xp.stats.bestf = np.array([ens2ml(E) for E in xp.stats.E.f])

        self.best_calculated = True

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

        if not self.best_calculated:
            self.calculate_best()

        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            rmse = self.truth[times] - xp.stats.bestf
            rmse = self.rmse(rmse, axis=1)
            h0, = self.axes[0, 0].plot(times, rmse, label=style[0], color=style[1],
                                       linestyle=style[2])

            rmse = self.truth[times] - xp.stats.besta
            rmse = self.rmse(rmse, axis=1)
            h0, = self.axes[0, 1].plot(times, rmse, label=style[0], color=style[1],
                                       linestyle=style[2])

            E = xp.stats.E.f
            std = np.hypot(np.std(E[:, :, 0], axis=1, ddof=1),
                           np.std(E[:, :, 1], axis=1, ddof=1))
            self.axes[1, 0].plot(times, std, label=style[0], color=style[1],
                                 linestyle=style[2])

            E = xp.stats.E.a
            std = np.hypot(np.std(E[:, :, 0], axis=1, ddof=1),
                           np.std(E[:, :, 1], axis=1, ddof=1))
            self.axes[1, 1].plot(times, std, label=style[0], color=style[1],
                                 linestyle=style[2])

            self.handles.append(h0)

        self.axes[0, 0].set_ylabel('forecast RMSE')
        self.axes[0, 1].set_ylabel('analysis RMSE')
        self.axes[1, 0].set_ylabel('forecast ensemble std. dev.')
        self.axes[1, 1].set_ylabel('analysis ensemble std. dev.')
        self.axes[0, 0].legend(handles=self.handles, loc='upper right')

        self.set_nice_ylim(self.axes[0], include=[0], minlim=0)
        self.set_nice_ylim(self.axes[1], include=[0], minlim=0)

        for ax in self.axes.flatten():
            self.set_nice_xlim(ax, (0, np.max(times)), max_ticks=10)
            ax.set_xlabel('Time')
            ax.grid()

    def plot_rmse_std_variables(self, fig_name='rmse_variables', mode='f'):
        """
        Plot RMSE and ensemble standard deviation. 
        """

        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 4, figsize=(11, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.98)
        self.handles = []

        if not self.best_calculated:
            self.calculate_best()

        for xp, style in zip(self.xps, self.styles()):
            print('XP ', xp.name)
            if mode == 'f':
                E, best, mode_label = xp.stats.E.f, xp.stats.bestf, 'Forecast'
            elif mode == 'a':
                E, best, mode_label = xp.stats.E.a, xp.stats.besta, 'Analysis'

            for ax, func in zip(self.axes.T, self.best_funcs()):
                times = xp.HMM.tseq.kko
                rmse = func(self.truth[times]) - func(best)
                if ax[0] == self.axes[0, -1]:
                    rmse = np.mod(rmse+180., 360.)-180.

                rmse = self.rmse(rmse.reshape((-1, 1)), axis=1)
                h0, = ax[0].plot(times, rmse, label=style[0], color=style[1],
                                 linestyle=style[2])

                E1 = func(E)
                if ax[1] == self.axes[1, -1]:
                    j = complex(0, 1)
                    E1 = np.exp(j*np.deg2rad(E1))
                    mu = np.prod(E1, axis=1)**(1/np.size(E1, 1))
                    E1 = E1 / mu.reshape((-1, 1))
                    E1 = np.rad2deg(np.imag(np.log(E1)))
                std = np.std(E1, axis=1, ddof=1)
                ax[1].plot(times, std, label=style[0], color=style[1],
                           linestyle=style[2])

            self.handles.append(h0)

        self.axes[0, 0].set_ylabel(mode_label+' RMSE')
        self.axes[1, 0].set_ylabel(mode_label+' ensemble std. dev.')
        for ax, title in zip(self.axes[0], ['x', 'y', 'radius', 'angle']):
            ax.set_title(title)

        for ax in self.axes[1]:
            ax.set_xlabel('Time')

        for ax in self.axes.flatten():
            self.set_nice_ylim(ax, include=[0], minlim=0)
            self.set_nice_xlim(ax, (0, np.max(times)), max_ticks=6)
            ax.grid()

        self.axes[0, 0].legend(handles=self.handles, loc='upper right')

    def plot_best1(self, times, xy, style):
        """ Plot best estimate. """
        for ax, func in zip(self.axes.flatten(), self.best_funcs()):
            h0, = ax.plot(times, func(xy), label=style[0], color=style[1],
                          linestyle=style[2])

        return h0

    def plot_best1_range(self, times, xy, E, style, p=.65):
        """ Plot range closest to to best estimate. """
        e = E - xy.reshape((np.size(E, 0), 1, np.size(E, -1)))
        e = np.linalg.norm(e, axis=-1)
        ind = np.argsort(e, axis=-1)

        Eind = []
        for E1, ind1 in zip(E, ind):
            Eind.append(E1[ind1[:int(p*len(ind1))], :])
        Eind = np.array(Eind)

        for ax, func in zip(self.axes.flatten()[:-1], self.best_funcs()):
            Eind1 = func(Eind)
            h0 = ax.fill_between(times, np.min(Eind1, axis=-1), np.max(Eind1, axis=-1),
                                 label=style[0], color=style[1], linestyle=style[2], alpha=.3)

        return h0

    def best_funcs(self):
        """ Subtract different variables from ensemble. """
        def xfunc(E): return E[..., 0]
        def yfunc(E): return E[..., 1]
        def rfunc(E): return np.linalg.norm(E, axis=-1)
        def tfunc(E): return np.mod(np.rad2deg(
            np.arctan2(yfunc(E), xfunc(E))), 360)
        return [xfunc, yfunc, rfunc, tfunc]

    def plot_best(self, fig_name='best'):
        """ Plot best estimates. """
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 8))
        self.fig.subplots_adjust(wspace=.25,  left=.1, right=.95)
        self.handles = []

        times = self.HMM.tseq.kk
        h0 = self.plot_best1(times, self.truth[times], ('truth', 'k', '-'))
        self.handles.append(h0)

        if not self.best_calculated:
            self.calculate_best()

        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            _ = self.plot_best1_range(
                times, xp.stats.besta, xp.stats.E.a, style)
            h0 = self.plot_best1(times, xp.stats.besta, style)
            self.handles.append(h0)

        self.axes[0, 0].legend(handles=self.handles, loc='upper right')
        self.axes[0, 0].set_ylabel('x')
        self.set_nice_ylim(self.axes[0, 0], symmetric=True)
        self.axes[0, 1].set_ylabel('y')
        self.set_nice_ylim(self.axes[0, 1], symmetric=True)
        self.axes[1, 0].set_ylabel('radius')
        self.set_nice_ylim(self.axes[1, 0], include=[0])
        self.axes[1, 1].set_ylabel('angle')
        self.axes[1, 1].set_yticks(np.arange(0, 361, 45))
        self.axes[1, 1].set_ylim(0, 360)

        for ax in self.axes[1]:
            ax.set_xlabel('Time')
        for ax in self.axes.flatten():
            self.set_nice_xlim(ax, lims=(0, np.max(times)), max_ticks=12)
            ax.grid()

    def plot_taylor(self, fig_name='taylor'):
        """ Plot Taylor diagram. """
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 8),
                                           subplot_kw={'projection': 'polar'})
        self.fig.subplots_adjust(wspace=.25, hspace=.35, bottom=.1,
                                 left=.1, right=.9)
        self.handles = []

        if not self.best_calculated:
            self.calculate_best()

        obs = self.xps[0].HMM.Obs(0)
        H = obs(np.eye(2))
        std = H@np.diag(np.sqrt(obs.noise.C.diag))
        std[std == 0] = np.nan

        ax = self.axes[0, 0]
        ax.set_title('Forecast x')
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            self.plot_taylor1(ax, self.truth[times, 0], xp.stats.bestf[:, 0],
                              style, std=std[0, 0])

        ax = self.axes[0, 1]
        ax.set_title('Forecast y')
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            h0 = self.plot_taylor1(ax, self.truth[times, 1], xp.stats.bestf[:, 1],
                                   style, std=std[1, -1])

        ax = self.axes[1, 0]
        ax.set_title('Analysis x')
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            self.plot_taylor1(ax, self.truth[times, 0], xp.stats.besta[:, 0],
                              style, std=std[0, 0])

        ax = self.axes[1, 1]
        ax.set_title('Analysis y')
        for xp, style in zip(self.xps, self.styles()):
            times = xp.HMM.tseq.kko
            h0 = self.plot_taylor1(ax, self.truth[times, 1], xp.stats.besta[:, 1],
                                   style, std=std[1, -1])
            self.handles.append(h0)

        self.axes[0, 0].legend(handles=self.handles, ncols=1,
                               bbox_to_anchor=(1.4, 1.05))

        for ax in self.axes.flatten():
            ax.set_xlim((0, .5*np.pi))
            ax.set_xticks(np.linspace(0, .5*np.pi, 11))
            ax.xaxis.set_major_formatter(
                lambda x, pos: np.round(1.-2*x/np.pi, 1))
            ax.set_ylabel('Correlation')

            ticks = self.nice_ticks(ax.get_ylim(), include=[0], max_ticks=8)
            ax.set_rticks(ticks)
            ax.set_xlabel('RMSE')
            ax.xaxis.set_label_coords(.5, -0.1)

    def plot_taylor1(self, ax, x1, x2, style, std=None):
        """ Add point to Taylor diagram. """
        from scipy.stats import bootstrap

        def corr2rad(c): return .5*np.pi*(1-c)

        rmse = self.rmse(x1 - x2)
        rmse_range = bootstrap((x1-x2,), self.rmse, method='percentile',
                               confidence_level=self.p_level)
        rmse_range = np.array(rmse_range.confidence_interval).reshape((2, 1))

        corr = correlation(np.array([x1, x2]), axis=-1)
        corr_range = bootstrap((np.array([x1, x2]),), correlation,
                               axis=-1, vectorized=True, method='percentile',
                               confidence_level=self.p_level)
        corr_range = np.array(corr_range.confidence_interval).reshape((2, 1))

        corr_range, corr = corr2rad(corr_range), corr2rad(corr)
        corr_range = np.abs(corr_range - corr)
        rmse_range = np.abs(rmse_range-rmse)
        ax.errorbar(corr, rmse, corr_range, rmse_range, color=style[1])
        h0, = ax.plot(corr, rmse, 'o', label=style[0], markersize=6,
                      color=style[1])

        if std is not None and not np.isnan(std):
            theta = np.linspace(0, .5*np.pi, 100)
            ax.plot(theta, std*np.ones_like(theta), 'k-')

        return h0


def correlation(x, axis=-1):
    """ Calculate correlation between x[0] and x[1]."""
    x = x - np.mean(x, axis=axis, keepdims=True)
    x2 = np.mean(x**2, axis=axis)
    x12 = np.mean(x[0]*x[1], axis=axis)
    return x12/np.sqrt(x2[0]*x2[1])


# %% Plot CDFs.
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
        if np.size(xy,1)==1:
            xy = np.concatenate((xy,np.zeros_like(xy)), axis=1)
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

        # Calculate p-value
        self.calc_kolmogorov_smirnov()

        def add_pvalue(key, style):
            style = list(style)
            mask = np.array(self.labels, dtype=str) == style[0]
            p = self.pValue[key][0][mask][0]
            style[0] = "{:s} ({:0.2f})".format(style[0], p)
            return style

        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            h00 = self.plot_cdf1(
                self.axes[0, 0], xy[:, 0], add_pvalue('x', style))
            h01 = self.plot_cdf1(
                self.axes[0, 1], xy[:, 1], add_pvalue('y', style))
            h10 = self.plot_cdf1(
                self.axes[1, 0], rt[:, 0], add_pvalue('r', style))
            h11 = self.plot_cdf1(self.axes[1, 1], np.rad2deg(rt[:, 1]),
                                 add_pvalue('theta', style))
            self.handles.append([h00, h01, h10, h11])

        self.handles = np.array(self.handles).T
        for handle, ax in zip(self.handles, self.axes.flatten()):
            ax.legend(handles=list(handle), framealpha=.7)
        self.axes[0, 0].set_xlabel('x')
        self.axes[0, 1].set_xlabel('y')
        self.axes[1, 0].set_xlabel('radius')
        self.axes[1, 1].set_xlabel('polar angle')

        for ax1 in self.axes[0]:
            self.set_nice_xlim(ax1, symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1, 0], include=[0], max_ticks=8)
        self.axes[1, 1].set_xlim(0, 360)
        self.axes[1, 1].set_xticks(np.arange(0, 361, 45))

        for ax in self.axes[:, 0]:
            ax.set_ylabel('CDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0, 1)
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

        # Calculate p-value
        self.calc_kolmogorov_smirnov()

        def add_pvalue(key, style):
            style = list(style)
            mask = np.array(self.labels, dtype=str) == style[0]
            p = self.pValue[key][0][mask][0]
            style[0] = "{:s} ({:0.2f})".format(style[0], p)
            return style

        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            h00 = self.plot_pdf1(
                self.axes[0, 0], xy[:, 0], add_pvalue('x', style))
            h01 = self.plot_pdf1(
                self.axes[0, 1], xy[:, 1], add_pvalue('y', style))
            h10 = self.plot_pdf1(
                self.axes[1, 0], rt[:, 0], add_pvalue('r', style))
            h11 = self.plot_pdf1(self.axes[1, 1], np.rad2deg(rt[:, 1]),
                                 add_pvalue('theta', style))
            self.handles.append([h00, h01, h10, h11])

        self.handles = np.array(self.handles).T
        for handle, ax in zip(self.handles, self.axes.flatten()):
            ax.legend(handles=list(handle), framealpha=.7)
        self.axes[0, 0].set_xlabel('x')
        self.axes[0, 1].set_xlabel('y')
        self.axes[1, 0].set_xlabel('radius')
        self.axes[1, 1].set_xlabel('polar angle')

        for ax1 in self.axes[0]:
            self.set_nice_xlim(ax1, symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1, 0], include=[0], max_ticks=8)
        self.axes[1, 1].set_xlim(0, 360)
        self.axes[1, 1].set_xticks(np.arange(0, 361, 45))

        for ax in self.axes[:, 0]:
            ax.set_ylabel('PDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0, 1)
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

        # Calculate p-value
        self.calc_kolmogorov_smirnov()

        def add_pvalue(key, style):
            style = list(style)
            mask = np.array(self.labels, dtype=str) == style[0]
            p = self.pValue[key][0][mask][0]
            style[0] = "{:s} ({:0.2f})".format(style[0], p)
            return style

        for xy, rt, style in zip(self.cartesian, self.polar, self.styles()):
            h0 = self.plot_cdf1(self.axes[0], xy[:, 0], add_pvalue('x', style))
            h1 = self.plot_cdf1(self.axes[1], xy[:, 1], add_pvalue('y', style))
            self.handles.append([h0, h1])

        self.handles = np.array(self.handles).T
        for handle, ax in zip(self.handles, self.axes.flatten()):
            ax.legend(handles=list(handle), framealpha=.7)
        self.axes[0].set_xlabel('z1')
        self.axes[1].set_xlabel('z2')

        self.set_nice_xlim(self.axes[0], symmetric=True, max_ticks=8)
        self.set_nice_xlim(self.axes[1], symmetric=True, max_ticks=8)

        self.axes[0].set_ylabel('CDF')
        for ax in self.axes.flatten():
            ax.set_ylim(0, 1)
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
        self.statistic = {'method': 'KS', 'x': [],
                          'y': [], 'r': [], 'theta': []}
        self.pValue = {'method': 'KS', 'x': [], 'y': [], 'r': [], 'theta': []}

        for label1, xy1 in zip(self.labels, self.cartesian):
            for label2, xy2 in zip(self.labels, self.cartesian):
                # X
                res = test(xy1[:, 0], xy2[:, 0], alternative='two-sided')
                self.statistic['x'].append(res.statistic)
                self.pValue['x'].append(res.pvalue)
                # Y
                res = test(xy1[:, 1], xy2[:, 1], alternative='two-sided')
                self.statistic['y'].append(res.statistic)
                self.pValue['y'].append(res.pvalue)

        for label1, xy1 in zip(self.labels, self.polar):
            for label2, xy2 in zip(self.labels, self.polar):
                # X
                res = test(xy1[:, 0], xy2[:, 0], alternative='two-sided')
                self.statistic['r'].append(res.statistic)
                self.pValue['r'].append(res.pvalue)
                # Y
                res = test(np.rad2deg(xy1[:, 1]), np.rad2deg(xy2[:, 1]),
                           alternative='two-sided')
                self.statistic['theta'].append(res.statistic)
                self.pValue['theta'].append(res.pvalue)

        for key in self.statistic.keys():
            if key == 'method':
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
        self.statistic = {'method': 'cramer',
                          'x': [], 'y': [], 'r': [], 'theta': []}
        self.pValue = {'method': 'cramer',
                       'x': [], 'y': [], 'r': [], 'theta': []}

        for label1, xy1 in zip(self.labels, self.cartesian):
            for label2, xy2 in zip(self.labels, self.cartesian):
                # X
                res = test(xy1[:, 0], xy2[:, 0])
                self.statistic['x'].append(res.statistic)
                self.pValue['x'].append(res.pvalue)
                # Y
                res = test(xy1[:, 1], xy2[:, 1])
                self.statistic['y'].append(res.statistic)
                self.pValue['y'].append(res.pvalue)

        for label1, xy1 in zip(self.labels, self.polar):
            for label2, xy2 in zip(self.labels, self.polar):
                # X
                res = test(xy1[:, 0], xy2[:, 0])
                self.statistic['r'].append(res.statistic)
                self.pValue['r'].append(res.pvalue)
                # Y
                res = test(np.rad2deg(xy1[:, 1]), np.rad2deg(xy2[:, 1]))
                self.statistic['theta'].append(res.statistic)
                self.pValue['theta'].append(res.pvalue)

        for key in self.statistic.keys():
            if key == 'method':
                continue
            n = len(self.labels)
            self.statistic[key] = np.array(self.statistic[key]).reshape((n, n))
            self.pValue[key] = np.array(self.pValue[key]).reshape((n, n))


# %% Plot covariances

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
                                           figsize=(8, 6))
        self.fig.subplots_adjust(wspace=.25,  left=.05, right=.95)

        # Plot principal components
        self.handles = []
        for xy, pc, angle, style in zip(self.xy, self.pc, self.angle, self.styles()):
            h = self.plot_pc1(self.axes[0], [xy, pc, angle], style)
            self.plot_ellipses1(self.axes[1], [xy, pc, angle], style)
            self.handles.append(h)

        # Complete layout
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
        for bin0, bin1 in zip(bins[:-1], bins[1:]):
            mask = np.logical_and(x >= bin0, x <= bin1)
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
        theta = rt[:, 1]
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

        # Calculate covariance matrices
        def R(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        covs = np.array(
            [R(theta1)@np.diag(pc1)@R(theta1).T for theta1, pc1 in zip(angle, pc)])

        # Average covariance matrices by first turning them into vectors
        data = np.reshape(covs, (-1, 4))
        data = np.concatenate((data, rt), axis=-1)
        bins = np.linspace(0, 2*np.pi, 25, endpoint=True)
        theta, mean, std = self.binner(rt[:, 1], data, axis=0, bins=bins)

        # Plot eclipses
        def add_ellipse(r, theta, cov):
            if np.isnan(r):
                return

            # Principal axes ecllips
            s, V = np.linalg.eig(cov)
            V = V@np.diag(s)

            # Curve ecllips
            t = np.linspace(0, 2*np.pi, 50, endpoint=True).reshape((1, -1))
            eclipse = V[:, :1] * np.cos(t) + V[:, 1:] * np.sin(t)
            eclipse[0] += r*np.cos(theta)
            eclipse[1] += r*np.sin(theta)

            # Plot ellips in polar coordinates
            ax.plot(np.arctan2(eclipse[1], eclipse[0]),
                    np.linalg.norm(eclipse, axis=0), color=color)

        for data1 in mean:
            r1, theta1 = data1[4], data1[5]
            cov1 = np.reshape(data1[:4], (2, 2))
            add_ellipse(r1, theta1, cov1)
            
#%% Plot climate distribution 

class ReconstructionPlot(BasePlots):
    
    def __init__(self, fig_dir):
        self.fig_dir = fig_dir 
        
    
    def add_samples(self, xx, zz):
        self.xx, self.zz = xx, zz 
        
    def plot_circle(self, ax, radius=1):
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), '-',
                color=(.7, .7, .7))
        
    def plot(self, fig_name='reconstruction_plot'):
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.subplots_adjust(left=.1, right=.98, wspace=.215,
                                 bottom=.1, top=.94)
        
        #Plot xx 
        ax = self.axes[0]
        self.plot_circle(ax)
        ax.plot(self.xx[:,0], self.xx[:,1], 'ko', markersize=.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_title('Climatology state')
        
        #Plot latent 
        ax = self.axes[1]
        z = np.linspace(-4,4,8*10)
        histo = scipy.stats.rv_histogram(np.histogram(self.zz, bins=z))
        z = .5*(z[1:]+z[:-1])
        norm = scipy.stats.norm(loc=np.mean(self.zz), scale=np.std(self.zz))
        ax.plot(z, norm.pdf(z), 'k-')
        ax.plot(z, histo.pdf(z), 'b-')
        
        ax.set_xlim(-4,4)
        ax.set_title('Climatology latent')
        ax.set_xlabel('z')
        ax.set_ylabel('prob(z)')
        
        for ax in self.axes:
            ax.grid()
        

# %% 2D plot of circle


class CirclePlot(BasePlots):

    def __init__(self, fig_dir):
        self.fig_dir = fig_dir
        self.labels = set()
        self.ens_for, self.ens_ana, self.tracks, self.obs = {}, {}, {}, {}
        self.latent_tracks, self.latent_for, self.latent_ana = {}, {}, {}
        self.latent_dim = 0

    def add_ens_for(self, label, times, E):
        # Store data in dict.
        self.ens_for[label] = {'time': times, 'data': E}
        self.labels = self.labels.union([label])

    def add_ens_ana(self, label, times, E):
        # Store data in dict.
        self.ens_ana[label] = {'time': times, 'data': E}
        self.labels = self.labels.union([label])
        
    def add_latent_for(self, label, times, E):
        self.latent_dim = np.size(E,-1)
        self.latent_for[label] = {'time':times, 'data':E}
        self.labels = self.labels.union([label])
        
    def add_latent_ana(self, label, times, E):
        self.latent_ana[label] = {'time':times, 'data':E}
        self.labels = self.labels.union([label])
        
    def add_track_latent(self, label, times, x):
        self.latent_tracks[label] = {'time': times, 'data': x}
        self.labels = self.labels.union([label])
        
    def add_track(self, label, times, x):
        self.tracks[label] = {'time': times, 'data': x}
        self.labels = self.labels.union([label])

    def add_obs(self, times, y):
        self.obs = {'time': times, 'data': y}

    def plot_circle(self, ax, radius=1):
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), '-',
                color=(.7, .7, .7))
        
    def add_handle(self, handle):
        if handle.get_label() in (h.get_label() for h in self.handles):
            return
        else:
            self.handles.append(handle)

    def assign_styles(self):
        styles = self.styles()
        for style in styles:
            if style[0] in self.tracks:
                self.tracks[style[0]]['style'] = style
            if style[0] in self.ens_for:
                self.ens_for[style[0]]['style'] = style
            if style[0] in self.ens_ana:
                self.ens_ana[style[0]]['style'] = style
            if style[0] in self.latent_ana:
                self.latent_ana[style[0]]['style'] = style
            if style[0] in self.latent_for:
                self.latent_for[style[0]]['style'] = style
                
    def plot_ensemble(self, ax, time, data):
        for key, value in data.items():
            mask = value['time'] == time

            # Plot forecast ensemble
            ax.plot(value['data'][mask][0, :, 0],
                    value['data'][mask][0, :, 1], 'o',
                    alpha=.2, color=value['style'][1],
                    label=value['style'][0], markeredgewidth=0)

            # Plot mean
            m = np.mean(value['data'][mask], axis=1)
            h, = ax.plot(m[0, 0], m[0, 1], 'o',
                         alpha=1., color=value['style'][1],
                         label=value['style'][0])
            
            self.add_handle(h)
            
    def plot_pdf(self, ax, time, data):
        z = np.linspace(-2,2,100)
        
        for key, value in data.items():
            mask = value['time'] == time
            data = np.sort(value['data'][mask].flatten())
            z = np.linspace(0,1,len(data))
            
            z1 = np.linspace(0,1,16)
            data1 = np.interp(z1,z,data)
            Dz1 = (z1[1:]-z1[:-1])/(data1[1:]-data1[:-1])
            Ddata1 = .5*data1[1:]+.5*data1[:-1]
            
            norm = scipy.stats.norm(loc=np.mean(data),
                                    scale=np.std(data, ddof=1))
            z = np.linspace(-2,2,100)
            ax.plot(z, norm.pdf(z), color=value['style'][1], linestyle='--')
            ax.plot(Ddata1, Dz1, color=value['style'][1], linestyle='-')

    def plot_time(self, time, fig_name='trajectory'):

        if not hasattr(self, 'axes') or self.axes is None:
            plt.close('all')
            self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 4))
            self.fig.subplots_adjust(left=.1, right=.98, wspace=.215,
                                     bottom=.1, top=.94)
            self.assign_styles()
        else:
            for ax in self.axes.flatten():
                ax.clear()

        self.fig_name = fig_name
        self.handles = []
        self.plot_circle(self.axes[0])
        
        #Plot forecast
        self.plot_ensemble(self.axes[0], time, self.ens_for)
        #Plot analysis
        self.plot_ensemble(self.axes[0], time, self.ens_ana)
        #Plot latent space
        if self.latent_dim==1:
            self.plot_pdf(self.axes[1], time, self.latent_for)
            self.plot_pdf(self.axes[1], time, self.latent_ana)
            self.axes[1].set_xlabel('latent')
            self.axes[1].set_ylabel('PDF')
        elif self.latent_dim==2:
            self.plot_ensemble(self.axes[1], time, self.latent_for)
            self.plot_ensemble(self.axes[1], time, self.latent_ana)
            self.axes[1].set_xlabel('latent_1')
            self.axes[1].set_ylabel('latent_2')
            
        # Add observation
        mask = self.obs['time'] == time
        self.axes[0].plot(np.array([1, 1])*self.obs['data']
                          [mask][0], np.array([-2, 2]), 'k--')

        # Add truth
        for key, value in self.tracks.items():
            mask = value['time'] == time
            h, = self.axes[0].plot(value['data'][mask][0, 0], 
                                   value['data'][mask][0, 1],
                                   'o', alpha=1., color=value['style'][1],
                                   label=value['style'][0])
            self.add_handle(h)
        
        ax = self.axes[0]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-2., 2.)
        ax.set_ylim(-2., 2.)
        ax.set_aspect(1)
        
        ax = self.axes[1]
        self.set_nice_xlim(ax, lims=[-2,2])
        self.set_nice_ylim(ax, minlim=0)

        for ax in self.axes.flatten():
            ax.grid()
            ax.legend(handles=self.handles, loc='upper left')
            ax.set_title('Time {:5d}'.format(time))            

    def animate_time(self, times, fig_name='movie_time', fps=30):
        """ 
        Create animation of ensemble. 
        """
        fig, ax = plt.subplots(1, 1)

        if self.fig_dir is not None:
            tmp_dir = os.path.join(self.fig_dir, 'tmp')
            os.mkdir(tmp_dir)

            print('Printing figures.')
            for it, t in enumerate(times):
                fig_name1 = os.path.join('tmp', 'frame_{:04d}'.format(it))
                self.plot_time(t, fig_name=fig_name1)
                self.save()

            print('Compiling figures into animation.')
            fmt = os.path.join(tmp_dir, 'frame_%04d')
            file_path = os.path.join(self.fig_dir, fig_name+'.mp4')
            cmd = (f'ffmpeg -f image2 -r {fps} -i {fmt} -vcodec libx264 -y '
                   f'-profile:v high444 -refs 16 -crf 0 -preset ultrafast {file_path}')
            os.system(cmd)
            shutil.rmtree(tmp_dir)

#%%

class DifferenceWeights(BasePlots):
    
    def __init__(self, fig_dir):
        self.fig_dir = fig_dir 
        
    def add_models(self,ref_model, model):
        self.ref_model, self.model = ref_model, model 
        
    def rel_weights(self, ref_layer, layer):
        w0 = ref_layer.get_weights()
        w  = layer.get_weights()
        
        rel_differences = []
        for v0, v in zip(w0,w):
            if not ref_layer.trainable:
                rel_differences.append(np.nan)
            elif not np.isclose(np.linalg.norm(v0), 0.0):
                rel_differences.append(np.linalg.norm(v0-v) / np.linalg.norm(v0))
            else:
                rel_differences.append(np.nan)
            
            
        return rel_differences
        
    def plot_difference(self, fig_name='weight_diff'):
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8))
        self.fig.subplots_adjust(left=.1, right=.98, wspace=.25,
                                 bottom=.12, top=.94, hspace=.24)
        
        self.plot_component(self.axes[0,0], self.ref_model.encoders[1],
                            self.model.encoders[1])
        self.plot_component(self.axes[0,1], self.ref_model.encoders[2],
                            self.model.encoders[2])
        self.plot_component(self.axes[1,0], self.ref_model.decoders[1],
                            self.model.decoders[1])
        self.plot_component(self.axes[1,1], self.ref_model.decoders[2],
                            self.model.decoders[2])
    
        for ax in self.axes.flatten():
            ax.grid()
        
    def plot_component(self, ax, ref_component, component):
        llist = zip(ref_component.layers, component.layers)
        data = [[n]+self.rel_weights(l0,l) for (n,(l0,l)) in enumerate(llist)]
        data = np.array([d for d in data if len(d)==3])
        
        ax.plot(data[:,0], data[:,1], 'ko', label='kernel')
        ax.plot(data[:,0], data[:,2], 'bx', label='bias')
        ax.set_title(component.name)
        ax.set_ylabel('Rel. difference')
        ax.set_xlabel('Layer number')
        ax.legend(loc='upper left')

# %% Axes


def plot_circle2d(ax, radius=1):
    """ Add circle to plot of complex plane. """
    theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-')
    ax.set_aspect(1)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid()
