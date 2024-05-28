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
from scipy.stats import norm, bootstrap, poisson
import os
import shutil
from abc import ABC, abstractmethod
import scipy
import xarray as xr

# complex
I = complex(0, 1)

BOOT_SAMPLES = 999
CONFIDENCE_LEVEL = 0.9

# Default settings for layout.
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'


def filter_dict(key, data):
    return [data1[key] for data1 in data if key in data1]


def smod(x1, x2, *args):
    """ Modulo remainder between [-0.5x2, 0.5x2] """
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

def correlation(x, axis=-1):
    """ Calculate correlation between x[0] and x[1]."""
    x = x - np.mean(x, axis=axis, keepdims=True)
    x2 = np.mean(x**2, axis=axis)
    x12 = np.mean(x[0]*x[1], axis=axis)
    return x12/np.sqrt(x2[0]*x2[1])


def best_funcs():
    """ Subtract different variables from ensemble. """
    def xfunc(E): return E[..., 0]
    def yfunc(E): return E[..., 1]
    def rfunc(E): return np.linalg.norm(E, axis=-1)
    def tfunc(E): return np.mod(np.rad2deg(
        np.arctan2(yfunc(E), xfunc(E))), 360)
    return [xfunc, yfunc, rfunc, tfunc]


def smallest_angle(E, xx):
    """ Smallest angle between truth and ensemble members. """
    xx = np.mod(xx, 360)
    E = smod(E - xx, 360)
    E += xx
    return E, xx


class StepGenerator(object):
    """ Returns possible tick distances in decreasing order. """

    def __init__(self, factor=1):
        self.steps = np.array([.2, .25, .5, 1.], dtype=float) * factor
        self.n = len(self.steps)

    def __iter__(self):
        return self

    def __next__(self):
        self.n = self.n - 1
        if self.n < 0:
            self.steps *= 0.1
            self.n = len(self.steps) - 1
        return self.steps[self.n]


class AngularGenerator(object):
    """ Returns possible tick distances in decreasing order. """

    def __init__(self, factor=1):
        self.steps = np.array(
            [1., 5., 15., 45., 60., 90., 180.], dtype=float) * factor
        self.n = len(self.steps)

    def __iter__(self):
        return self

    def __next__(self):
        self.n = self.n - 1
        if self.n < 0:
            self.steps /= 60.
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
                   minlim=-np.inf, maxlim=np.inf, outliers=0.0, angular=False):
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
        def calc_ticks(step, lims):
            steps = np.array([np.floor(lims[0] / step),
                              np.ceil(lims[-1] / step)])
            ticks = np.arange(steps[0], steps[1]+1)*step
            return ticks

        if max_ticks < 2:
            raise ValueError('Minimum number of ticks must be 2.')

        lims = np.array(lims)
        lims = [np.nanquantile(lims, outliers),
                np.nanquantile(lims, 1.0-outliers)]
        lims = np.append(include, lims)
        lims = np.array([np.min(lims), np.max(lims)])
        lims = np.maximum(minlim, lims)
        lims = np.minimum(maxlim, lims)
        if symmetric:
            lims = np.array([-1, 1]) * np.max(np.abs(lims))

        # Width interval
        dlim = max(1e-6, np.max(lims) - np.min(lims))
        dlim = 10**np.ceil(np.log10(dlim))

        # Generator used
        if angular:
            generator = AngularGenerator(360.)
        else:
            generator = StepGenerator(dlim)

        # Find right step
        step = 0.1*dlim
        ticks0 = calc_ticks(0.1*dlim, lims)[[0, -1]]
        for step in generator:

            ticks = calc_ticks(step, lims)
            if len(ticks) > max_ticks:
                return ticks0
            else:
                ticks0 = ticks

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
        markers = ['o', 's', 'X', 'H', 'v', '^', 'P', 'D', '8']
        return zip(self.labels, colors, styles, markers)

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


class ConfidencePlots(BasePlots):
    """ 
    Plot y as function of x for different series with confidence interval. 
    """

    def __init__(self, fig_dir):
        super().__init__(fig_dir)

    def set_axes_labels(self, xlabel, clabel, reps):
        self.xlabel, self.clabel, self.reps = xlabel, clabel, reps

    def calculate_rms(self, data, level):
        data = data.transpose(*(self.xlabel, self.reps),
                              transpose_coords=True)
        data = np.array(data)**2

        mean, low, high = [], [], []
        for data1 in data:
            confidence = bootstrap((data1,), np.mean, confidence_level=level)
            low.append(confidence.confidence_interval.low)
            high.append(confidence.confidence_interval.high)
            mean.append(np.mean(data1))

        return np.sqrt(np.array(mean)), np.sqrt(np.array(low)), np.sqrt(np.array(high))

    def plot_rms(self, data, fig_name=None, level=.9):
        plt.close('all')
        self.fig, self.axes = plt.subplots(1, 1)

        if fig_name is None:
            self.fig_name = data.name+'_'+self.xlabel+'_'+self.clabel
        else:
            self.fig_name = fig_name

        self.labels = np.array(data.coords[self.clabel])
        styles = self.styles()
        xvalues = np.array(data.coords[self.xlabel])

        ax = self.axes
        for style in styles:
            mean, low, high = self.calculate_rms(data.sel({self.clabel: style[0]}),
                                                 level)
            ax.fill_between(xvalues, low, high, alpha=.3, color=style[1])
            ax.plot(xvalues, mean, color=style[1], linestyle=style[2],
                    label=style[0])

        for ax in np.reshape(self.axes, (-1,)):
            ax.grid()
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(data.name)

        self.set_nice_xlim(ax, minlim=0, include=[0])
        self.set_nice_ylim(ax, minlim=0, include=[0])
        ax.legend(loc='lower left', framealpha=1.0)

# %% Classes for processing of primal data.


class CRPS:
    """ 
    Class to calculate the Continuous Rank Probability Score. 

    See Hersbach (2000) for definitions and explanation. 
    """

    def __init__(self, ens, truth, name='', weights=1.0):
        """ 
        Class constructor. 

        ens : TxNxM numpy array 
            Array with N ensemble members at different times T. 
        truth : TxM 
            Array with truth at different times T. 
        weights : float | int | M numpy array 
            Array with weighting of different variables. 

        """

        # Ensemble members along axis=0 and time along axis=-1
        self.ens = np.transpose(ens, (1, 2, 0))
        self.truth = np.transpose(truth, (1, 0))
        self.ens = np.sort(self.ens, axis=0)
        self.N = np.size(self.ens, 0)
        self.T = np.size(self.ens, -1)
        self.weights = np.zeros(
            (1, np.size(self.ens, 1), 1)) + np.reshape(weights, (1, -1, 1))
        self.name = name

    def calculate_widths(self):
        """ 
        Average width of spacing between ensemble members. 
        """
        truth = self.truth[None, ...]
        ens = np.concatenate((truth, self.ens, truth), axis=0)

        # Width between ensemble members.
        diff = np.diff(ens, axis=0)
        # Average width.
        mask = diff > 0.0
        diff = np.nanmean(self.weights*diff, axis=-1, where=mask)
        w = np.nanmean(self.weights + np.zeros_like(mask), axis=-1,
                       where=mask)
        diff = np.where(~np.isnan(w), diff/w, 0.0)

        return diff

    def calculate_indicator(self):
        """
        Average fraction of spacing smaller than the truth. 
        """
        # H(x-x_truth)
        beta = self.ens[1:] - np.maximum(self.truth[None, ...], self.ens[:-1])
        beta = np.nanmean(np.maximum(beta, 0.0) * self.weights, axis=-1)
        beta /= np.nanmean(self.weights, axis=-1)

        # Width between ensemble members.
        diff = np.maximum(self.ens[1:] - self.ens[:-1],
                          np.finfo(float).resolution)
        diff = np.nanmean(self.weights * diff, axis=-1)
        diff /= np.nanmean(self.weights, -1)

        # Probability smaller than ensemble.
        under = self.truth[None, ...] < self.ens[:1]
        under = np.mean(under*self.weights, axis=-1)
        under /= np.mean(self.weights, axis=-1)

        # Probability larger than ensemble.
        over = self.truth[None, ...] > self.ens[-1:]
        over = np.mean(over*self.weights, axis=-1)
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
        ranks = np.sum(self.ens <= self.truth[None, ...], axis=0)
        rank_count = np.zeros((self.N+1, np.size(ranks, -1)))
        for rank in bins:
            rank_count[rank, :] = np.sum(ranks == rank, axis=0)

        # Calculate confidence interval
        P = poisson(M)
        dp = 0.5*(1-p)
        low, high = P.isf(1-dp), P.isf(dp)

        return rank_count / M, (low/M, high/M)

    def calculate_crps(self, widths, indicators):
        probs = np.arange(0, self.N+1) / self.N
        probs = np.reshape(probs, (-1, 1))

        crps = (1-indicators) * probs**2
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

        crps = xr.DataArray(crps[0], name=self.name,
                            dims={'variable': 1},
                            coords={'variable': ('variable', ['crps'])})
        reliability = xr.DataArray(reliability[0], name=self.name,
                                   dims={'variable': 1},
                                   coords={'variable': ('variable', ['reliability'])})
        resolution = xr.DataArray(resolution[0], name=self.name,
                                  dims={'variable': 1},
                                  coords={'variable': ('variable', ['resolution'])})
        uncertainty = xr.DataArray(uncertainty, name=self.name,
                                   dims={'variable': 1},
                                   coords={'variable': ('variable', ['uncertainty'])})
        result = xr.merge([crps, reliability, resolution, uncertainty])

        return result


class Histogram:
    """ 
    Create histogram of output ensemble members as function of truth.
    """

    def __init__(self, ens, truth, name):
        """ 
        Class constructor. 

        ens : TxNxM numpy array 
            Array with N ensemble members at different times T. 
        truth : TxM 
            Array with truth at different times T. 
        weights : float | int | M numpy array 
            Array with weighting of different variables. 

        """

        # Ensemble members along axis=0
        self.ens = ens
        self.N = np.size(self.ens, 1)
        self.T = np.size(self.ens, 0)
        self.truth = truth
        self.name = name

    @staticmethod
    def get_bins(name):
        # Bins for histogram
        if name == 'radius':
            return np.arange(-.05, 1.451, .1)
        elif name == 'angle':
            return np.arange(0, 360.1, 15)
        else:
            return np.arange(-1.45, 1.451, .1)

    def __call__(self):
        truth = np.reshape(self.truth, (self.T, 1, -1))
        truth = truth * np.ones((1, self.N, 1))
        # Bin to calculate the probability density.
        bins = self.get_bins(self.name)
        H = np.histogram2d(truth.ravel(), self.ens.ravel(), bins, density=True)
        # Centre of bins.
        bin_range = np.max(bins) - np.min(bins)
        bins = .5*bins[:-1] + .5*bins[1:]
        data = xr.DataArray(H[0],
                            dims={'true '+self.name: len(bins),
                                  'ensemble '+self.name: len(bins)},
                            coords={'true '+self.name: ('true '+self.name, bins),
                                    'ensemble '+self.name: ('ensemble '+self.name, bins),
                                    },
                            name=self.name,
                            attrs={'bin range': bin_range})
        return data


class EnsError:
    """ 
    Calculate error of different ensemble members. 
    """

    def __init__(self, ens, truth, name):
        """ 
        Class constructor. 

        ens : TxNxM numpy array 
            Array with N ensemble members at different times T. 
        truth : TxM 
            Array with truth at different times T. 
        weights : float | int | M numpy array 
            Array with weighting of different variables. 

        """

        # Ensemble members along axis=0
        self.N = np.size(ens, 1)
        self.T = np.size(ens, 0)
        self.ens = np.transpose(ens, (1, 2, 0))
        self.truth = np.transpose(truth, (1, 0))[None, ...]
        self.name = name

    def __call__(self):

        output = xr.Dataset()
        for var in ['truth', 'ensemble', 'error']:
            dims = {'member': self.N, 'variable': 1, 'metric': 1}
            coords = {'member': ('member', range(self.N)),
                      'variable': ('variable', [var])}

            if var == 'truth':
                values = self.truth * np.ones((self.N, 1, 1))
            elif var == 'ensemble':
                values = self.ens
            elif var == 'error':
                values = self.ens - self.truth
            values = np.reshape(values, (np.size(values, 0), -1))

            # Calculate bias
            mean = np.mean(values, axis=-1)
            mean = np.reshape(mean, (self.N, 1, 1))
            mean = xr.DataArray(mean, name=self.name, dims=dims,
                                coords={**coords, 'metric': ('metric', ['mean'])})
            # Calculate variance
            variance = np.var(values, ddof=1, axis=-1)
            variance = np.reshape(variance, (self.N, 1, 1))
            variance = xr.DataArray(variance, name=self.name, dims=dims,
                                    coords={**coords, 'metric': ('metric', ['variance'])})
            # Calculate covariance
            truth = np.reshape(self.truth, (-1,))
            E = np.reshape(values, (self.N, -1))
            rho = [np.cov(e, truth)[1, 0] for e in E]
            rho = np.reshape(rho, (self.N, 1, 1))
            rho = xr.DataArray(rho, name=self.name, dims=dims,
                               coords={**coords, 'metric': ('metric', ['covariance'])})
            # Add to output
            output = xr.merge([output, mean, variance, rho])

        return output


def calculate_stat(stat, xp, xx, seed, stage='analysis', **kwargs):
    """ Calculate statistics for each of the 4 variables x,y,radius angle. """
    results = {}
    xx = xx[xp.HMM.tseq.kko]
    results = xr.Dataset()

    for func, varname in zip(best_funcs(), ['x', 'y', 'radius', 'angle']):
        # Truth
        x = func(xx)[..., None]
        # Ensemble
        if 'ana' in stage:
            E = func(xp.stats.E.a)[..., None]
        elif 'for' in stage:
            E = func(xp.stats.E.f)[..., None]
        # Adjust ensemble to get smallest angle to truth.
        if varname == 'angle':
            E, x = smallest_angle(E, x[..., None])
            x = x[:, :, 0]
        # Calculate the stat.
        result = stat(E, x, varname, **kwargs)()
        result = result.expand_dims(dim={'experiment': 1, 'seed': 1})
        result = result.assign_coords({'experiment': ('experiment', [xp.name]),
                                       'seed': ('seed', [seed])})
        result = result.assign_attrs({'stage': stage})

        # Combine
        results = xr.merge([results, result])

    return results

# %% Classes to generate plots.


class ProbDensityPlots(BasePlots):
    """ Plot probability density as function of true value. """

    def __init__(self, fig_dir, data):
        super().__init__(fig_dir)
        self.data = data
        self.variables = [key for key in self.data.keys()]

    def calculate_condition_prob(self):
        # Calculate conditional probs.
        output = xr.Dataset()
        for variable in self.variables:
            cond = self.data[variable]
            cond = cond.mean(dim='seed')
            cond = cond / cond.sum(dim=['ensemble '+variable])
            cond = cond / self.data[variable].attrs['bin range']
            cond = cond.where(cond > 0.0)

            cmin = np.array(cond.quantile(.01, dim=[
                            'ensemble '+variable, 'true '+variable, 'experiment'], skipna=True))
            cmax = np.array(cond.quantile(.99, dim=[
                            'ensemble '+variable, 'true '+variable, 'experiment'], skipna=True))
            cticks = self.nice_ticks(
                np.log10(np.array([cmin, cmax])), max_ticks=4)
            cond = cond.assign_attrs(
                {**self.data[variable].attrs, 'cticks': 10**cticks})

            output = xr.merge([output, cond])

        return output

    def plot_scatter_density(self, fig_name='ensemble_true'):
        # Number of experiments
        N_xp = len(self.data.coords['experiment'])
        # Conditional probability
        data = self.calculate_condition_prob()

        # Create figure
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(
            N_xp, 4, figsize=(8.5, min(11, 1+2*N_xp)))
        if np.ndim(self.axes) == 1:
            self.axes = np.reshape(self.axes, (1, -1))
        self.fig.subplots_adjust(wspace=.1, hspace=.02, left=.1, right=.98,
                                 bottom=.14, top=.98)

        # Plot each prob. plots
        im = {}
        for xp, axes in zip(np.array(data.coords['experiment']), self.axes):
            for ax, variable in zip(axes, ['x', 'y', 'radius', 'angle']):
                # Select data for this experiment and this variable.
                data1 = data[variable].sel(experiment=xp)
                # Bin centres.
                y, x = np.meshgrid(data1['true '+variable],
                                   data1['ensemble '+variable])
                # Plot
                norm = mpl.colors.LogNorm(np.min(data1.attrs['cticks']),
                                          np.max(data1.attrs['cticks']))
                levels = self.nice_ticks(
                    np.log10(data1.attrs['cticks']), max_ticks=11)
                levels = 10**levels
                im[variable] = ax.pcolormesh(x, y, np.array(data1), cmap='afmhot_r',
                                             norm=norm,
                                             )

        # Set layout axes.
        for ax in np.reshape(self.axes[:, :2], (-1,)):
            self.set_nice_xlim(ax, max_ticks=5)
            self.set_nice_ylim(ax, max_ticks=5)
            ax.set_xlim(np.array([-.5, .5])*data['x'].attrs['bin range'])
            ax.set_ylim(np.array([-.5, .5])*data['x'].attrs['bin range'])
        for ax in np.reshape(self.axes[:, 2], (-1,)):
            self.set_nice_xlim(ax, max_ticks=5, minlim=0.0)
            self.set_nice_ylim(ax, max_ticks=5, minlim=0.0)
        for ax in np.reshape(self.axes[:, 3], (-1,)):
            self.set_nice_xlim(ax, max_ticks=5, angular=True)
            self.set_nice_ylim(ax, max_ticks=5, angular=True)
        for ax, xp in zip(self.axes[:, 0], np.array(data.coords['experiment'])):
            ax.set_ylabel('ensemble '+xp)
        for ax, variable in zip(self.axes[-1, :], self.variables):
            ax.set_xlabel('true '+variable)
        for ax in np.reshape(self.axes[:-1, :], (-1,)):
            ax.set_xticklabels([])
        for ax in np.reshape(self.axes[:, 1:], (-1,)):
            ax.set_yticklabels([])
        for ax, variable in zip(np.reshape(self.axes[-1, :], (-1,)), self.variables):
            bbox = ax.get_position()
            cax = self.fig.add_axes([bbox.x0, .06, bbox.width, .02])
            cbar = plt.colorbar(im[variable], cax, orientation='horizontal',
                                extend='min')
        for ax in np.reshape(self.axes, (-1,)):
            ax.plot(ax.get_xlim(), ax.get_ylim(), 'k-')
            ax.set_aspect(1)
            ax.grid('on')


class CrpsPlots(BasePlots):
    """ 
    Plot CRPS, reliability and resolution. 
    """

    def __init__(self, fig_dir, data):
        super().__init__(fig_dir)
        self.data = data

    def calculate_mean(self, data):
        axis = [n for n, dim in enumerate(data.dims) if dim == 'seed']
        boot = bootstrap([np.array(data.data)], np.mean, axis=axis[0],
                         vectorized=True, n_resamples=BOOT_SAMPLES,
                         confidence_level=CONFIDENCE_LEVEL)

        mean = data.mean(dim='seed')
        low = mean.copy(data=boot.confidence_interval.low)
        high = mean.copy(data=boot.confidence_interval.high)

        return xr.Dataset({'mean': mean, 'low': low, 'high': high})

    def plot_crps(self, fig_name='crps'):
        """ 
        Plot CRPS, reliability and resolution. 
        """

        # Create figure
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8.5, 6))
        self.fig.subplots_adjust(wspace=.14, hspace=.24, left=.08, right=.98,
                                 bottom=.14, top=.92)

        keys = list(self.data.keys())
        xps = np.array(self.data.coords['experiment'])
        for ax, key in zip(self.axes.ravel(), keys):
            self.plot_crps1(ax, self.data[key])
            ax.set_title(key)

        for ax in self.axes.ravel():
            self.set_nice_ylim(ax, include=[0.0], minlim=0.0)
            ax.set_xticklabels([])
            ax.set_xticks(range(len(xps)))
            ax.set_xlim(-.5, len(xps)-.5)
            ax.grid()
        for ax in self.axes[-1]:
            ax.xaxis.set_tick_params(rotation=30)
            ax.set_xticklabels(xps)

        self.axes[-1, 0].legend(loc='upper right', framealpha=1.0)

    def plot_crps1(self, ax, data):
        xps = np.array(data.coords['experiment'])

        # Plot
        self.labels = ['crps', 'reliability', 'resolution']
        width = np.linspace(-.5,.5,len(self.labels)+1)*.6
        width = .5*width[1:]+.5*width[:-1]
        for n,style in enumerate(self.styles()):
            varname = style[0]
            data1 = data.sel(variable=varname)
            data1 = self.calculate_mean(data1)

            mean = np.array(data1['mean'].data)
            low = mean - np.array(data1['low'].data)
            high = np.array(data1['high'].data) - mean
            
            #ax.errorbar(range(len(data1.coords['experiment'])), mean,
            #            np.array([low, high]), linewidth=0.0, elinewidth=2.0,
            #            color=style[1], marker=style[3], label=varname, capsize=3)
            ax.bar(range(len(data1.coords['experiment'])) + width[n],
                   mean, yerr=np.array([low,high]), label=varname,
                   width = np.diff(width[:2]))

class TaylorPlots(BasePlots):
    """ 
    Plot Taylor diagrams of mean values.
    """

    def __init__(self, fig_dir, data):
        super().__init__(fig_dir)
        self.data = data

    def calculate_mean(self, data, weights):
        EPS = 1e-6
        func = lambda x,y,axis: np.sum(x,axis=axis)/(EPS+np.sum(y,axis=axis))
        data  = (np.array(data.data),np.array(weights.data))
        data  = np.reshape(data, (2,-1))
        boot  = bootstrap(data,
                          func, n_resamples=BOOT_SAMPLES,
                          confidence_level=CONFIDENCE_LEVEL,
                          vectorized=True, axis=0, paired=True)
        result = func(*data,axis=0)
        
        return result, boot.confidence_interval.low, boot.confidence_interval.high

    def cor2rad(self, cor):
        return .5*np.pi*(1-cor)
    
    def rad2cor(self, rad):
        return (.5*np.pi-rad) / (.5*np.pi)
    
    def rmse(self, cor, std, std0):
        return np.sqrt(std**2+std0**2-2*std*std0*cor)

    def plot_taylor(self, fig_name='taylor'):
        # Create figure
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(1,4, figsize=(8, 3.2),
                                           subplot_kw={'projection': 'polar'})
        self.axes = np.reshape(self.axes,(1,-1))
        self.fig.subplots_adjust(wspace=0.2, hspace=0.08, bottom=.15,
                                 left=.05, right=.95,top=.9)
        self.handles = []

        #Plot output experiments. 
        for ax, key in zip(self.axes.ravel(), list(self.data.keys())):
            self.plot_taylor2(ax, self.data[key])
            ax.set_title(key)

        #Add legend. 
        lax = self.fig.add_axes([.5, .01, .9, .1])
        lax.axis('off')
        lax.legend(handles=self.handles, ncols=4, framealpha=1.0,
                   loc='lower center', bbox_to_anchor=[0.0, 0.0])

        #Layout axes.
        for ax in self.axes.flatten():
            ax.set_xlim((self.cor2rad(1.05), self.cor2rad(-.05)))
            ax.set_xticks(self.cor2rad(np.linspace(0,1,6)))
            ax.xaxis.set_major_formatter(
                lambda x, pos: np.round(self.rad2cor(x), 1))
           
            self.set_nice_ylim(ax, include=[0.0],max_ticks=5)
            ax.xaxis.set_label_coords(.5, -0.15)
            ax.set_xlabel('')
        
        #Axes labels
        for ax in self.axes[-1,:]:
            ax.set_xlabel('standard deviation')
        for ax in self.axes[:,0]:
            ax.set_ylabel('correlation')

    def interval(self, data):
        mean = np.array(data['mean'].data)
        low = np.array(mean - data['low'])
        low = np.where(np.isnan(low), 0.0, low)
        high = np.array(data['high']-mean)
        high = np.where(np.isnan(high), 0.0, high)
        return mean, low, high

    def plot_taylor2(self, ax, data):
        # Experiments
        self.labels = np.array(data.coords['experiment'])

        self.handles = []
        for n, style in enumerate(self.styles()):
            ssEE = data.sel(experiment=style[0], variable='ensemble', 
                            metric='variance')
            ssTT = data.sel(experiment=style[0],
                            variable='truth', metric='variance')
            covET = data.sel(experiment=style[0], variable='ensemble', 
                             metric='covariance')
            ssET = data.sel(experiment=style[0], variable='ensemble', 
                            metric='variance')**.5
            ssET *= data.sel(experiment=style[0],
                            variable='truth', metric='variance')**.5

            #Take expectation value
            sTT = np.array(self.calculate_mean(ssTT, np.ones_like(ssTT)))**.5
            sEE = np.array(self.calculate_mean(ssEE, np.ones_like(ssEE)))**.5
            corET = np.array(self.calculate_mean(covET, ssET))
            print('ET',style[0],corET, sTT, sEE)
            #Plot point
            corET = self.cor2rad(corET)
            
            #Truth 
            cor = np.linspace(-.05, 1.05, 100)
            r = np.ones_like(cor) * sTT[0]
            h, = ax.plot(self.cor2rad(cor), r, 'k-', label='truth')
            
            #Plot
            h, = ax.plot(corET[0], sEE[0], style[3], label=style[0],
                         color=style[1])
            self.handles.append(h)

            #Plot rmse
            rticks = self.nice_ticks(ax.get_ylim(), max_ticks=5, include=[0])
            r,cor = np.meshgrid(np.linspace(0,max(rticks),100),
                                np.linspace(-.05,1.05,100))
            rmse = self.rmse(cor,r,sTT[0])
            levels = self.nice_ticks(rmse,max_ticks=12,include=[0.0])
            contour = ax.contour(self.cor2rad(cor),r,rmse,
                                 levels=levels,colors=[(0,0,0)],linewidths=1,
                                 linestyles=['--'])
            ax.clabel(contour,levels[::2], fontsize=10)
            
            #Plot point
            ax.errorbar(corET[0], sEE[0],
                        xerr=np.array([[corET[0]-min(corET)], 
                                       [max(corET)-corET[0]]]),
                        yerr=np.array([[sEE[0]-min(sEE)],
                                       [max(sEE)-sEE[0]]]),
                        label=style[0], color=style[1], marker=style[3])

    def plot_taylor1(self, ax, data):

        # Calculate confidence intervals
        mean = self.calculate_mean(data)
        # Standard deviation
        sel = {'variable': 'RMSE'}
        std0, stdL, stdH = self.interval(mean.sel(sel))
        # Correlation
        sel = {'variable': 'correlation'}
        corr0, corrL, corrH = self.interval(mean.sel(sel))
        corr0 = .5*np.pi * (1-corr0)
        corrL, corrH = .5*np.pi*corrH, .5*np.pi*corrL

        # Experiments
        self.labels = np.array(data.coords['experiment'])

        self.handles = []
        for n, style in enumerate(self.styles()):
            ax.errorbar(corr0[n], std0[n],
                        xerr=np.array([corrL[n:n+1], corrH[n:n+1]]),
                        yerr=np.array([stdL[n:n+1], stdH[n:n+1]]),
                        label=style[0], color=style[1], marker=style[3])
            h, = ax.plot(corr0[n], std0[n], style[3], label=style[0],
                         color=style[1])
            self.handles.append(h)


# %% Needs revision

class EnsStatsPlots(BasePlots):
    """ 
    Class that plots statistics of analysis ensemble DA. 
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

    def add_truth(self, HMM, truth):
        """ Add truth for calculating errors. """
        self.HMM = HMM
        self.truth = truth

    def add_xp(self, xp):
        """ Add experiment for processing. """
        self.labels.append(xp.name)
        self.xps.append(xp)

    def plot_crps1(self, axes, E, xx, style, mode='f'):
        times = self.HMM.tseq.kko
        crps = CRPS(E, xx)
        stats = crps().data

        for stat1, ax1 in zip(stats, axes[:-1]):
            h0, = ax1.plot(times, stat1, label=style[0], color=style[1],
                           linestyle=style[2])

        axes[-1].plot(times, stats[-1]*np.ones_like(times), 'k--',
                      label='uncertainty')

        return h0

    def calculate_crps(self):
        results = {}
        times = self.HMM.tseq.kko

        xp_names = [xp.name for xp in self.xps]
        results = []

        for func, varname in zip(self.best_funcs(), ['x', 'y', 'radius', 'angle']):
            xx = func(self.truth)[times]
            xx = xx[None, ...]

            results1 = []
            for xp in self.xps:
                E = func(xp.stats.E.a)[..., None]
                N = np.size(E, 1)
                E = np.reshape(np.swapaxes(E, 0, 2), (1, N, -1))

                if varname == 'angle':
                    E = smod(E - xx[None, ...], 360)
                    E += xx[None, ...]

                crps = CRPS(E, xx)()
                crps = crps.expand_dims(dim={'experiment': 1, 'variable': 1})
                crps = crps.assign_coords({'variable': ('variable', [varname]),
                                           'experiment': ('experiment', [xp.name])})
                results1.append(crps)
                results.append(xr.concat(results1, dim='experiment'))

        results = xr.concat(results, dim='variable')

        return results

    def calculate_rmse(self, fig_name='rmse'):
        results = {}
        times = self.HMM.tseq.kko

        results = []
        for func, varname in zip(self.best_funcs(), ['x', 'y', 'radius', 'angle']):
            xx = func(self.truth)[times]
            xx = xx[None, ...]

            for xp in self.xps:
                E = func(xp.stats.E.a)[..., None]
                N = np.size(E, 1)
                E = np.reshape(np.swapaxes(E, 0, 2), (1, N, -1))

                # Ensemble mean
                if varname == 'angle':
                    angle_mean = np.exp(I*E)
                    angle_mean = np.prod(angle_mean)**(1/N)
                    angle_mean = np.imag(np.log(angle_mean))
                    value = angle_mean + 0
                else:
                    value = np.mean(E)
                mean = xr.DataArray([[value]], name='ensemble_mean',
                                    dims=('experiment', 'variable'),
                                    coords={'experiment': ('experiment', [xp.name]),
                                            'variable': ('variable', [varname])})

                # Ensemble var
                if varname == 'angle':
                    angle_var = np.exp(I*E) / np.exp(I*angle_mean)
                    angle_var = np.imag(np.log(angle_var))
                    value = np.mean(angle_var**2)
                else:
                    value = np.var(E, ddof=1)
                var = xr.DataArray([[value]], name='ensemble_variance',
                                   dims=('experiment', 'variable'),
                                   coords={'experiment': ('experiment', [xp.name]),
                                           'variable': ('variable', [varname])})

                # RMSE
                if varname == 'angle':
                    angle_rmse = np.exp(I*E) / np.exp(I * xx[None, :, :])
                    angle_rmse = np.imag(np.log(angle_rmse))
                    value = self.rmse(angle_rmse)
                else:
                    value = self.rmse(E)
                rmse = xr.DataArray([[value]], name='rmse',
                                    dims=('experiment', 'variable'),
                                    coords={'experiment': ('experiment', [xp.name]),
                                            'variable': ('variable', [varname])})

                # Bias
                value = np.mean(E)
                bias = xr.DataArray([[value]], name='bias',
                                    dims=('experiment', 'variable'),
                                    coords={'experiment': ('experiment', [xp.name]),
                                            'variable': ('variable', [varname])})

                results += [mean, var, rmse, bias]

        for xp in self.xps:
            E = xp.stats.E.a
            xx = self.truth[times]
            N = np.size(E, 1)
            E = np.reshape(np.swapaxes(E, 0, 1), (N, -1, np.size(xx, -1)))
            error = E - xx[None, ...]
            rmse = np.sqrt(np.mean(error**2))

            rmse = xr.DataArray([[rmse]], name='rmse',
                                dims=('experiment', 'variable'),
                                coords={'experiment': ('experiment', [xp.name]),
                                        'variable': ('variable', ['position'])})
            results.append(rmse)

        results = xr.merge(results)
        return results

    def calculate_best(self):
        for xp in self.xps:
            xp.stats.besta = np.array([ens2ml(E) for E in xp.stats.E.a])
            xp.stats.bestf = np.array([ens2ml(E) for E in xp.stats.E.f])

        self.best_calculated = True

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

    def write_crps(self, fig_name='stats'):
        # Data
        results = self.calculate_crps()
        # Formatting
        header = "{:10s}" + 4*" {:>12s}"
        fmt = "{:10s}" + 4*" {:12.3f}" + '\n'
        # Write
        with open(os.path.join(self.fig_dir, fig_name+'.txt'), 'w') as stream:
            # Write header
            stream.write(header.format('', "CRPS", "Reliability",
                         "Resolution", "Uncertainty\n"))
            # Write for each variable
            for varname, xps in results.items():
                stream.write('Variable '+varname+'\n')
                for xp, values in xps.items():
                    stream.write(fmt.format(xp, *stats))

    def write_rmse(self, fig_name='rmse'):
        # Data
        results = self.calculate_rmse()
        # Formatting
        header = "{:10s}" + 2*" {:>12s}"
        fmt = "{:10s}" + 2*" {:12.3f}" + '\n'
        # Write
        with open(os.path.join(self.fig_dir, fig_name+'.txt'), 'w') as stream:
            # Write header
            stream.write(header.format('', "Bias", "RMSE\n"))
            # Write for each variable
            for varname, xps in results.items():
                stream.write('Variable '+varname+'\n')
                for xp, values in xps.items():
                    stream.write(fmt.format(xp, *stats))

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

    def plot_rmse_enssize_xp(self, data, fig_name='rmse_enssize_xp',
                             level=0.9):
        """ Plot RMSE as function ensemble size for different experiments. """

        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(1, 1, figsize=(8, 4))

        results, intervals = {}, {}

        data = filter_dict('position', data)
        for name, xp in data[0].items():
            data_xp = filter_dict(name, data)
            data_xp = np.array(data_xp)[:, 0]
            data_xp = np.random.normal(size=np.shape(data_xp))
            data_xp = np.array([data_xp])
            print(data_xp)

            result = bootstrap(data_xp, self.rmse, n_resamples=1000, vectorized=True,
                               axis=0, confidence_level=level, method='percentile')
            print('LOW', result.confidence_interval)


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
        if np.size(xy, 1) == 1:
            xy = np.concatenate((xy, np.zeros_like(xy)), axis=1)
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

# %% Plot climate distribution


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

        # Plot xx
        ax = self.axes[0]
        self.plot_circle(ax)
        ax.plot(self.xx[:, 0], self.xx[:, 1], 'ko', markersize=.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title('Climatology state')

        # Plot latent
        ax = self.axes[1]
        z = np.linspace(-4, 4, 8*10)
        histo = scipy.stats.rv_histogram(np.histogram(self.zz, bins=z))
        z = .5*(z[1:]+z[:-1])
        norm = scipy.stats.norm(loc=np.mean(self.zz), scale=np.std(self.zz))
        ax.plot(z, norm.pdf(z), 'k-')
        ax.plot(z, histo.pdf(z), 'b-')

        ax.set_xlim(-4, 4)
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
        self.latent_dim = np.size(E, -1)
        self.latent_for[label] = {'time': times, 'data': E}
        self.labels = self.labels.union([label])

    def add_latent_ana(self, label, times, E):
        self.latent_ana[label] = {'time': times, 'data': E}
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
        z = np.linspace(-2, 2, 100)

        for key, value in data.items():
            mask = value['time'] == time
            data = np.sort(value['data'][mask].flatten())
            z = np.linspace(0, 1, len(data))

            z1 = np.linspace(0, 1, 16)
            data1 = np.interp(z1, z, data)
            Dz1 = (z1[1:]-z1[:-1])/(data1[1:]-data1[:-1])
            Ddata1 = .5*data1[1:]+.5*data1[:-1]

            norm = scipy.stats.norm(loc=np.mean(data),
                                    scale=np.std(data, ddof=1))
            z = np.linspace(-2, 2, 100)
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

        # Plot forecast
        self.plot_ensemble(self.axes[0], time, self.ens_for)
        # Plot analysis
        self.plot_ensemble(self.axes[0], time, self.ens_ana)
        # Plot latent space
        if self.latent_dim == 1:
            self.plot_pdf(self.axes[1], time, self.latent_for)
            self.plot_pdf(self.axes[1], time, self.latent_ana)
            self.axes[1].set_xlabel('latent')
            self.axes[1].set_ylabel('PDF')
        elif self.latent_dim == 2:
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
        self.set_nice_xlim(ax, lims=[-2, 2])
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

# %%


class DifferenceWeights(BasePlots):
    """ Plot differences in training weights. """

    def __init__(self, fig_dir):
        self.fig_dir = fig_dir

    def add_models(self, ref_model, model):
        self.ref_model, self.model = ref_model, model

    def rel_weights(self, ref_layer, layer):
        w0 = ref_layer.get_weights()
        w = layer.get_weights()

        rel_differences = []
        for v0, v in zip(w0, w):
            if not ref_layer.trainable:
                rel_differences.append(np.nan)
            elif not np.isclose(np.linalg.norm(v0), 0.0):
                rel_differences.append(
                    np.linalg.norm(v0-v) / np.linalg.norm(v0))
            else:
                rel_differences.append(np.nan)

        return rel_differences

    def plot_difference(self, fig_name='weight_diff'):
        plt.close('all')
        self.fig_name = fig_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8))
        self.fig.subplots_adjust(left=.1, right=.98, wspace=.25,
                                 bottom=.12, top=.94, hspace=.24)

        self.plot_component(self.axes[0, 0], self.ref_model.encoders[1],
                            self.model.encoders[1])
        self.plot_component(self.axes[0, 1], self.ref_model.encoders[2],
                            self.model.encoders[2])
        self.plot_component(self.axes[1, 0], self.ref_model.decoders[1],
                            self.model.decoders[1])
        self.plot_component(self.axes[1, 1], self.ref_model.decoders[2],
                            self.model.decoders[2])

        for ax in self.axes.flatten():
            ax.grid()

    def plot_component(self, ax, ref_component, component):
        llist = zip(ref_component.layers, component.layers)
        data = [[n]+self.rel_weights(l0, l)
                for (n, (l0, l)) in enumerate(llist)]
        data = np.array([d for d in data if len(d) == 3])

        ax.plot(data[:, 0], data[:, 1], 'ko', label='kernel')
        ax.plot(data[:, 0], data[:, 2], 'bx', label='bias')
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
