#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train VAE on climatology of model.

@author: ivo
"""
import numpy as np
import dapper.mods as modelling
import dapper.da_methods as da
from scipy import stats
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import I
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import basic as vae
from dapper.vae.basic import tuner
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import shutil

fig_path = '/home/ivo/Figures/vae/norot'
#if fig_path:
#    shutil.copyfile(__file__, fig_path)

# %% Run the model.

Dyn = {'M': 2, 'model': circle.step, 'linear': circle.step, 'noise': 0}

# Actual observation operator.
Obs = {'time_dependent': circle.create_obs}

# Time steps
dt, K, dko = 1, 10000, 10
tseq = modelling.Chronology(dt=dt, K=1*K, dko=dko, Tplot=K*dt, BurnIn=0*K*dt)

# State Space System setup.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, circle.X0)

# Run the model
xx, yy = HMM.simulate()
climate = circle.data2pandas(xx)
xp = None

# %% Data assimilate

xpClim = da.EnId(N=40)
xpClim.assimilate(HMM, xx, yy, liveplots=False)
xpClim.xfor = np.mean(xpClim.stats.E.f, axis=1)
xpClim.xana = np.mean(xpClim.stats.E.a, axis=1)

xp = da.EnKF('Sqrt', N=40, infl=1.02, rot=True)
xp.assimilate(HMM, xx, yy, liveplots=False)
xp.xfor = np.mean(xp.stats.E.f, axis=1)
xp.xana = np.mean(xp.stats.E.a, axis=1)

# %% Tune model

if False:
    vae.tune_DenseVae(xx)

# %% Variational autoencoder

#encoder = vae.build_encoder(2, 2, 4)
#decoder = vae.build_decoder(2, 2, 4)
#vmodel = vae.build_vae(encoder, decoder, zx_ratio=10)


def rotate(x, theta, axis=-1):
    x = np.swapaxes(x, axis, 0)
    if np.any(np.shape(theta) != np.shape(x[0])):
        raise ValueError("Shape theta does not match that of input.")
    x = np.stack((np.cos(theta)*x[0]-np.sin(theta)*x[1],
                  np.sin(theta)*x[0]+np.cos(theta)*x[1]), axis=0)
    x = np.swapaxes(x, 0, axis)
    return x


hypermodel = vae.DenseVae()
vae.hp.Fixed('no_layers', 4)
vae.hp.Fixed('no_nodes', 50)
vae.hp.Fixed('mc_samples', 20)
vae.hp.Fixed('use_rotation',True)
model = hypermodel.build(vae.hp)
hypermodel.fit(vae.hp, model, xx)

# Sample encoder
zz_mu, zz_sig, zz = model.encoder.predict(xx)
zz_sig = np.exp(.5*zz_sig)

# Sample decoder
z = np.random.normal(size=(np.size(xx[1::dko], 0), 2))
zxx_mu, zxx_pc, zxx_angle = model.decoder.predict(z)
zxx_pc = np.exp(.5 * zxx_pc)
zxx_std = rotate(zxx_pc * np.random.normal(size=np.shape(zxx_pc)), zxx_angle[:, 0])
zxx = zxx_mu + zxx_std


# %% Plot histograms

plot = plots.ProbPlots(fig_path)
if xx is not None:
    plot.add_series('clima', xx[1::dko])
if xp is not None:
    plot.add_series('analysis_EnKF', xp.xana)
if zxx_mu is not None:
    plot.add_series('clima_VAE_mean', zxx_mu)
if zxx is not None:
    plot.add_series('clima_VAE', zxx)
plot.plot_cdfs()


# %% Plot

if fig_path is not None:
    plt.close('all')
    fig, axes = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    axes = [axes]
    
    polar = circle.cartesian2polar(zxx)
    polar[:,1] = np.rad2deg(polar[:,1])
    
    #Order by increasing angle
    ind = np.argsort(polar[:,1])
    polar = polar[ind]
    v = zxx_pc[ind]
    
    bins = plots.adaptable_bins(polar[:,1])
    theta, mu, sigma= [], [], []
    for bin0,bin1 in zip(bins[:-1],bins[1:]):
        mask = np.logical_and(polar[:,1]>=bin0, polar[:,1]<=bin1) 
        theta.append(np.mean(polar[mask,1]))
        mu.append(np.mean(v[mask],axis=0))
        sigma.append(np.std(v[mask],axis=0))
    theta = np.array(theta); mu=np.array(mu); sigma=np.array(sigma)
        
    #Plot first PC
    theta = np.deg2rad(theta)
    h0,=axes[0].plot(theta, mu[:,0], label='PC_x')
    axes[0].fill_between(theta,mu[:,0]-sigma[:,0],mu[:,0]+sigma[:,0],
                         alpha=.3)
    h1,=axes[0].plot(theta, mu[:,1], label='PC_y')
    axes[0].fill_between(theta,mu[:,1]-sigma[:,1],mu[:,1]+sigma[:,1],
                         alpha=.3)
    plt.legend(handles=[h0,h1])
    
    fig.savefig(os.path.join(fig_path, 'PC_xy'),
                dpi=400, format='png')
        
        
if fig_path is not None:
    plt.close('all')
    fig, axes = plt.subplots(1, 1)
    axes = [axes]
    
    polar = circle.cartesian2polar(zxx)
    polar[:,1] = np.rad2deg(polar[:,1])
    
    #Order by increasing angle
    ind = np.argsort(polar[:,1])
    polar = polar[ind]
    v = zxx_angle[ind,0]
    
    bins = plots.adaptable_bins(polar[:,1])
    theta, mu, sigma= [], [], []
    for bin0,bin1 in zip(bins[:-1],bins[1:]):
        mask = np.logical_and(polar[:,1]>=bin0, polar[:,1]<=bin1) 
        theta.append(np.mean(polar[mask,1]))
        mu.append(np.mean(v[mask],axis=0))
        sigma.append(np.std(v[mask],axis=0))
    theta = np.array(theta); mu=np.array(mu); sigma=np.array(sigma)
    mu = np.rad2deg(np.arcsin(mu))    
    
    #Plot first PC
    axes[0].scatter(theta, mu)
    axes[0].set_xlabel('Polar angle')
    axes[0].grid('Rotation PC')
    
    fig.savefig(os.path.join(fig_path, 'polar_pc'),
                dpi=400, format='png')
    
    
