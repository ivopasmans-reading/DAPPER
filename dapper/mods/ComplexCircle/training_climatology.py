#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train VAE on climatology of model.

@author: ivo
"""
import numpy as np
import dapper.mods as modelling
import dapper.da_methods as da
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import basic as vae
import shutil

#Directory in which the figures will be stored. 
FIG_PATH = None

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

if False:
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

def rotate(x, theta, axis=-1):
    x = np.swapaxes(x, axis, 0)
    if np.any(np.shape(theta) != np.shape(x[0])):
        raise ValueError("Shape theta does not match that of input.")
    x = np.stack((np.cos(theta)*x[0]-np.sin(theta)*x[1],
                  np.sin(theta)*x[0]+np.cos(theta)*x[1]), axis=0)
    x = np.swapaxes(x, 0, axis)
    return x

hypermodel = vae.DenseVae()
hp = hypermodel.build_hp(no_layers=4, no_nodes=50, use_rotation=True)
model = hypermodel.build(hp)
hypermodel.fit(hp, model, xx)

# Sample encoder
zz_mu, zz_sig, zz = model.encoder.predict(xx[dko::dko])
zz_sig = np.exp(.5*zz_sig)

# Sample decoder
z = np.random.normal(size=(np.size(xx[dko::dko], 0), hp.get('latent_dim')))
zxx_mu, zxx_pc, zxx_angle = model.decoder.predict(z)
zxx_pc = np.exp(.5 * zxx_pc)
zxx_std = rotate(zxx_pc * np.random.normal(size=np.shape(zxx_pc)), zxx_angle[:, 0])
zxx = zxx_mu + zxx_std


# %% Plot histograms in state space

plotState = plots.ProbPlots(FIG_PATH)
if xx is not None:
    plotState.add_series('clima', xx[1::dko])
if xp is not None:
    plotState.add_series('analysis_EnKF', xp.xana)
if zxx_mu is not None:
    plotState.add_series('clima_VAE_mean', zxx_mu)
if zxx is not None:
    plotState.add_series('clima_VAE', zxx)
plotState.plot_cdfs_state(fig_name='cdfs_state')
plotState.save()

#%% Plot histogram in latent space

plotLatent = plots.ProbPlots(FIG_PATH)
if xx is not None:
    plotLatent.add_series('normal', z)
if zxx_mu is not None:
    plotLatent.add_series('clima_VAE_mean', zz_mu)
if zxx is not None:
    plotLatent.add_series('clima_VAE', zz)
plotLatent.plot_cdfs_latent(fig_name='cdfs_latent')
plotLatent.save()

#%% Plot principal components 

plotPC = plots.PrincipalPlots(FIG_PATH)
plotPC.add_series('VAE', zxx_mu, zxx_pc, zxx_angle)
plotPC.plot_pc()
plotPC.save()

#%% Save this script. 

if __name__ == '__main__' and FIG_PATH is not None:
    shutil.copyfile(__file__, FIG_PATH)


            


        