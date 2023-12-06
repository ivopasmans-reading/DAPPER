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
from datetime import datetime 

#Directory in which the figures will be stored. 
FIG_DIR = None #'/home/ivo/Figures/vae/norot'
#File path used to save model 
MODEL_PATH = '/home/ivo/dpr_data/vae/circle/clima.keras'
#Number of ensemble member
Nens = 64

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
    xpClim = da.EnId(N=Nens)
    xpClim.assimilate(HMM, xx, yy, liveplots=False)
    xpClim.xfor = np.mean(xpClim.stats.E.f, axis=1)
    xpClim.xana = np.mean(xpClim.stats.E.a, axis=1)

    xp = da.EnKF('Sqrt', N=50, infl=1.02, rot=True)
    xp.assimilate(HMM, xx, yy, liveplots=False)
    xp.xfor = np.mean(xp.stats.E.f, axis=1)
    xp.xana = np.mean(xp.stats.E.a, axis=1)

# %% Tune model

if False:
    tuned = vae.tune_DenseVae(xx)
    print(datetime.now())

# %% Variational autoencoder


#Create model
hypermodel = vae.DenseVae()
hp = hypermodel.build_hp(no_layers=4, no_nodes=64, use_rotation=False)
model = hypermodel.build(hp)

#Fit model weights
hypermodel.fit(hp, model, xx, verbose=True)
if MODEL_PATH is not None:
    model.save(MODEL_PATH)

#Add model to HiddenMarkovModel
clima = {'hypermodel':hypermodel,'hp':hp,'clima':model}

# Sample encoder
zz_mu, zz_sig, zz = model.encoder.predict(xx[dko::dko])
zz_sig = np.exp(.5*zz_sig)

# Sample decoder
z = np.random.normal(size=(np.size(xx[dko::dko], 0), hp.get('latent_dim')))
zxx_mu, zxx_pc, zxx_angle = model.decoder.predict(z)
zxx_pc = np.exp(.5 * zxx_pc)
zxx_std = vae.rotate(zxx_pc * np.random.normal(size=np.shape(zxx_pc)), zxx_angle[:, 0])
zxx = zxx_mu + zxx_std

#%% Recreate for training 

import importlib
import tensorflow as tf
importlib.reload(da)

#if MODEL_PATH is not None:
#    model = tf.keras.saving.load_model(MODEL_PATH)

xp = da.EnVae(clima, Nens, 16*Nens)
xp.assimilate(HMM, xx, yy)

# %% Plot histograms in state space

plotState = plots.ProbPlots(FIG_DIR)
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

plotLatent = plots.ProbPlots(FIG_DIR)
if xx is not None:
    plotLatent.add_series('normal', z)
if zxx_mu is not None:
    plotLatent.add_series('clima_VAE_mean', zz_mu)
if zxx is not None:
    plotLatent.add_series('clima_VAE', zz)
plotLatent.plot_cdfs_latent(fig_name='cdfs_latent')
plotLatent.save()

#%% Plot principal components 

plotPC = plots.PrincipalPlots(FIG_DIR)
plotPC.add_series('VAE', zxx_mu, zxx_pc, zxx_angle)
plotPC.plot_pc()
plotPC.save()

#%% 2D plot on circle 

plotEns = plots.CirclePlot(FIG_DIR)
plotEns.add_track('truth', HMM.tseq.tto, xx[HMM.tseq.kko])
plotEns.add_ens_for('forecast_EnKF', HMM.tseq.tto, xp.stats.E.f)
plotEns.add_ens_for('analysis_EnKF', HMM.tseq.tto, xp.stats.E.a)
plotEns.add_obs(HMM.tseq.tto, yy)
animation = plotEns.animate_time(HMM.tseq.tto, fig_name='EnKF', fps=5)

#%% Save this script. 

if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, FIG_DIR)



        
        



        