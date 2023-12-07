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
from dapper.tools.seeding import set_seed
import shutil
from datetime import datetime 

#Directory in which the figures will be stored. 
FIG_DIR = '/home/ivo/Figures/vae/vae_da'
#File path used to save model 
MODEL_PATH = '/home/ivo/dpr_data/vae/circle/clima.keras'
#Number of ensemble member
Nens = 64
#Times 

def run_model(Ko, seed): 
    Dyn = {'M': 2, 'model': circle.step, 'linear': circle.step, 'noise': 0}

    # Actual observation operator.
    Obs = {'time_dependent': circle.create_obs}

    # Time steps
    dt, dko = 1, 10
    tseq = modelling.Chronology(dt=dt, K=Ko*dko, dko=dko, Tplot=Ko*dko*dt, BurnIn=0)

    # State Space System setup.
    circle.X0.seed = seed
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, circle.X0)

    # Run the model
    xx, yy = HMM.simulate()
    climate = circle.data2pandas(xx)

    return HMM, xx, yy

# %% Variational autoencoder for climatology.

HMM, xx, _ = run_model(1000, 1000)

#Create model
hypermodel = vae.DenseVae()
hp = hypermodel.build_hp(no_layers=4, no_nodes=64, use_rotation=False)
model = hypermodel.build(hp)

if False:
    tuned = vae.tune_DenseVae(xx)
    print(datetime.now())

#Fit model weights
hypermodel.fit(hp, model, xx, verbose=True)
if MODEL_PATH is not None:
    model.save(MODEL_PATH)

#Add model to HiddenMarkovModel
clima = {'hypermodel':hypermodel,'hp':hp,'clima':model}

# Sample encoder
zz_mu, zz_sig, zz = model.encoder.predict(xx[HMM.tseq.dko::HMM.tseq.dko])
zz_sig = np.exp(.5*zz_sig)

# Sample decoder
z = np.random.normal(size=(np.size(xx[HMM.tseq.dko::HMM.tseq.dko], 0), hp.get('latent_dim')))
zxx_mu, zxx_pc, zxx_angle = model.decoder.predict(z)
zxx_pc = np.exp(.5 * zxx_pc)
zxx_std = vae.rotate(zxx_pc * np.random.normal(size=np.shape(zxx_pc)), zxx_angle[:, 0])
zxx = zxx_mu + zxx_std

# %% Data assimilate

import importlib
import tensorflow as tf
importlib.reload(da)

HMM, xx, yy = run_model(10, 2000)
_, _, _ = run_model(100,3000)
xps = []

if False:
    xps.append(da.EnId(N=Nens))
    xps.append(da.EnKF('Sqrt', N=Nens, infl=1.02, rot=True))
 
#xps.append(da.EnVae(clima, N=Nens, No=128*Nens, name='EnKF-D'))
#xps.append(da.EnVae(clima, N=Nens, No=128*Nens, latent_background=True, name='VKF-B'))
#xps.append(da.EnVae(clima, N=Nens, No=128*Nens, latent_background=True, 
#                    latent_obs=True, name='VKF-BD'))

for xp in xps:
    xp.assimilate(HMM, xx, yy, liveplots=False)
    xp.xfor = np.mean(xp.stats.E.f, axis=1)
    xp.xana = np.mean(xp.stats.E.a, axis=1)

# %% Plot histograms in state space

plotState = plots.ProbPlots(FIG_DIR)
if xx is not None:
    plotState.add_series('clima', xx[HMM.tseq.dko::HMM.tseq.dko])
if zxx_mu is not None:
    plotState.add_series('clima_VAE_mean', zxx_mu)
if zxx is not None:
    plotState.add_series('clima_VAE', zxx)
for xp in xps:
    plotState.add_series(xp.name, xp.ensembles['ana'].reshape((-1,2)))
    
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
for xp in xps:
    plotLatent.add_series(xp.name, xp.ensembles['latent_ana'].reshape((-1,2)))
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



        
        



        