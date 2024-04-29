#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train VAE on climatology of model.

@author: ivo
"""
import tensorflow as tf
import importlib
import numpy as np
import dapper.mods as modelling
import dapper.da_methods as da
import dapper.da_methods.ensemble as eda
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import basic as vae
from dapper.tools.seeding import set_seed
import shutil
import os
from datetime import datetime
from sklearn import preprocessing
import scipy

# Directory in which the figures will be stored.
FIG_DIR = '/home/ivo/Figures/vae/vae_obs/test'
# File path used to save model
MODEL_PATH = '/home/ivo/dpr_data/vae/circle/clima.keras'
# Number of ensemble member
Nens = 64
dko, sigo = 10, .1

sigo = np.deg2rad(5)

if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR, 'training_climatology.py'))


def run_model(K, dko, seed, obs_type, amplitude=0.0):
    Dyn = {'M': 2, 'model': circle.step_factory(amplitude=amplitude), 
           'linear': circle.step_factory(amplitude=amplitude), 'noise': 0}

    # Actual observation operator.
    obs = circle.create_obs_factory([0], sigo, obs_type)
    Obs = {'time_dependent': obs}

    # Time steps
    dt = 1
    tseq = modelling.Chronology(dt=dt, K=K, dko=dko, Tplot=K*dt, BurnIn=0)

    # State Space System setup.
    circle.X0.seed = seed
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, circle.X0)

    # Run the model
    circle.set_seed(seed)
    xx, yy = HMM.simulate()
    climate = circle.data2pandas(xx)

    return HMM, xx, yy

# %% Variational autoencoder for climatology.

from matplotlib import pyplot as plt
from scipy.stats import norm, rv_histogram

#Generate climatology
HMM0, xx0, _ = run_model(10000, 1, 1000, 'beta', amplitude=0.0) 


# Create model
hypermodel = vae.DenseVae()
hp = hypermodel.build_hp(no_layers=6, no_nodes=32, use_rotation=False,
                         batch_size=32, latent_dim=1, mc_samples=1)
model = hypermodel.build(hp)
inno_model = hypermodel.build_inno(hp, model, 1)

# Fit model weights
hypermodel.fit(hp, model, xx0, verbose=True, shuffle=True)
if MODEL_PATH is not None:
    model.save(MODEL_PATH)

# Add model to HiddenMarkovModellatent_dim = self.hp.values['latent_dim']
vae_trans = eda.VaeTransform(hypermodel, hp, model)
cycle_trans = eda.CyclingVaeTransform(hypermodel, hp, None)
bkg_trans = eda.BackgroundVaeTransform(hypermodel, hp, model)
inno_trans = eda.InnoVaeTransform(hypermodel, hp, model, 4096, HMM0.Obs(0).noise.add_sample)

# Sample encoder
model.set_trainable=False
samples = xx0[HMM0.tseq.dko::HMM0.tseq.dko]
zz_mu, zz_sig, zz = model.encoder.predict(samples)
zz_sig = np.exp(.5*zz_sig)

# Sample decoder
z = np.random.normal(size=(np.size(samples, 0), hp.get('latent_dim')))
zxx_mu, zxx_sig, zxx_angle, zxx = model.decoder.predict(z)
zxx_sig = np.exp(.5*zxx_sig)

#Plot distributions
plotReconstruction = plots.ReconstructionPlot(FIG_DIR)
plotReconstruction.add_samples(zxx, zz)
plotReconstruction.plot()
plotReconstruction.save()

# %% Data assimilate

importlib.reload(da)
from copy import deepcopy

HMM, xx, yy = run_model(dko, dko, 2000, 'beta', amplitude=0.0)
HMM1, xx1, yy1 = run_model(dko, dko, 2000, 'normal', amplitude=0.0)
_, _, _ = run_model(dko, dko, 3000, 'beta')

factory = eda.EndaFactory() 

xps = []
xps.append(eda.EnDa(Nens, [], name='no DA'))
xps.append(factory.build(Nens, 'Sqrt svd', name='ETKF', rot=False))
#xps.append(factory.build(Nens, 'ETKF_D', '',
#           No=1*Nens, name='EnKF_D1', rot=False))
#xps.append(factory.build(Nens, 'ETKF_D', '',
#           No=64*Nens, name='EnKF_D64', rot=False))
xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='VKF-clima',
                         VaeTransforms=[vae_trans]))
xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='VKF-clima-obs',
                         VaeTransforms=[inno_trans,vae_trans]))
#xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='VKF-cycle',
#                         VaeTransforms=[cycle_trans]))
#xps.append(factory.build(Nens, 'ETKF_D', No=64*Nens, name='VKF-trans',
#                         VaeTransforms=[bkg_trans]))

for xp in xps:
    print("Assimilating ",xp.name)
    xp.assimilate(HMM, xx, yy, liveplots=False)
    xp.HMM = HMM
    
for xp in xps:
    print("Assimilating ",xp.name)
    xp.assimilate(HMM1, xx1, yy1, liveplots=False)
    xp.HMM1 = HMM1

# %%

plotCRPS = plots.EnsStatsPlots(FIG_DIR)
plotCRPS.add_truth(HMM, xx)
for xp in xps:
    plotCRPS.add_xp(xp)
plotCRPS.plot_crps()
plotCRPS.calculate_crps()

# %% Plot principal components

plotPC = plots.PrincipalPlots(FIG_DIR)
plotPC.add_series('VAE', zxx_mu, zxx_pc, zxx_angle)
plotPC.plot_pc()
plotPC.save()

#%% Difference weights 

plotDiff = plots.DifferenceWeights(FIG_DIR)

xp = [xp for xp in xps if xp.name=='VKF-trans']
if len(xp)>0:
    plotDiff.add_models(xp[0].processors[-2].ref_model, 
                        xp[0].processors[-2].model)
    plotDiff.plot_difference('weights_trans')
    plotDiff.save()
    

xp = [xp for xp in xps if xp.name=='VKF-cycle']
if len(xp)>0:
    plotDiff.add_models(xp[0].processors[-2].ref_model, 
                        xp[0].processors[-2].model)
    plotDiff.plot_difference('weights_cycle')
    plotDiff.save()

# %% Plot histograms in state space

if FIG_DIR is not None:
    plotState = plots.ProbPlots(FIG_DIR)
    plotState.add_series('clima', xx0)
    plotState.plot_cdfs_state(fig_name='cdfs_clima_full')
    plotState.save()
    plotState.plot_pdfs_state(fig_name='pdfs_clima_full')
    plotState.save()

plotState = plots.ProbPlots(FIG_DIR)
if xx is not None:
    plotState.add_series('truth', xx[HMM.tseq.dko::HMM.tseq.dko])
# if zxx_mu is not None:
#    plotState.add_series('clima_VAE_mean', zxx_mu)
if zxx is not None:
    plotState.add_series('clima_VAE', zxx)
for xp in xps:
    plotState.add_series(xp.name, xp.stats.E.a.reshape((-1, 2)))

plotState.plot_cdfs_state(fig_name='cdfs_state')
plotState.save()

# %% Plot histogram in latent space

plotLatent = plots.ProbPlots(FIG_DIR)
if z is not None:
    plotLatent.add_series('normal', z)
# if zxx_mu is not None:
#    plotLatent.add_series('clima_VAE_mean', zz_mu)
if zz is not None:
    plotLatent.add_series('clima_VAE', zz)
for xp in xps:
    if not hasattr(xp.stats, 'Elatent'):
        continue
    E = np.array(xp.stats.Elatent['f'])
    plotLatent.add_series(xp.name, E.reshape((-1, 2)))
plotLatent.plot_cdfs_latent(fig_name='cdfs_latent')
plotLatent.save()

# %% Plot best estimates.

importlib.reload(plots)

plotStats = plots.EnsStatsPlots(FIG_DIR)
plotStats.add_truth(HMM, xx)
for xp in xps:
    print('Adding ', xp.name)
    plotStats.add_xp(xp)

plotStats.plot_taylor()
plotStats.save()

plotStats.plot_best()
plotStats.save()

plotStats.plot_rmse_std()
plotStats.save()

plotStats.plot_rmse_std_variables()
plotStats.save()

# %% 2D plot on circle

for xp in xps:
    plotEns = plots.CirclePlot(FIG_DIR)
    plotEns.add_track('truth', HMM.tseq.tto, xx[HMM.tseq.kko])
    plotEns.add_ens_for(xp.name+' for', HMM.tseq.tto, xp.stats.E.f)
    plotEns.add_ens_for(xp.name+' ana', HMM.tseq.tto, xp.stats.E.a)
    plotEns.add_obs(HMM.tseq.tto, yy)
    
    if hasattr(xp.stats, 'Elatent'):
        plotEns.add_latent_for(xp.name+' for', HMM.tseq.tto, xp.stats.Elatent['f'])
        plotEns.add_latent_ana(xp.name+' ana', HMM.tseq.tto, xp.stats.Elatent['a'])
        
    animation = plotEns.animate_time(HMM.tseq.tto, fig_name=xp.name, fps=3)
