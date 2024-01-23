#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train VAE on climatology of model.

@author: ivo
"""
import numpy as np
import dapper.mods as modelling
import dapper.da_methods as da
import dapper.da_methods.ensemble as eda
from dapper.mods import ComplexCircle as circle
from dapper.mods.ComplexCircle import vae_plots as plots
from dapper.vae import basic as vae
from dapper.tools.seeding import set_seed
import shutil, os
from datetime import datetime 

#Directory in which the figures will be stored. 
FIG_DIR = None #'/home/ivo/Figures/vae/vae_B/dko10_sigo01'
#File path used to save model 
MODEL_PATH = '/home/ivo/dpr_data/vae/circle/clima.keras'
#Number of ensemble member
Nens = 64
dko,sigo = 10, .1

if __name__ == '__main__' and FIG_DIR is not None:
    shutil.copyfile(__file__, os.path.join(FIG_DIR,'training_climatology.py'))

def run_model(K, dko, seed): 
    Dyn = {'M': 2, 'model': circle.step, 'linear': circle.step, 'noise': 0}

    # Actual observation operator.
    obs = circle.create_obs_factory([0],sigo)
    Obs = {'time_dependent': obs}

    # Time steps
    dt = 1
    tseq = modelling.Chronology(dt=dt, K=K, dko=dko, Tplot=K*dt, BurnIn=0)

    # State Space System setup.
    circle.X0.seed = seed
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, circle.X0)

    # Run the model
    xx, yy = HMM.simulate()
    climate = circle.data2pandas(xx)

    return HMM, xx, yy

# %% Variational autoencoder for climatology.

HMM0, xx0, _ = run_model(10000, 1, 1000)
if FIG_DIR is not None:
    plotState = plots.ProbPlots(FIG_DIR)
    plotState.add_series('clima', xx0)
    plotState.plot_cdfs_state(fig_name='cdfs_clima_full')
    plotState.save()
    plotState.plot_pdfs_state(fig_name='pdfs_clima_full')
    plotState.save()

#Create model
hypermodel = vae.DenseVae()
hp = hypermodel.build_hp(no_layers=4, no_nodes=64, use_rotation=False)
model = hypermodel.build(hp)

if False:
    tuned = vae.tune_DenseVae(xx0)
    print(datetime.now())

#Fit model weights
hypermodel.fit(hp, model, xx0, verbose=True)
if MODEL_PATH is not None:
    model.save(MODEL_PATH)

#Add model to HiddenMarkovModel
vae_trans = eda.VaeTransform(hypermodel, hp, model)

# Sample encoder
zz_mu, zz_sig, zz = model.encoder.predict(xx0[HMM0.tseq.dko::HMM0.tseq.dko])
zz_sig = np.exp(.5*zz_sig)

# Sample decoder
z = np.random.normal(size=(np.size(xx0[HMM0.tseq.dko::HMM0.tseq.dko], 0), hp.get('latent_dim')))
zxx_mu, zxx_pc, zxx_angle = model.decoder.predict(z)
zxx_pc = np.exp(.5 * zxx_pc)
zxx_std = vae.rotate(zxx_pc * np.random.normal(size=np.shape(zxx_pc)), zxx_angle[:, 0])
zxx = zxx_mu + zxx_std

# %% Data assimilate

import importlib
import tensorflow as tf
importlib.reload(da)

HMM, xx, yy = run_model(1000, dko, 2000)
_, _, _ = run_model(1000, dko, 3000)

factory = eda.EndaFactory()

xps = []
xps.append(eda.EnDa(Nens, [], name='no DA'))
xps.append(da.EnKF('Sqrt', Nens, name='ETKF', rot=True))
xps.append(factory.build(Nens, 'EnKF_D','',No=128*Nens, name='EnKF_D'))
xps.append(factory.build(Nens, 'EnKF_D', No=128*Nens, name='VKF-clima', 
                        VaeTransforms = [vae_trans]))
xps.append(factory.build(Nens, 'EnKF_D', No=128*Nens, name='VKF-B', 
                        VaeTransforms = [vae_trans]))

for xp in xps:
    xp.assimilate(HMM, xx, yy, liveplots=False)
    xp.HMM = HMM
    
#%% Plot principal components 

plotPC = plots.PrincipalPlots(FIG_DIR)
plotPC.add_series('VAE', zxx_mu, zxx_pc, zxx_angle)
plotPC.plot_pc()
plotPC.save()
    
# %% Plot histograms in state space

plotState = plots.ProbPlots(FIG_DIR)
if xx is not None:
    plotState.add_series('truth', xx[HMM.tseq.dko::HMM.tseq.dko])
#if zxx_mu is not None:
#    plotState.add_series('clima_VAE_mean', zxx_mu)
if zxx is not None:
    plotState.add_series('clima_VAE', zxx)
for xp in xps:
    plotState.add_series(xp.name, xp.stats.E.a.reshape((-1,2)))
    
plotState.plot_cdfs_state(fig_name='cdfs_state')
plotState.save()

#%% Plot histogram in latent space

plotLatent = plots.ProbPlots(FIG_DIR)
if xx is not None:
    plotLatent.add_series('normal', z)
#if zxx_mu is not None:
#    plotLatent.add_series('clima_VAE_mean', zz_mu)
if zxx is not None:
    plotLatent.add_series('clima_VAE', zz)
for xp in xps:
    if not hasattr(xp, 'processors'):
        continue 
    for process in xp.processors:
        if hasattr(process, 'Elatent'):
            E = np.array(process.Elatent['f'])
            plotLatent.add_series(xp.name, E.reshape((-1,2)))
plotLatent.plot_cdfs_latent(fig_name='cdfs_latent')
plotLatent.save()

#%% Plot best estimates. 

importlib.reload(plots)
            
plotStats = plots.EnsStatsPlots(FIG_DIR)
plotStats.add_truth(HMM, xx)
for xp in xps:
    print('Adding ',xp.name)
    plotStats.add_xp(xp)

plotStats.plot_taylor()
plotStats.save()

plotStats.plot_best()
plotStats.save()

plotStats.plot_rmse_std()
plotStats.save()

plotStats.plot_rmse_std_variables()
plotStats.save()



#%% 2D plot on circle 

if False:
    plotEns = plots.CirclePlot(FIG_DIR)
    plotEns.add_track('truth', HMM.tseq.tto, xx[HMM.tseq.kko])
    plotEns.add_ens_for('forecast_EnKF', HMM.tseq.tto, xp.stats.E.f)
    plotEns.add_ens_for('analysis_EnKF', HMM.tseq.tto, xp.stats.E.a)
    plotEns.add_obs(HMM.tseq.tto, yy)
    animation = plotEns.animate_time(HMM.tseq.tto, fig_name='EnKF', fps=5)



        
        



        