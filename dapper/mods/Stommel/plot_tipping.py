#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot probability tipping.
"""

import dapper.mods.Stommel.plotters as plotters
from dapper.mods.Stommel.experiment import run_hypercube
import dapper.mods.Stommel as stommel
import numpy as np
import dill,os
import matplotlib.pyplot as plt
import dapper.mods.Stommel.plotters as plotter

#%% Run experiment with different levels of forcing. 

#data = run_hypercube('tipping3', T_melts=np.arange(400,4001,400)*stommel.year,
#                      A_polars = np.arange(0,7.1,.5))

#%% Run experiments with real observations.

filepath = os.path.join(stommel.fig_dir,'tipping3.pkl')
with open(filepath,'rb') as stream:
    data = dill.load(stream)

#%% 

plt.close('all')
P = plotter.TippingPlot(data)
P.plot(filename='fig12_tipping_percentage')

