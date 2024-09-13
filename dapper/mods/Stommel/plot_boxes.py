#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:25:08 2024

Plot boxes ET4

@author: ivo
"""
import os, dill
import dapper.mods.Stommel.plotters as plotters
import dapper.mods.Stommel as stommel
import matplotlib.pyplot as plt

cluster_file = os.path.join(stommel.DIR, 'clusters_boxed_hadley_inverted0906.pkl')
with open(cluster_file,'rb') as stream:
    data = dill.load(stream)
    
#%% 

plt.close('all')
boxes_plot = plotters.BoxesPlot(data['indices'], data['output'])
boxes_plot.plot('boxes.png')    
