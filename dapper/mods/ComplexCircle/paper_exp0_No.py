#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:18:47 2024

Code for experiment 3: skewed distribution distribution. 

@author: ivo
"""

from climate import DapperModel, DaExperiment, run_exp, FIG_DIR
from vae_plots import SeriesPlots, set_styles
import dill
import os, re
import numpy as np

#Create the settings for the experiment
exp_name = "paper_exp0"
xp_parameters = {'names':['ETKF','ETKFD']}
clima_parameters = {}

exps = []
for No in np.array([1,2,4,6,8,10])*64:
    exp = DaExperiment(exp_name+"_No{:+03d}".format(int(No)), Nruns=8, Nclima=8,
                       da_model=DapperModel(obs_type=('skewedNormal',0.)),
                       xp_parameters={**xp_parameters, 'No':int(No)}, 
                       clima_parameters=clima_parameters)
    exp.No = No
    exps.append(exp)

def create_No_plot(exps, exp_name):
    plot = SeriesPlots('No', os.path.join(FIG_DIR, exp_name))
    pattern = re.compile(".*_(No[0-9]+).pkl")
    
    for exp in exps:
        with open(os.path.join(exp.filepath),'rb') as stream:
            data_for, data_ana = dill.load(stream)
            
        plot.add_exp(data_for, exp.No)
        
    plot = set_styles(plot)
    plot.plot()    

#%%

#for exp in exps:
#     run_exp(exp)

create_No_plot(exps, exp_name)

    
       
        
