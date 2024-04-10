#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 02:51:35 2024

@author: oceancirculation059

Plots the RMSE and Standard Deviations for each key quantity
from each of the twin experiments against one another. For example, it produces
a graph of the RMSE of the Polar Temperature in each of the twin
experiments over the 100 year period.
"""

import pickle as pkl
import os
import matplotlib.pyplot as plt
import dapper.mods.Stommel as stommel

dict_DIR = os.path.join(stommel.DIR,'paramdict.pkl')

names = ['Polar Temp [C]','Equatorial Temp [C]','Polar Salinity [ppt]'
         ,'Equatorial Salinity [ppt]', 'Surface Temp Flux Coefficient [mms-1]',
         'Surface Salinity Flux Coefficient [mms-1]','Advective Transport Flux Coefficient [ms-1]']

with open(dict_DIR, 'rb') as handle:
    param_dict = pkl.load(handle)

#print(param_dict['noWarmingSynthDA']['RMSE'][3]-param_dict['WarmingSynthDA']['RMSE'][3])

for n in range(7):
    fig, ax = plt.subplots(1,2, figsize=(8,4),constrained_layout=True)
    for ax1 in ax:
        ax1.grid()
        ax1.set_xlabel('Time [year]')
        ax1.axvline(x=2023, color='k', linestyle='--')
    ax[0].set_ylabel(f'SD {names[n]}')
    ax[1].set_ylabel(f'RMSE {names[n]}')
    for exp_type in ['noWarmingNoDA','WarmingNoDA','noWarmingSynthDA','WarmingSynthDA']:
        ax[0].plot(param_dict['time'],param_dict[exp_type]['Spread'][n])
        ax[1].plot(param_dict['time'],param_dict[exp_type]['RMSE'][n])
    fig.savefig(f'{stommel.fig_dir}/errors/{n}',
                 format='png',dpi=600)
    