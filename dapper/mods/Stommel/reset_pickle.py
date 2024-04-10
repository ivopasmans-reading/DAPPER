#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:07:34 2024

@author: oceancirculation059

Resets the RMSE/Standard Deviation pkl to its default, empty values
"""

import pickle as pkl
import os
import dapper.mods.Stommel as stommel

keyList = ['noWarmingNoDA','WarmingSynthDA','WarmingNoDA','noWarmingSynthDA']

param_dict = {}

for i in keyList:
    param_dict[i] = {}

param_dict['time'] = None

with open(os.path.join(stommel.DIR,'paramdict.pkl'), 'wb') as handle:
    pkl.dump(param_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

print(param_dict)