#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot single case scenarios. 
"""

import dapper.mods.Stommel.plotters as plotters
import dapper.mods.Stommel.experiment as exp
import dapper.mods.Stommel as stommel
import numpy as np

#%% Run twin experiments

data = exp.run_list('twin3', 
                observations=[[],['twin'],[],['twin'],[]],
                T_melts=np.array([np.inf,np.inf,1e4,1e4,10e2])*stommel.year,
                A_polars = [0.0,0.0,6.0,6.0,6.0],
                labels=['nCC-nDA','nCC-yDA','yCC-nDA','yCC-yDA','yCC-nDA 1000y']
                )

P = plotters.DiffPhase(data[:-1])
P.plot('fig04_diff_phase_10000.png')
P = plotters.ParameterPlot(**data[1])
P.plot('fig05_parameters_yDA.png')
P = plotters.RmseStdPlot(data[:-1])
P.plot('fig06_rmse_std.png')
P = plotters.DiffPhase(data[-1:])
P.plot_flip('fig07_diff_phase_1000.png')

#%% Run experiments with real observations.

data = exp.run_list('scenarios3',
                    observations= [stommel.hadley['yy'][1:]]*2,
                    T_melts=[np.inf, 1e4*stommel.year],
                    A_polars = [0.0, 6.0], 
                    labels=['nCC-EN4','yCC-EN4'])

#Plot output
P = plotters.ParameterPlot(**data[0])
P.plot('fig08_parameters_yCC-EN4.png')
P = plotters.EtaPlot(**data[-1])
P.plot('fig09_etas_yCC-EN4.png')
P = plotters.DiffPhase(data[:-1])
P.plot('fig10_diff_phase_nCC_EN4.png')
P = plotters.DiffPhase(data[-1:])
P.plot('fig11_diff_phase_yCC_EN4.png')
P = plotters.TransportPlot(**data[-1])
P.plot('fig12_transport_yCC-EN4.png')
    
    