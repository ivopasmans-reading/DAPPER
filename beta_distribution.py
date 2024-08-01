#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:35:39 2024

@author: ivo
"""

import numpy as np
from scipy.stats import norm, beta 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib as mpl 


# Default settings for layout.
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['font.size'] = 12

y, sig = .9, .1

fitfunc = lambda par : np.array([(par[0]-1)/(np.sum(par)-2) - (.5*y+.5),
                                 np.prod(par)/np.sum(par)**2/(np.sum(par)+1)-sig**2*.5**2])
par = fsolve(fitfunc, np.array([2,2]))

x = np.linspace(0,1.2,200)
pnorm = norm(loc=y,scale=sig).pdf(x)
pbeta = beta(par[0],par[1]).pdf(.5*x+.5)*.5

fig = plt.figure(figsize=(6,4.5))
ax = fig.subplots(1,1)

ax.plot(x,pnorm,'b-',label='normal')
ax.plot(x,pbeta,'g--',label='beta')
ax.plot([y,y],[0,5],'k--',label='truth')
ax.grid()
ax.set_ylim(0,5)
ax.set_xlabel('observed x-coordinate')
ax.set_ylabel('probability density')
ax.legend(loc='upper left',framealpha=1)
fig.savefig('beta_distribution.png',dpi=400)