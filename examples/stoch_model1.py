# ## Reproduce basics of `bib.grudzien2020numerical`, and do live plotting.
# TODO: more description.

# #### Imports
# <b>NB:</b> If you're on <mark><b>Gooble Colab</b></mark>,
# then replace `%matplotlib notebook` below by
# `!python -m pip install git+https://github.com/nansencenter/DAPPER.git` .
# Also note that liveplotting does not work on Colab.

# %matplotlib notebook
from mpl_tools import is_notebook_or_qt as nb

import dapper as dpr
import dapper.da_methods as da

# #### Load experiment setup: the hidden Markov model (HMM)

from dapper.mods.Lorenz96s.grudzien2020 import HMMs

# #### Generate the same random numbers each time this script is run

seed = dpr.set_seed(3000)

# #### Simulate synthetic truth (xx) and noisy obs (yy)

xx, yy = HMMs().simulate()

# The model trajectories only use half the integration steps
# of the true trajectory (compare `t5`, `t10`). Subsample `xx`
# so that the time series are compatible for comparison.

xx = xx[::2]

# #### Specify a DA method configuration ("xp" for "experiment")

xp = da.EnKF('PertObs', N=100)

# #### Assimilate yy, knowing the HMM; xx is used to assess the performance

xp.assimilate(HMMs("RK4"), xx, yy, liveplots=not nb)

# #### Average the time series of various statistics; print some averages

xp.stats.average_in_time()
print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))

# #### The above used the Runge-Kutta scheme. Repeat it, but with Euler-Maruyama

xp.assimilate(HMMs("EM"), xx, yy, liveplots=not nb)
xp.stats.average_in_time()
print(xp.avrgs.tabulate(['rmse.a', 'rmv.a']))