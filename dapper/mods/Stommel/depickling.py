import pickle as pkl
import dapper.mods.Stommel as stommel
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
import numpy as np

fig_dir = stommel.fig_dir

with open(os.path.join(fig_dir,'melt_values.pkl'), 'rb') as handle:
    b = pkl.load(handle)

print(b)

fig, ax = plt.subplots()
ax.set_xlabel("Melt Period (years)")
ax.set_ylabel("Yearly Temperature Warming (C)")
mm = b['mm']
tt = b['tt']
probs = b['probs']
cont1 = ax.contourf(mm,tt,probs, levels = np.arange(0,101,1),norm=cm.colors.Normalize(vmax=100, vmin=0), cmap = cm.bwr)
clb = fig.colorbar(cont1, ax=ax, orientation='vertical', fraction=.1)
clb.set_ticks(np.arange(0, 101, 20))
clb.set_label('Percent of Ensemble Flipping',labelpad = 10, rotation = 270)
fig.savefig(os.path.join(fig_dir,'melt_clima_probs'),format='png',dpi=300)
