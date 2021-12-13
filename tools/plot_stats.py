import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

_XSMALL_SIZE = 12
_SMALL_SIZE = 14
_MEDIUM_SIZE = 16
_BIGGER_SIZE = 18

plt.rc('font', size=_SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=_SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=_MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=_SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=_XSMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=_BIGGER_SIZE)  # fontsize of the figure title

# Read stats from file
data = pd.read_csv("/fastscratch/mbonanni/wildfire_ml/real_test/export/stats_burn_fraction.csv", index_col=0)
name = "burn_fraction"
title = "Burned Area Fraction"
bins = np.linspace(0, 1, 21)

n = len(data)

# Compute histograms
hist = np.zeros((n, len(bins)-1))
edges = np.zeros((n+1, len(bins)))
time = np.zeros((n+1, len(bins)))
for i in range(n):
    hist[i,:], edges[i,:] = np.histogram(data.iloc[i], bins=bins)
    time[i,:] = i
time[n, :] = n
edges[n, :] = edges[n-1, :]

bound_u = data['mu'] + 2*data['sigma']
bound_l = data['mu'] - 2*data['sigma']

# Plot stats
time_vec = np.arange(0, n)
lw = 3
fig, axs = plt.subplots(1, 2, figsize=(12,5))
im = axs[0].pcolor(edges, time, hist)
plt.colorbar(im, ax=axs[0], label="Occurrences")
axs[0].set_xlabel(title)
axs[0].set_ylabel("Step")
axs[0].plot(data['mu'], time_vec, color='magenta', linewidth=lw, label="$\mu$")
axs[0].plot(data['median'], time_vec, color='r', linewidth=lw, label="median")
axs[0].plot(bound_u, time_vec, color='k', linewidth=lw, linestyle='--', label="$\mu \pm 2\sigma$")
axs[0].plot(bound_l, time_vec, color='k', linewidth=lw, linestyle='--')
axs[0].legend(loc='lower right')

hist_norm = hist / hist.max(axis=1, keepdims=True)
im = axs[1].pcolor(edges, time, hist_norm)
plt.colorbar(im, ax=axs[1], label="Occurrences / Max Occurrences")
axs[1].set_xlabel(title)
axs[1].set_ylabel("Step")
axs[1].plot(data['mu'], time_vec, color='magenta', linewidth=lw, label="$\mu$")
axs[1].plot(data['median'], time_vec, color='r', linewidth=lw, label="median")
axs[1].plot(bound_u, time_vec, color='k', linewidth=lw, linestyle='--', label="$\mu \pm 2\sigma$")
axs[1].plot(bound_l, time_vec, color='k', linewidth=lw, linestyle='--')
axs[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig("stats_" + name + ".png", bbox_inches='tight', dpi=300)

# Plot random cases
cases = data.drop(columns=['mu', 'median', 'sigma'])
fig, ax = plt.subplots()
for i in range(5):
    ax.plot(cases.iloc[:,i], label="case " + cases.columns[i])
ax.set_xlabel("Step")
ax.set_ylabel("Burned Area Fraction")
ax.legend()

plt.savefig("random_" + name + ".png", bbox_inches='tight', dpi=300)

import code; code.interact(local=locals())