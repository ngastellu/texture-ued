#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt

with open('../../data/pole_figure3_sph_fixed.dat') as fo:
    radii, thetas = np.array([list(map(float,fline.split())) for fline in fo.readlines()[1:]]).T

r_bins = np.linspace(0,1,50)
theta_bins = np.linspace(0,2*np.pi,70)

tt, rr = np.meshgrid(theta_bins,r_bins)

pole_counts, th_edges, r_edges = np.histogram2d(thetas,radii,bins=(theta_bins,r_bins))

plt.figure(figsize=(7,7))
plt.subplot(projection='polar')

plt.pcolormesh(tt,rr,pole_counts.T)

plt.colorbar()
plt.grid()
plt.show()
