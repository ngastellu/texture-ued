#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt

def cos_phi(r):
    """Input: radial coordinate of a pole figure point
       Output: cosine of the polar angle of its corresponding point on the unit sphere"""

    return ( (r**2) - 1 )/( 1 + (r**2) )


def sin_phi(r):
    """Input: radial coordinate of a pole figure point
       Output: sine of the polar angle of its corresponding point on the unit sphere"""

    return (2*r)/( 1 + (r**2) )


with open('../data/pole_figure3_sph_fixed.dat') as fo:
    radii, thetas = np.array([list(map(float,fline.split())) for fline in fo.readlines()[1:]]).T

r_bins = np.linspace(0,1,30)
theta_bins = np.linspace(0,2*np.pi,60)

tt, rr = np.meshgrid(theta_bins,r_bins)

print(tt.shape)
print(rr.shape)

dtheta = np.array([theta_bins[k+1] - theta_bins[k] for k in range(theta_bins.shape[0]-1)])
dphi = np.array([-cos_phi(r_bins[k]) + cos_phi(r_bins[k+1]) for k in range(r_bins.shape[0] - 1)])

dt, dph = np.meshgrid(dtheta,dphi)
solid_angles = dt*dph/(4.0*np.pi)
print(solid_angles[0])

heatmap, th_edges, r_edges = np.histogram2d(thetas,radii,bins=(theta_bins,r_bins))

print(heatmap.shape)

#th, r = np.mgrid[0:2*np.pi:60j,0:1.0:30j]

plt.figure(figsize=(7,7))
plt.subplot(projection='polar')

plt.pcolormesh(tt,rr,heatmap.T/solid_angles)

plt.plot(tt,rr,ls='none',c='k')

plt.colorbar()
plt.grid()
plt.show()
