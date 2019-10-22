#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from ..orientation import orientation_matrix

def getTheta(direction,change_range=False):
    """Returns the azimuthal angle of a 3D vector."""
    x,y,z = direction
    theta = np.arctan2(y,x)
    if change_range == False or theta>=0:
        return theta
    else:
        return theta + 2*np.pi
    
def getPhi(direction):
    """Returns the polar angle of a 3D vector."""
    x,y,z = direction
    r = np.linalg.norm(direction)
    return np.arccos(z/r)

def stereographic_cartesian(point):
    """Returns the cartesian coords of the stereographic projection of a point on the unit sphere
    whose coords are also expressed in cartesian format"""
    x,y,z = point

    if z == 1:
        print('Stereographic projection of the pole (0,0,1) is undefined. Returning 0 vector.')
        return np.zeros(3)
    else:
        return np.array([x/(1-z),y/(1-z)])

def stereographic_polar(point):
    """Returns the polar coords of the stereographic projection of a point on the unit sphere
    whose position is determined by angles theta (azimuthal) and phi (polar)."""

    theta,phi = point

    if phi%(2.0*np.pi) == 0:
        print('Stereographic projection of the pole (0,0,1) is undefined. Returning 0 vector.')
        return np.zeros(3)
    else:
        r = np.sin(phi)/(1-np.cos(phi))
        return np.array([r,theta])

# ** MAIN **

with open('../../../data/Project3.txt') as fo:
    flines = fo.readlines()
    nlines = len(flines)
    print(nlines)
    angles = np.zeros((nlines,2))
    poles = np.zeros((nlines,3))
    euler_angles = np.zeros((nlines,3))
    fo.seek(0)
    for k,line in enumerate(fo):
        eul_ang1,eul_ang2,eul_ang3 = list(map(float,line.split(';')[2:-1])) #recover euler angles from EBSD data
        euler_angles[k] = [eul_ang1,eul_ang2,eul_ang3]
        pole = orientation_matrix(eul_ang1,eul_ang2,eul_ang3).T[2]
        poles[k,:] = pole/np.linalg.norm(pole)
        angles[k,0] = getTheta(pole,change_range=False)
        angles[k,1] = getPhi(pole)


#``with open('angles2.dat','w') as fo:
#``    fo.write('Azimuthal\tPolar\n')
#``    for angle_vec in angles:
#``        fo.write('{0[0]:8.7f}\t{0[1]:8.7f}\n'.format(angle_vec))



interesting_indices = np.all(angles==np.zeros(2),axis=1).nonzero()[0]

#problematic_poles = np.array([poles[k] for k in interesting_indices])
#problematic_angles = np.array([euler_angles[k] for k in interesting_indices])

#print(np.all(problematic_angles==0))

good_indices = np.all(angles!=np.zeros(2),axis=1).nonzero()[0]
good_angles = np.array([angles[k] for k in good_indices])

thetas = good_angles[:,0]
phis = good_angles[:,1]

#plt.hist(thetas,bins=200)

heatmap, xedges, yedges = np.histogram2d(thetas,phis,bins=200)
hist_extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]

delta_thetas = np.array([xedges[k+1]-xedges[k] for k in range(xedges.shape[0]-1)])
delta_cos_phis = np.array([np.cos(yedges[k+1])-np.cos(yedges[k]) for k in range(yedges.shape[0]-1)])

solid_angle_matrix = np.array([delta_thetas,delta_cos_phis]).T

theta_grid = np.linspace(-np.pi,np.pi,199)
phi_grid = np.linspace(0,np.pi,199)
full_grid = np.meshgrid(theta_grid,phi_grid)

#plt.imshow(solid_angle_matrix,extent=(theta_grid[0],theta_grid[-1]))
#plt.show()
#
#
#final_map = heatmap/solid_angle_map

plt.rc('text',usetex=True)

#plt.figure(figsize=(10,5))
#n, bins, patches = plt.hist(thetas,100)
#plt.xlabel(r'$\theta$ [radians]')
#plt.ylabel('Counts')
#plt.show()

plt.figure(figsize=(10,5))
n, bins, patches = plt.hist(thetas,1000)
plt.xlabel(r'$\varphi$ [radians]')
plt.ylabel('Counts')
plt.show()

max_count = np.max(n)
max_index = np.asarray(n == max_count).nonzero()[0][0]
print(max_index)

print(bins[max_index],bins[max_index+1])

#plt.figure()
#img = plt.imshow(heatmap.T,extent=hist_extent)
#plt.colorbar(img,label='counts')
#plt.xlabel(r'$\theta$ [radians]')
#plt.ylabel(r'$\phi$ [radians]')
#plt.show()
