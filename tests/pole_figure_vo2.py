#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from texture import orientation_matrix, pole_figure
from skued import Crystal
from time import perf_counter



    
vo2 = Crystal.from_database('vo2-m1')
reciprocal_matrix = np.array(vo2.reciprocal_vectors).T #matrix of reciprocal lattice vectors arranged into columns

pole_input = input('Miller indices of pole being projected (space-separated): ')
pole_indices = np.array(list(map(int,pole_input.split())))
unrotated_pole = reciprocal_matrix @ pole_indices

run_type = input('Spherical or cartesian? ') #Determines whether or not poles will be translated to spherical coords before
#run_type = 'both'                              #the stereographic projection
if run_type.lower().rstrip().lstrip() not in ['cartesian','spherical','both']:
    print('Invalid input. Acceptable run types are \'spherical\',\'cartesian\', or \'both\'.')

else:
    global_start = perf_counter()
    with open('../data/Project3.txt') as fo:
        flines = fo.readlines()
        nlines = len(flines)
        if run_type in ['spherical','both']:
            angles = np.zeros((nlines,2))
        poles = np.zeros((nlines,3))
        euler_angles = np.zeros((nlines,3))
        fo.seek(0)
        for k,line in enumerate(fo):
            eul_ang1,eul_ang2,eul_ang3 = list(map(float,line.split(';')[2:-1])) #recover euler angles from EBSD data
            euler_angles[k] = [eul_ang1,eul_ang2,eul_ang3]
            if np.all(euler_angles[k]==0):
                continue
            o_matrix = orientation_matrix(*euler_angles[k])
            pole = o_matrix @ unrotated_pole
            poles[k,:] = pole/np.linalg.norm(pole)
            if run_type in ['spherical','both']:
                angles[k,0] = getTheta(pole)
                angles[k,1] = getPhi(pole)


#    with open('../data/angles2.dat','w') as fo:
#        fo.write('Azimuthal\tPolar\n')
#        for angle_vec in angles:
#            fo.write('{0[0]:8.7f}\t{0[1]:8.7f}\n'.format(angle_vec))

    #with open('angles2.dat') as fo:
    #    fo.readline() #move buffer to 2nd line (skips 1st line)
    #    angles = np.array([list(map(float,line.split())) for line in fo])



    if run_type == 'both':
        good_indices_spherical = np.all(angles!=np.zeros(2),axis=1).nonzero()[0]
        print(good_indices_spherical.shape)
        good_poles_spherical = np.array([angles[k] for k in good_indices_spherical])
        radii_spherical, thetas_spherical = pole_figure(good_poles_spherical)
        data_spherical = np.array([radii_spherical,thetas_spherical]).T

        good_indices_cartesian = np.any(euler_angles!=0,axis=1).nonzero()[0]
        print(good_indices_cartesian.shape)
        good_poles_cartesian = np.array([poles[k] for k in good_indices_cartesian])
        radii_cartesian, thetas_cartesian = pole_figure(good_poles_cartesian)
        data_cartesian = np.array([radii_cartesian,thetas_cartesian]).T

        print('Good indices: ', np.all(good_indices_spherical == good_indices_cartesian)) #prints True
        
        common_pts = np.zeros((data_cartesian.shape[0],2),dtype=np.float) #take advantage of the fact that both arrays are the same size
        only_spherical = np.zeros((data_cartesian.shape[0],2),dtype=np.float)
        for k, sph in enumerate(data_spherical):
            if np.any(np.all(data_cartesian==sph,axis=1)): #check if sph is also in data_cartesian
                common_pts[k] = sph
            else:
                only_spherical[k] = sph

    else: 
        good_indices = np.any(euler_angles!=0,axis=1).nonzero()[0]
        print(good_indices.shape)
        good_poles = np.array([poles[k] for k in good_indices])
        print(good_poles.shape)

        radii, thetas = pole_figure(good_poles)

        #with open('pole_figure3_sph_fixed.dat','w') as fo: #save results to file
        #    fo.write('{0:^10}\t{1:^10}\n'.format('radius','theta'))
        #    for r, th in zip(radii,thetas):
        #        fo.write('{0:^8.7f}\t{1:^8.7f}\n'.format(r,th))


#  Plot the resulting pole figure
    plt.axes(projection='polar')
    
    if run_type in ['cartesian','spherical']:
        plt.polar(thetas,radii,'ro',ms=0.08)
    else:
        plt.polar(thetas_cartesian,radii_cartesian,'ro',ms=0.08,label='cartesian',alpha=0.5)
        plt.polar(thetas_spherical,radii_spherical,'bo',ms=0.08,label='spherical',alpha=0.5)
        #plt.polar(only_spherical[:,1],only_spherical[:,0],'bo',ms=0.08,label='spherical')
        #plt.legend()
    
    plt.show()
