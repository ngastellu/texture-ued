#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from orientation import orientation_matrix
from skued import Crystal
from time import perf_counter

def getTheta(direction):
    """Returns the azimuthal angle of a 2D or 3D vector."""
    
    direction = np.array(direction) #cast input as a NumPy array

    if direction.shape == (2,):
        x,y = direction
    elif direction.shape == (3,):
        x,y = direction[:2]
    else:
        print('Invalid input: getTheta can only take a 2D or 3D vector as an argument. Returning 0.')
        return 0
    
    theta = np.arctan2(y,x)
    if theta >= 0:
        return theta
    else:
        return theta + 2.0*np.pi



def getPhi(direction):
    """Returns the polar angle of a 3D vector."""
    x,y,z = direction
    r = np.linalg.norm(direction)
    return np.arccos(z/r)



def stereographic_projection(point):
    """Returns the polar coords of the stereographic projection of a point on the unit sphere
    whose coords are either expressed as a cartesian vector in 3D space or a pair spherical angles."""
    
    point = np.array(point) #cast point to NumPy array to avoid problems

    if point.shape[0] == 3: #point expressed as 3D cartesian vector
        x,y,z = point
        
        if z == 1:
            print('Stereographic projection of the point (0,0,1) is undefined. Returning 0 vector.')
            return np.zeros(2)
        else:
            if z > 0:
                point *= -1 #invert the coordinates wrt the origin if point is on the upper hemisphere
                x,y,z = point
            projection = np.array([x/(1-z),y/(1-z)])
            return np.array([np.linalg.norm(projection), getTheta(point)])
            #return np.array(cartesian_to_polar(projection))
    
    elif point.shape[0] == 2: #point expressed as a pair spherical angles
        theta,phi = point

        if phi%(2.0*np.pi) == 0:
            print('Stereographic projection of the point (0,0,1) is undefined. Returning 0 vector.')
            return np.zeros(2)
        else:  #we want all of the projection to lie within the unit circle
            if phi >= np.pi/2.0: #if the point lies below the xy axis, process it as is
                r = np.sin(phi)/(1-np.cos(phi))
            else: #if z>0, invert the point about the origin (direction is preserved) so that now z<0
                phi = np.pi - phi
                theta += np.pi
                r = np.sin(phi)/(1-np.cos(phi))
            return np.array([r,theta])
    else:
        print('Invalid input: Point must be given as a 3D cartesian vector or a pair of spherical angles (azimuthal, polar).\nReturning [0,0].')



def cartesian_to_polar(point):
    """Input: a point in 2D space defined with cartesian coords
       Output: the same point expressed in polar coords"""

    return np.array([np.linalg.norm(point),getTheta(point)])



def pole_figure(orientations):
    """Inputs: A set of orientation vectors described by cartesian coords or two spherical angles.
       Output: Coords of the stereographic projection of each orientation vector in polar coordinates."""
    
    radii = np.zeros(orientations.shape[0],dtype=np.float)
    thetas = np.zeros(orientations.shape[0],dtype=np.float)

    if orientations.shape[1] not in [2,3]:
        print('Invalid input: orientations must be given as a list of cartesian coordinates in 3D space or a pair of spherical angles (azimuthal,polar).\nReturning [0,0].')
        return np.zeros(2)
    else:
        radii = np.zeros(orientations.shape[0],dtype=np.float)
        thetas = np.zeros(orientations.shape[0],dtype=np.float)
        for k, pole in enumerate(orientations):
            radii[k], thetas[k] = stereographic_projection(pole)
        return np.array([radii, thetas])



# ** MAIN **
def main():
    
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
        with open('../../../data/Project3.txt') as fo:
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

            with open('pole_figure3_sph_fixed.dat','w') as fo: #save results to file
                fo.write('{0:^10}\t{1:^10}\n'.format('radius','theta'))
                for r, th in zip(radii,thetas):
                    fo.write('{0:^8.7f}\t{1:^8.7f}\n'.format(r,th))


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

if __name__ == '__main__': main()
