#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from .. import orientation_matrix
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
