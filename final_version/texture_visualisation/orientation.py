#!/usr/bin/env pythonw

import numpy as np
from skued import Crystal, electron_wavelength, affine
from math import gcd
from itertools import combinations_with_replacement, permutations
from time import perf_counter


def orientation_matrix(euler1,euler2,euler3):
    """Input: Three Euler angles (Bunge convention), expressed in degrees
       Output: Cartesian coordinates of the orientation direction"""
    e1, e2, e3 = np.eye(3) #standard basis (sample coord system)
    sample_basis = np.array([e1,e2,e3])
    
    deg2rad = np.pi/180.0 #degree to radian conversion factor
    
    #rot1 = affine.rotation_matrix(euler1*deg2rad,axis=e3) @ sample_basis
    #rot2 = affine.rotation_matrix(euler2*deg2rad,axis=rot1.T[0]) @ rot1
    #rot3 = affine.rotation_matrix(euler3*deg2rad,axis=rot2.T[2]) @ rot2
    rot1 = affine.rotation_matrix(euler1*deg2rad, axis=e3)
    rot2 = rot1 @ affine.rotation_matrix(euler2*deg2rad, axis=e1)
    rot3 = rot2 @ affine.rotation_matrix(euler3*deg2rad, axis=e3)
    return rot3

def generate_miller(reciprocal_lattice):
    """Input: A matrix whose columns are the reciprocal lattice vectors of a crystal.
       Output: Column martix of directions in the crystal, whose Miller indices are no greater than 12."""
    
    # ** Generating list relevant directions in reciprocal space (i.e. described by Miller indices): [0 0 1] to [12 12 12] ** 
    refl_start = perf_counter()

    combinations = list(combinations_with_replacement(range(-12,13),3))
    perms_temp = [list(permutations(n)) for n in combinations]
    perms = sum(perms_temp,[])

    #remove duplicates
    reflections_temp = np.array([np.array(p) for p in perms])
    no_duplicates1 = np.unique(reflections_temp,axis=0)

    reflections_miller = no_duplicates1[int(no_duplicates1.shape[0]/2)+1:] #keep only vectors that point in the 'positive' direction 
                                                                    #i.e. no_duplicates contains both [-1,1,-1] and [1,-1,1];
                                                                    #keep only [1,-1,1].
    refl_end = perf_counter()
    refl_time = refl_end - refl_start
    print('Generating the list of reflections took %s seconds.'%(str(refl_time)))

    return reflections_miller.T
    

def main():

    #load crystal structure to get reciprocal lattice vectors
    vo2_crystal = Crystal.from_database('vo2-m1')
    reciprocal_basis = np.array(vo2_crystal.reciprocal_vectors).T #transpose to get each RV as a column
    cond1_check = input('Use full Ewald sphere? [y/n] ').rstrip().lstrip().lower()
    
    #generate list of Miller indices of relections visible on the pattern
    if cond1_check == 'y':
        incident_kinetic_energy = 90 #in keV
        electron_wvl = electron_wavelength(incident_kinetic_energy)
        ewald_radius = 2.0*np.pi/electron_wvl
        shortest_norm = np.min(np.array([np.linalg.norm(b) for b in reciprocal_basis]))
        scale_int = 0
        while shortest_norm*scale_int < 2*ewald_radius:
            scale_int += 1
        print(scale_int)

        reflections_miller = generate_miller(scale_int)
    
    else: reflections_miller = generate_miller(12)

    #generate list of unrotated reciprocal lattice vectors
    print(reflections_miller)
    unrotated_reflections = reciprocal_basis @ reflections_miller
    
    #this array will store how many crystallites contribute to each reflection
    counts = np.zeros(reflections_miller.shape[1])

    tolerance = 1e-1 #reflection z coordinates below this number will be considered null

    inp_file = '/Users/nico/Desktop/McGill/Winter_2019/data/' + input('Input file: ')
    out_file = inp_file.split('/')[-1].split('.')[0] + '_orientations_rad_newest.dat'

    with open(inp_file) as f:
        start_time = perf_counter()
        
        nlines = len(f.readlines())
        poles = np.empty(shape = (3, nlines), dtype = np.float)
        f.seek(0)

        for index, line in enumerate(f):
            euler_angles = list(map(float,line.split(';')[2:-1]))
            rotation_matrix = orientation_matrix(*euler_angles)
            pole = rotation_matrix.T[2]
            pole = pole/np.linalg.norm(pole)
            poles[:, index] = pole
            
            #reciprocal_rotation_matrix =  rotation_matrix @ reciprocal_basis
            reflections_real_space = rotation_matrix @ unrotated_reflections
            reflections_normalised = reflections_real_space/np.linalg.norm(reflections_real_space, axis=0)

            reflections_z_coords = reflections_normalised[2]
            counts += (np.abs(reflections_z_coords) < tolerance)

    end_time = perf_counter()

    loop_time = end_time - start_time
    print('Iterating over the data file took %s seconds.'%(str(loop_time)))

    #write the results
    with open('poles2_fixed.dat', 'w') as fo:
        for pole in poles.T:
            fo.write('{0[0]:9.6f}\t{0[1]:9.6f}\t{0[2]:9.6f}\n'.format(pole))

    with open(out_file, 'w') as fo:
        for reflection, count in zip(reflections_miller.T,counts):
            fo.write('['+';'.join(list(map(str,reflection)))+']\t'+str(count)+'\n')

if __name__ == '__main__': main()
