#!/usr/bin/env pythonw

from skued import Crystal, pseudo_voigt, structure_factor, bounded_reflections, baseline_dt, electron_wavelength
import numpy as np

def powderdiff(crystal,q,reflections=None,crystallite_size=None,count_list=1,compute_lorenz=False,return_peak_list=False,**kwargs):
    """Modification of scikit-ued's `powdersim` function that takes sample texture into account.
       
       Inputs
       ---
       crystal: a scikit-ued `Crystal` object representing the crystal from which to diffract

       q: a (N,) NumPy array contatining a range of scattering vector norms

       reflections: a (M,3) NumPy array of Miller indices [optional]

       crystallite_size: a float representing the average crystallite size in the sample [optional]
       
       count_list: a (M,) NumPy array containing the number of crystallite contributing to each diffraction peak [optional]
       
       compute_lorenz: a boolean which determines whether or not the Lorenz factor is evaluated for the pattern's diffracion peaks

       return_peak_list: a boolean that determines whether or not to return `peak_list` array [optional]
       
       Output
       ---
       pattern: a (N,) NumPy array containing the diffraction pattern

       peak_list: a (M,6) NumPy array containing the following information about each reflection:
         - postion
         - Miller indices
         - intensity
         - closest point in the `q` array
       """    
    if np.all(reflections) == None:
        reflections = np.array(bounded_reflections(crystal,nG=q.max()))
        #refls = np.vstack(tuple(bounded_reflections(crystal,nG=q.max())))
        #h, k, l = np.hsplit(refls, 3)
        #Gx, Gy, Gz = change_basis_mesh(h, k, l, basis1=crystal.reciprocal_vectors, basis2=np.eye(3))
    
    h, k, l = reflections
    Gx, Gy, Gz = crystal.scattering_vector(h, k, l)
    qs = np.sqrt(Gx ** 2 + Gy ** 2 + Gz ** 2)
    
    if crystallite_size == None:
        fwhm_g = 0.03
    else:
        fwhm_g = 0.9*2.0*np.pi/crystallite_size #Scherrer eqn from DOI: 10.1533/9780857096296.1.3

    intensities = (np.absolute(structure_factor(crystal, h, k, l))*count_list) ** 2
    pattern = np.zeros_like(q)

    peak_list = np.zeros((qs.shape[0],6),dtype=np.float)
    peak_list[:,0] = qs
    peak_list[:,1:4] = reflections.transpose()
    peak_list[:,4:] = -1 #will hold closest point to q in sgrid and intensity of signal at that point
    
    for qi, i in zip(qs, intensities):
        if compute_lorenz and qi !=0:
            wvl = electron_wavelength(90)
            lorenz_factor = 4*np.pi/(wvl*qi)   
        else: lorenz_factor=1
        pattern += i * pseudo_voigt(q, qi, fwhm_g, fwhm_g*2.0) * lorenz_factor

    return pattern, peak_list
