#!/usr/bin/env pythonw

from skued import Crystal, pseudo_voigt, structure_factor, bounded_reflections, baseline_dt, electron_wavelength
import numpy as np
import matplotlib.pyplot as plt
from iris import PowderDiffractionDataset
from time import perf_counter
from scipy.integrate import simps

def powdersim(crystal,q,reflections=None,crystallite_size=None,count_list=1,compute_lorenz=False,**kwargs):
    """Modification of scikit-ued's `powdersim` function that takes sample texture into account.
       Inputs
       ---
       crystal: a scikit-ued `Crystal` object representing the crystal from which to diffract

       q: a (N,) NumPy array contatining a range of scattering vector norms

       reflections: a (M,3) NumPy array of Miller indices [optional]

       crystallite_size: a float representing the average crystallite size in the sample [optional]
       
       count_list: a (M,) NumPy array containing the number of crystallite contributing to each diffraction peak [optional]
       
       compute_lorenz: a boolean which determines whether or not the Lorenz factor is evaluated for the pattern's diffracion peaks."""
       
    
    if np.all(reflections) == None:
        refls = np.vstack(tuple(crystal.bounded_reflections(q.max())))
        h, k, l = np.hsplit(refls, 3)
        Gx, Gy, Gz = change_basis_mesh(h, k, l, basis1=crystal.reciprocal_vectors, basis2=np.eye(3))
    else:
        h, k, l = reflections
        Gx, Gy, Gz = crystal.scattering_vector(h, k, l)
    
    qs = np.sqrt(Gx ** 2 + Gy ** 2 + Gz ** 2)
    
    if crystallite_size == None:
        fhwm_g = 0.03
    else:
        fwhm_g = 0.9*2.0*np.pi/crystallite_size #Scherrer eqn from DOI: 10.1533/9780857096296.1.3

    intensities = (np.absolute(structure_factor(crystal, h, k, l))*count_list) ** 2
    pattern = np.zeros_like(q)

    peak_list = np.zeros((qs.shape[0],6),dtype=np.float)
    peak_list[:,0] = qs
    peak_list[:,1:4] = reflections.transpose()
    peak_list[:,4:] = -1 #will hold closest point to q in sgrid and intensity of signal at that point
    
    for qi, i in zip(qs, intensities):
        if compute_lorenz:
            wvl = electron_wavelength(90)
            lorenz_factor = 4*np.pi/(wvl*qi)   
        else: lorenz_factor=1
        pattern += i * pseudo_voigt(q, qi, fwhm_g, fwhm_g*2.0) * lorenz_factor

    return pattern, peak_list


vo2 = Crystal.from_database('vo2-m1')
reciprocal_vectors = np.array(vo2.reciprocal_vectors)


inp_file = '../data/Project3_orientations_rad_newest.dat'

with open(inp_file) as fo:
    nlines = len(fo.readlines())
    refs = np.zeros((3,nlines))
    counts = np.zeros(nlines)
    
    fo.seek(0)
    for k, line in enumerate(fo):
        split_line = line.split()
        refs[:,k] = np.array(list(map(float, split_line[0].lstrip('[').rstrip(']').split(';'))))  
        counts[k] = float(split_line[1])


exptl_data = PowderDiffractionDataset('../../data/vance_data_v5.hdf5',mode='r')
exptl_s_full = exptl_data.scattering_vector
#exptl_data.compute_baseline(first_stage='sym4',wavelet='qshift3') 
exptl_pattern_w_bg_full = exptl_data.powder_data(0,bgr=False)

#keep only relevant parts of pattern
start_index = np.min(np.asarray(exptl_s_full>(0.16*4*np.pi)).nonzero()[0])
end_index = np.min(np.asarray(exptl_s_full>(0.75*4*np.pi)).nonzero()[0])

exptl_s = exptl_s_full[start_index:end_index]
exptl_pattern_w_bg = exptl_pattern_w_bg_full[start_index:end_index]

#baseline removal; first_stage can be set to 'sym4' or 'sym5'
baseline = baseline_dt(exptl_pattern_w_bg,first_stage='sym5',wavelet='qshift3',max_iter=100,level=6)
exptl_pattern = exptl_pattern_w_bg - baseline

mean_crystallite_size = 1.3e3/(4.0*np.pi) #in angstroms
#mean_crystallite_size = 8e2/(4.0*np.pi) #in angstroms
#sgrid = np.linspace(2.5,5,2048)
sgrid = np.linspace(2.5,16.288476969050766,5096)
textured_out, peaks = powdersim(vo2,sgrid,refs,mean_crystallite_size,count_list=counts)
ideal_out, ideal_peaks = powdersim(vo2,sgrid,refs,mean_crystallite_size,count_list=1.0,compute_lorenz=True)
test_out, test_peaks = powdersim(vo2,sgrid,refs,mean_crystallite_size,count_list=counts,compute_lorenz=True)

tolerance = 5.0e-4

exptl_start_index = np.min(np.asarray(exptl_s>(0.207*4*np.pi)).nonzero()[0])

#max_I = simps(textured_out[:935],sgrid[:935]/(4*np.pi))
#max_I_ideal = simps(ideal_out[:935],sgrid[:935]/(4*np.pi))
max_I_exptl = simps(exptl_pattern[exptl_start_index:],exptl_s[exptl_start_index:])
max_I = simps(textured_out,sgrid)
max_I_ideal = simps(ideal_out,sgrid)
max_I_test = np.sum(test_out[:3073])*(sgrid[1]-sgrid[0])

plt.figure(1)
plt.rc('text',usetex=True)
plt.rc('font',size=14)

interesting_refs = np.array([[2,2,0],[0,2,-1],[3,0,-2],[3,1,-3],[2,0,0],[2,-3,-1],[4,0,-2],[2,1,0]])

plt.plot(sgrid/(4.0*np.pi),test_out/max_I_test,linewidth=0.9,label='textured',c='#d54067')
plt.plot(sgrid/(4.0*np.pi),ideal_out/max_I_ideal,linewidth=0.9,label='ideal',c='#4067d5')

#Print s coordinate of peaks of interest
#for peak in peaks:
#    miller_indices = peak[1:4]
#    check = np.all(interesting_refs==miller_indices, axis=1) #checks if the reflection is in interesting_refs
#    if np.any(check):
#        print(miller_indices)
#        print(peak[0]/(4.0*np.pi))
#        plt.axvline(ymin=0,ymax=1,x=peak[0]/(4.0*np.pi),linewidth=0.5,c='k')
#        plt.annotate('[{0[0]:1.0f},{0[1]:1.0f},{0[2]:1.0f}]'.format(miller_indices),xy=(peak[0]/(4.0*np.pi)+0.003,1.05))

plt.xlabel(r'$q$ (\AA$^{-1}$)')
plt.ylabel('Normalised intensity')
plt.xlim([0.2,0.36])
plt.legend(loc=(0.17,0.85))
plt.show()
