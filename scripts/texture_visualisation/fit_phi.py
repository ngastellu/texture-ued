#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pf_histograms import skewed_gaussian, mode_estimate, bin_centers
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import chisquare

def fit_func(xdata,omega,alpha):
    """Inputs: -xdata, an array of values at which the function needs to be evaluated
               -mu, the mean of the unskewed normal distribution 
               -omega, the stdev of the unskewed normal distribution
               -alpha, parameter which controls how skewed the final distribution will be

       Outputs: The pdf of a skewed gaussian distribution, evaluated at all points in xdata"""
    
    mu = 32.5999999999*np.pi/180
    x_std = (xdata-mu)/omega #shifts everything so that the unskewed distribution is N(0,1)
    phi = np.exp(-(x_std**2)/2.0)/np.sqrt(2.0*np.pi)
    Phi = (1 + erf(alpha*x_std))/2.0
    
    return (2.0/omega)*phi*Phi


with open('../../../data/Project3.txt') as fo:
    n = len(fo.readlines())
    phis = np.zeros(n)
    fo.seek(0)
    
    for k, line in enumerate(fo):
        phi = float(line.split(';')[3])
        if phi >= 90:
            phi = 180 - phi
        phis[k] = phi*np.pi/180

c_axis_angle = 32.599999*np.pi/180

good_indices = np.asarray(phis!=0).nonzero()[0]
good_phis = np.array([phis[k] for k in good_indices])

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{bm}',r'\usepackage{amsmath}']
rcParams['font.size'] = 14

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
#cts, bins, patches = ax.hist(good_phis,200,color='b',alpha=0.5,density=True)
cts, bins = np.histogram(good_phis,bins=1000,density=True,weights=np.sin(good_phis))

midpoints = bin_centers(bins)

params, params_err = curve_fit(skewed_gaussian,midpoints,cts)
print('Fit parameters [mu,sigma,alpha]: ', params, '\nErrors: ', np.diag(params_err))

mu, sigma, alpha = params
estimated_peak_position = mode_estimate(params)

fitted_function = skewed_gaussian(midpoints,mu,sigma,alpha)

max_probability = np.max(fitted_function)
max_index = np.asarray(fitted_function == max_probability).nonzero()[0][0]
peak_phi = midpoints[max_index]
print('The sample mode is: ',np.degrees(peak_phi), ' degrees.') 

ax.plot(midpoints,cts,'o',color='#7f00ff',ms=0.8)
ax.plot(midpoints,fitted_function,'r',lw=0.8,label='skewed Gaussian fit')
#plt.axvline(estimated_peak_position,c='k',ls='--',lw=0.8,label='Approx. mode $m_0(\mu,\sigma,\\alpha)$')
plt.axvline(peak_phi,c='k',ls='-.',lw=0.8,label='Sample mode $\\varphi_{\\text{pref}}$')
plt.axvline(mu,c='k',ls='-',lw=0.8,label='Gaussian Mean $\mu$')

ax.axvline(c_axis_angle,lw=0.85,c='k',label = r'Tilt of monoclinic $c$-axis',ls=':')


ax.set_xlim([0,np.pi/2])
ax.legend()

#ax2 = fig.add_subplot(212)
#cts, bins, patches = ax2.hist(good_phis,200,color='b',alpha=0.5,density=True)
#ax2.plot(midpoints,cts,'ro',ms=0.8)
#ax2.plot(midpoints,fitted_function2,'r',lw=0.8,label='skewed Gaussian fit')
#ax2.axvline(c_axis_angle,lw=0.85,c='k',label = r'Tilt of monoclinic $c$-axis',ls=':')


plt.xlabel('$\\varphi$ [radians]')
plt.ylabel(r'Normalised counts')
plt.show()
