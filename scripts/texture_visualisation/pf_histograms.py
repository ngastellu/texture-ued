#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.special import erf, erfc

def skewed_gaussian(xdata,mu,omega,alpha):
    """Inputs: -xdata, an array of values at which the function needs to be evaluated
               -mu, the mean of the unskewed normal distribution 
               -omega, the stdev of the unskewed normal distribution
               -alpha, parameter which controls how skewed the final distribution will be

       Outputs: The pdf of a skewed gaussian distribution, evaluated at all points in xdata"""
    
    x_std = (xdata-mu)/omega #shifts everything so that the unskewed distribution is N(0,1)
    phi = np.exp(-(x_std**2)/2.0)/np.sqrt(2.0*np.pi)
    Phi = (1 + erf(alpha*x_std))/2.0
    
    return (2.0/omega)*phi*Phi
       
def mode_estimate(parameters):
    """Inputs: list of parameters describing a skewed Gaussian distribution

       Outputs: mode (i.e. x coord of the maximum) of the distribution"""

    mu, omega, alpha = parameters

    delta = alpha/np.sqrt(1+(alpha**2))
    mu_z = np.sqrt(2/np.pi)*delta
    sigma_z = np.sqrt(1-(mu_z**2))
    gamma = ((4-np.pi)/2) * ( mu_z / np.sqrt(1 - (mu_z**2)) )**3

    m0 = mu_z - ((gamma*sigma_z)/2) - (np.sign(alpha)/2)*np.exp(-2*np.pi/np.abs(alpha))

    return mu + omega*m0


def modified_SG_derivative(xdata,mu,omega,alpha):
    """Inputs: -xdata, an array of values at which the function needs to be evaluated
               -mu, the mean of the unskewed normal distribution 
               -omega, the stdev of the unskewed normal distribution
               -alpha, parameter which controls how skewed the final distribution will be

       Outputs: The only factor of the derivative of the pdf of a skewed gaussian distribution that can be null, 
                evaluated at all points in xdata. We do not compute the full derivative to make it easier for us 
                to numerically find its root (i.e. the mode of the distribution)."""
    y = alpha*(xdata-mu)/(omega*np.sqrt(2))

    return alpha*np.exp(-(y**2))/(np.pi*(omega**2)) + (xdata-mu)*(-2+erfc(y))/((omega**3)*np.sqrt(2*np.pi))

def bin_centers(bin_edges):
    """Input: Array of bin edges for a 1D histogram

       Output: Array of the centers of the bins described by the input array"""
    
    bin_edges = np.asarray(bin_edges)
    return np.array([bin_edges[k]+((bin_edges[k+1]-bin_edges[k])/2.0) for k in range(bin_edges.shape[0]-1)])

def main():
    with open('pole_figure3_sph_fixed.dat') as fo:
        data = np.array([list(map(float,fline.split())) for fline in fo.readlines()[1:]]).T

    xlabels = ['Radius $r$ of projection', 'Angle $\\theta$ of projection']

    plt.rc('text',usetex=True)

    #for dat2,dat3,xlbl in zip(data2,data3,xlabels):
    #    average2 = np.mean(dat2)
    #    average3 = np.mean(dat3)
    #    n2, bins2, patches2 = plt.hist(dat2,500,alpha=0.5,color='b',label='2',density=True)
    #    n3, bins3, patches3 = plt.hist(dat3,500,alpha=0.5,color='r',label='3',density=True)
    #    plt.axvline(x=average2,ymin=0,ymax=1.5,c='b',lw=0.9,label='$\mu_2$')
    #    plt.axvline(x=average3,ymin=0,ymax=1.5,c='r',lw=0.9,label='$\mu_3$')
    #    plt.xlabel(xlbl)
    #    plt.ylabel('Counts')
    #    plt.legend()
    #    plt.show()
    print(data)
    radii = data[0,:]
    print(radii)

    #param_guesses = np.array([[0.25039944, 0.23618676, 2.23226873],[[0.23364881,0.34802574,4.89120597]]])

    n, bins, patches = plt.hist(radii,1000,density=True,color='b',alpha=0.5)
    midpoints = bin_centers(bins)

    params, params_err = curve_fit(skewed_gaussian,midpoints,n)
    print('Fit parameters [mu,sigma,alpha]: ', params, '\nErrors: ', np.diag(params_err))

    mu, sigma, alpha = params
    estimated_peak_position = mode_estimate(params)
    mean = mu + sigma*np.sqrt(2/np.pi)*alpha/np.sqrt(1+(alpha**2))

    fitted_function = skewed_gaussian(midpoints,mu,sigma,alpha)
    max_probability = np.max(fitted_function)
    max_index = np.asarray(fitted_function == max_probability).nonzero()[0][0]
    #print(max_probability,max_index)
    peak_position = midpoints[max_index]
    print('The mode of the radius distribution is: ',peak_position)
    peak_phi = np.degrees(2*np.arctan(1/peak_position))
    print('This correspomds to a spherical polar angle (phi) of ',peak_phi, ' degrees.')
    #peak_position = fsolve(modified_SG_derivative,estimated_peak_position,args=(tuple(params)))

    plt.plot(midpoints,n,'ro',ms=0.8)
    plt.plot(midpoints,fitted_function,'r',lw=0.8,label='skewed Gaussian fit')
    plt.axvline(estimated_peak_position,c='k',ls='--',lw=0.8,label='Approx. mode $m_0(\mu,\sigma,\\alpha)$')
    plt.axvline(peak_position,c='k',ls='-.',lw=0.8,label='Numerically estimated mode')
    plt.axvline(mean,c='k',ls='-',lw=0.8,label='Mean $\mu_\\alpha$')
    plt.legend()
    plt.show()

if __name__ == '__main__': main()
