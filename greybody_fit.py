# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:40:13 2022

@author: Akhil
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import matplotlib.ticker as ticker
from functools import partial
from console_progressbar import ProgressBar

#%%

## GREYBODY FITTING FUNCTION


# Blackbody function in terms of wavelength

def bbf(l,T_d):
    
    h = 6.626e-34    # Js
    c = 3e8          # m/s
    k = 1.38e-23     # J/K
    
    return (2*h*c**2)/((l**5)*(np.exp((h*c)/(l*k*T_d))-1))

# Dust opacity law

def k_nu(l):
    
    c = 3e8          # m/s
    B = 2
    
    return 0.1*(c/(l*1e12))**B

# Expression for the optical depth
def t_nu(l,N_H2):
    
    mu = 2.86
    mH = 1.67e-24    # g
    
    return mu*mH*k_nu(l)*N_H2

# Factor to convert the blackbody function to a function of frequency
def ltov(l):
    
    c = 3e8          # m/s
    
    return (l**2)/c

# Final greybody equation
def fluxden(omega,l,T_d,N_H2):
    
    return 1e26*omega*ltov(l)*bbf(l,T_d)*(1-np.exp(-t_nu(l,N_H2)))

# Function to fit a "cuboid" of data to the greybody function. For a final grid dimension of p x q, and for fitting r wavelengths, the data cuboid must be a p x q x r array with images along the third axis in the same sequence as in the "wavelengths" array input. omega is the solid angle of a pixel, and sig_vals is the uncertainty factor in each image. The function outputs the temperature, column density, and reduced-chisquare maps in result (p x q x 3), and the 1-sigma error in temperature and column density for the final pixel.

def greybody_fit(data,wavelengths,omega,sig_vals):

    result = np.zeros([data.shape[0],data.shape[1],3])
    pb = ProgressBar(total=result.shape[0],prefix='Performing greybody fit...',suffix='Completed',decimals=1,length=50,fill='>',zfill=' ')
    for i in range(result.shape[0]):
        pb.print_progress_bar(i)
        #print(int(float((i+1)/result.shape[0])*100),'%')
        for j in range(result.shape[1]):
            fluxes = np.array(data[i,j,:])
            sig = np.multiply(fluxes,sig_vals)
            pfluxden = partial(fluxden,omega)
            popt,pcov = curve_fit(pfluxden, wavelengths, fluxes,[10,1e22],sigma=sig,absolute_sigma='True')
            result[i,j,0] = popt[0] #temperature
            result[i,j,1] = popt[1] #cdensity
            error = np.sqrt(np.diag(pcov))
            df = len(fluxes)-2
            r = pfluxden(wavelengths,*popt)-fluxes
            chisqred = np.sum(r**2/sig**2)/df
            result[i,j,2] = chisqred

    return result, error
