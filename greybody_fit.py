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

#%%

## DATA TO BE FIT

f1 = get_pkg_data_filename('HF09729_columndensity.fits')
f2 = get_pkg_data_filename('HF09729_msd_cdens_reg250.fits')
f3 = get_pkg_data_filename('SIGMA_250_reg250.fits')
f4 = get_pkg_data_filename('HF09729_850_finalcrop.fits')

dat1 = fits.getdata(f1)
dat2 = fits.getdata(f2)
dat3 = fits.getdata(f3)
dat4 = fits.getdata(f4)


#%%

## THE FUNCTION


## GREYBODY FITTING FUNCTION

def bbf(l,T_d):
    
    h = 6.626e-34    # Js
    c = 3e8          # m/s
    k = 1.38e-23     # J/K
    
    return (2*h*c**2)/((l**5)*(np.exp((h*c)/(l*k*T_d))-1))

def k_nu(l):
    
    c = 3e8          # m/s
    B = 2
    
    return 0.1*(c/(l*1e12))**B
    
def t_nu(l,N_H2):
    
    mu = 2.86
    mH = 1.67e-24    # g
    
    return mu*mH*k_nu(l)*N_H2

# Convert the Planck function to a function of frequency
def ltov(l):
    
    c = 3e8          # m/s
    
    return (l**2)/c

def fluxden(omega,l,T_d,N_H2):
    
    return 1e26*omega*ltov(l)*bbf(l,T_d)*(1-np.exp(-t_nu(l,N_H2)))

def greybody_fit(data,wavelengths,omega,sig_vals):

    result = np.zeros([data.shape[0],data.shape[1],3])

    for i in range(result.shape[0]):
        print(int(float((i+1)/result.shape[0])*100),'%')
        for j in range(result.shape[1]):
            fluxes = np.array(data[i,j,:])
            sig = np.multiply(fluxes,sig_vals)
            pfluxden = partial(fluxden,omega)
            popt,pcov = curve_fit(pfluxden, wavelengths, fluxes,[10,1e22],sigma=sig,absolute_sigma='True')
            result[i,j,0] = popt[0]
            result[i,j,1] = popt[1]
            error = np.sqrt(np.diag(pcov))
            df = len(fluxes)-2
            r = pfluxden(wavelengths,*popt)-fluxes
            chisqred = np.sum(r**2/sig**2)/df
            result[i,j,2] = chisqred

    return result, error

#%%

# =============================================================================
#
# PLOT OF TEMPERATURE AND COLUMN DENSITIES WITH REDUCED CHI-SQUARE MAP
# 
# =============================================================================

# =============================================================================
# wcs1 = WCS(fits.getheader(f1))
# plt.figure()
# 
# ax = plt.subplot(2,2,1, projection=wcs1)
# lon = ax.coords[0]
# lat = ax.coords[1]
# lon.set_major_formatter('d.dd')
# lat.set_major_formatter('d.dd')
# plt.imshow(result[:,:,0],cmap='gist_rainbow_r')
# cb = plt.colorbar(orientation='vertical',label='K',fraction=0.038,pad=0.04)
# tick_locator = ticker.MaxNLocator(nbins=8)
# cb.locator = tick_locator
# cb.update_ticks()
# plt.contour(result[:,:,1], levels=np.logspace(22.69,22.94,7),colors='black',alpha=0.8, linewidths=1.2)
# plt.xlabel("Galactic Longitude")
# plt.ylabel("Galactic Latitude")
# plt.title("Temperature")
# 
# ax = plt.subplot(2,2,2, projection=wcs1)
# lon = ax.coords[0]
# lat = ax.coords[1]
# lon.set_major_formatter('d.dd')
# lat.set_major_formatter('d.dd')
# plt.imshow(result[:,:,1],cmap='gist_rainbow_r')
# cb = plt.colorbar(label='cm$^{-2}$',orientation='vertical',fraction=0.038,pad=0.04)
# tick_locator = ticker.MaxNLocator(nbins=8)
# cb.locator = tick_locator
# cb.update_ticks()
# cb.add_lines(plt.contour(result[:,:,1], levels=np.logspace(22.69,22.94,7),colors='black',alpha=0.8, linewidths=1.2))
# plt.xlabel("Galactic Longitude")
# plt.ylabel("Galactic Latitude")
# plt.title("Column density")
# 
# ax = plt.subplot(2,2,3, projection=wcs1)
# lon = ax.coords[0]
# lat = ax.coords[1]
# lon.set_major_formatter('d.dd')
# lat.set_major_formatter('d.dd')
# plt.imshow(result[:,:,2],cmap='gist_rainbow_r')
# cb = plt.colorbar(orientation='vertical',fraction=0.038,pad=0.04)
# tick_locator = ticker.MaxNLocator(nbins=8)
# cb.locator = tick_locator
# cb.update_ticks()
# plt.contour(result[:,:,1], levels=np.logspace(22.69,22.94,7),colors='black',alpha=0.8, linewidths=1.2)
# plt.xlabel("Galactic Longitude")
# plt.ylabel("Galactic Latitude")
# plt.title("$\chi^{2}_{red}$")
# 
# =============================================================================

wcs1 = WCS(fits.getheader(f1))

plt.figure()

ax = plt.subplot(2,1,1,projection=wcs1)
lon = ax.coords[0]
lat = ax.coords[1]
lon.set_major_formatter('d.dd')
lat.set_major_formatter('d.dd')
lat.set_ticks(number=6)
plt.imshow(dat1,cmap='gist_rainbow_r')
plt.clim(3e22,8.5e22)
cb = plt.colorbar(label='cm$^{-2}$',orientation='vertical',fraction=0.038,pad=0.04)
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator
cb.update_ticks()
cb.add_lines(plt.contour(dat1, levels=np.logspace(22.69,22.94,7),colors='black',alpha=0.8, linewidths=0))
plt.xlabel("Galactic Longitude",fontsize=12)
plt.ylabel("Galactic Latitude",fontsize=12)
plt.title("Column density at $24^{\prime \prime}.9$")

wcs2 = WCS(fits.getheader(f2))

ax = plt.subplot(2,1,2,projection=wcs2)
lon = ax.coords[0]
lat = ax.coords[1]
lon.set_major_formatter('d.dd')
lat.set_major_formatter('d.dd')
plt.imshow(dat2,cmap='gist_rainbow_r')
plt.clim(4e22,1.35e23)
cb = plt.colorbar(label='cm$^{-2}$',orientation='vertical',fraction=0.038,pad=0.04)
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator
cb.update_ticks()
cb.add_lines(plt.contour(dat2, levels=np.logspace(22.69,22.94,7),colors='black',alpha=0.8, linewidths=0))
plt.ylabel("Galactic Latitude",fontsize=12)
plt.xlabel("Galactic Longitude",fontsize=12)
plt.title("Column density at $18^{\prime \prime}$.2")
