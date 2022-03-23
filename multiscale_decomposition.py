# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:33:01 2022

@author: Akhil
"""

import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import matplotlib.pyplot as plt
from greybody_fit import * #NOQA
from useful_routines import * #NOQA
from scipy.optimize import fsolve
from functools import partial

#%%

## MULTISCALE DECOMPOSITION

## ENTER 160,250,350,500 micron images with unchanged pixel size/resolution/units

print("Enter file names: ")

fstr160 = get_pkg_data_filename(str(input("Enter 160 mic name: ")))
fstr250 = get_pkg_data_filename(str(input("Enter 250 mic name: ")))
fstr350 = get_pkg_data_filename(str(input("Enter 350 mic name: ")))
fstr500 = get_pkg_data_filename(str(input("Enter 500 mic name: ")))


f160 = fits.PrimaryHDU(fits.getdata(fstr160),fits.getheader(fstr160))
f250 = fits.PrimaryHDU(fits.getdata(fstr250),fits.getheader(fstr250))
f350 = fits.PrimaryHDU(fits.getdata(fstr350),fits.getheader(fstr350))
f500 = fits.PrimaryHDU(fits.getdata(fstr500),fits.getheader(fstr500))

## KERNELS
print("Loading kernels...")
kernel160_250 = get_pkg_data_filename('Kernel_HiRes_PACS_160_to_SPIRE_250.fits')
kernel160_350 = get_pkg_data_filename('Kernel_HiRes_PACS_160_to_SPIRE_350.fits')
kernel160_500 = get_pkg_data_filename('Kernel_HiRes_PACS_160_to_SPIRE_500.fits')
kernel250_350 = get_pkg_data_filename('Kernel_HiRes_SPIRE_250_to_SPIRE_350.fits')
kernel250_500 = get_pkg_data_filename('Kernel_HiRes_SPIRE_250_to_SPIRE_500.fits')
kernel350_500 = get_pkg_data_filename('Kernel_HiRes_SPIRE_350_to_SPIRE_500.fits')

## BEAMAREAS
print("Loading beam areas...")
beamarea_250 = 469.35
beamarea_350 = 831.27
beamarea_500 = 1804.31

## PIXELSIZES
print("Loading pixel sizes...")
pix160 = find_pixel_scale(f160.header)
pix250 = find_pixel_scale(f250.header)
pix350 = find_pixel_scale(f350.header)
pix500 = find_pixel_scale(f500.header)

#%%

## TERM 1 COMPUTATION

print("Computing term 1...")

print("Processing f160...")
f160_conv500_jpp = convolveim(f160, kernel160_500)
f160_conv500_reg500_jpp = regridim(f160_conv500_jpp,pix500)
findat1 = f160_conv500_reg500_jpp.data

print("Processing f250...")
f250_conv500 = convolveim(f250, kernel250_500)
f250_conv500_reg500 = regridim(f250_conv500,pix500)
f250_conv500_reg500_jpp = mjps2jpp(f250_conv500_reg500) 
findat2 = f250_conv500_reg500_jpp.data

print("Processing f350...")
f350_conv500 = convolveim(f350, kernel350_500)
f350_conv500_reg500 = regridim(f350_conv500,pix500)
f350_conv500_reg500_jpp = mjps2jpp(f350_conv500_reg500) 
findat3 = f350_conv500_reg500_jpp.data

print("Processing f500...")
f500_jpp = mjps2jpp(f500)
findat4 = f500_jpp.data[:-1,:-1]

## The dimensions may not always match at this step, causing rebroadcasting errors in gen_cube. The difference in dimensions should not be more than about 2 pixels however, and should be resolvable by manually editting dataN.

print("Performing greybody fit...")

data_naxis1 = findat1.shape[0]
data_naxis2 = findat1.shape[1]
n_data = 4
data = np.zeros([data_naxis1,data_naxis2,n_data])
for i in range(1,n_data+1):
    data[:,:,i-1] = eval('findat%d'%(i))
        
wavelengths = np.array([160,250,350,500])*1e-6 # m
omega = (pix500/3600)**2*3.046e-4
sig_vals = np.array([0.15,0.15,0.15,0.15])

result, error = greybody_fit(data, wavelengths, omega, sig_vals)
sigma500_reg500 = fits.PrimaryHDU(result[:,:,1])
sigma500_reg500.header = f500_jpp.header

print("Regridding column density to pix250...")
sigma500_reg250 = regridim(sigma500_reg500,pix250)
msd_term1_reg250 = sigma500_reg250

print("Term 1 computed!")

#%%

## TERM 2 COMPUTATION

print("Computing term 2...")

print("Processing f160...")
f160_conv350_jpp = convolveim(f160, kernel160_350)
f160_conv350_reg350_jpp = regridim(f160_conv350_jpp,pix350)
findat1 = f160_conv350_reg350_jpp.data[:-1,:-1]

print("Processing f250...")
f250_conv350 = convolveim(f250, kernel250_350)
f250_conv350_reg350 = regridim(f250_conv350,pix350)
f250_conv350_reg350_jpp = mjps2jpp(f250_conv350_reg350) 
findat2 = f250_conv350_reg350_jpp.data

print("Processing f350...")
f350_jpp = mjps2jpp(f350) 
findat3 = f350_jpp.data[:-1,:-1]

## The dimensions may not always match at this step, causing rebroadcasting errors in gen_cube. The difference in dimensions should not be more than about 2 pixels however, and should be resolvable by manually editting dataN.

print("Performing greybody fit...")

data_naxis1 = findat1.shape[0]
data_naxis2 = findat1.shape[1]
n_data = 3
data = np.zeros([data_naxis1,data_naxis2,n_data])
for i in range(1,n_data+1):
    data[:,:,i-1] = eval('findat%d'%(i))
        
wavelengths = np.array([160,250,350])*1e-6 # m
omega = (pix350/3600)**2*3.046e-4
sig_vals = np.array([0.15,0.15,0.15])

result, error = greybody_fit(data, wavelengths, omega, sig_vals)
sigma350_reg350 = fits.PrimaryHDU(result[:,:,1])
sigma350_reg350.header = f350_jpp.header

print("Performing unsharp masking...")
sigma350_reg350_conv500 = convolveim(sigma350_reg350,kernel350_500)
msd_term2_reg350 = fits.PrimaryHDU(sigma350_reg350.data-sigma350_reg350_conv500.data)
msd_term2_reg350.header = sigma350_reg350_conv500.header

print("Regridding column density to pix250...")
msd_term2_reg250 = regridim(msd_term2_reg350,pix250)
msd_term2_reg250 = msd_term2_reg250

print("Term 2 computed!")

#%%

## TERM 3 COMPUTATION

print("Computing term 3...")

print("Processing f160...")
f160_conv250_jpp = convolveim(f160, kernel160_250)
f160_conv250_reg250_jpp = regridim(f160_conv250_jpp,pix250)
findat1 = f160_conv250_reg250_jpp.data[:-1,:-1]

print("Processing f250...")
f250_jpp = mjps2jpp(f250)
findat2 = f250_jpp.data

print("Producing color-temperature map from flux ratio...")

h = 6.626e-34    # Js
c = 3e8          # m/s
k = 1.38e-23     # J/K
mu = 2.86
mH = 1.67e-24    # g
B = 2
omega = (pix250/3600)**2*3.046e-4

def factor(r,T):
    return (np.exp((h*c)/(k*T*250e-6))-1)/(np.exp((h*c)/(k*T*160e-6))-1) - r

data_ratio = findat1/findat2
B_ratio = data_ratio*(160/250)**(B+3)
tempmap = np.zeros(findat1.shape)
for i in range(findat1.shape[0]):
    print(int(float((i+1)/findat1.shape[0])*100),'%')
    for j in range(findat1.shape[1]):
        r = B_ratio[i,j]
        part = partial(factor,r)
        root = fsolve(part,[7])
        tempmap[i,j] = root[0]

sigma250_reg250 = fits.PrimaryHDU(findat2/(1e26*omega*ltov(250e-6)*bbf(250e-6,tempmap)*mu*mH*k_nu(250e-6)))
sigma250_reg250.header = f250_jpp.header

print("Performing unsharp masking...")
sigma250_reg250_conv350 = convolveim(sigma250_reg250,kernel250_350)
msd_term3_reg250 = fits.PrimaryHDU(sigma250_reg250.data-sigma250_reg250_conv350.data)
msd_term3_reg250.header = sigma250_reg250_conv350.header

print("Term 3 computed!")

#%%

## PUTTING TOGETHER SCALES

print("Adding terms...")
msd_cdens_reg250 = fits.PrimaryHDU(msd_term1_reg250.data + msd_term2_reg250.data + msd_term3_reg250.data[:-2,:-2])
msd_cdens_reg250.header = msd_term1_reg250.header

## The dimensions may not always match at this step, causing rebroadcasting errors. The difference in dimensions should not be more than about 2 pixels however, and should be resolvable by manually editting dataN.

plt.subplot(2,2,1,projection=WCS(msd_cdens_reg250.header))
plt.imshow(msd_term1_reg250.data)
plt.subplot(2,2,2,projection=WCS(msd_cdens_reg250.header))
plt.imshow(msd_term2_reg250.data)
plt.subplot(2,2,3,projection=WCS(msd_cdens_reg250.header))
plt.imshow(msd_term3_reg250.data)
plt.subplot(2,2,4,projection=WCS(msd_cdens_reg250.header))
plt.imshow(msd_cdens_reg250.data)

#%%

## Ratio of term 1 to final map convolved to 500 psf.

msd_cdens_reg250_conv500 = convolveim(msd_cdens_reg250,kernel250_500)

ratiolow2hi = fits.PrimaryHDU(msd_term1_reg250.data/msd_cdens_reg250_conv500.data)
ratiolow2hi.header = msd_cdens_reg250.header

plt.imshow(ratiolow2hi.data)
print("Mean = ",ratiolow2hi.data.mean())
print("Standard deviation = ",ratiolow2hi.data.std())
