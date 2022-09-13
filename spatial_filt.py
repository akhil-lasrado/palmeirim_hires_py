# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:33:01 2022 v0.1

@author: Akhil Lasrado, akhil.lasrado@hotmail.com, IISER Kolkata


SPATIAL FILTERING ALGORITHM FOR HERSCHEL HI-GAL IMAGES (160,250,350,500 MIC)
BASED ON THE PRESCRIPTION DESCRIBED IN PALMEIRIM ET AL (2013) (https://doi.org/10.1051/0004-6361/201220500).

> DEPENDENCIES INCLUDE: numpy, astropy, scipy, functools, reproject, console-progressbar (Installation: pip install console-progressbar) (for visualization)

> Enter 160,250,350,500 micron image (with units in Jy/pixel!) file names in ./+spatial_filt.cfg file

> It is recommended to run the code section-by-section (Shift+Enter in spyder), and to run the code in a new folder 
  containing the three scripts (spatial_filt.py, greybody_fit.py, and useful_routines.py) along with the ./+spatial_filt.cfg file
  to prevent overwriting of existing files. The kernels folder must also be placed here.

"""

#%%

import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import matplotlib.pyplot as plt
from greybody_fit import * #NOQA
from useful_routines import * #NOQA
from scipy.optimize import fsolve
from functools import partial

#%%

with open('./+spatial_filt.cfg') as f:
    lines = [line.rstrip('\n') for line in f]

fstr160 = get_pkg_data_filename(lines[2][6:])
fstr250 = get_pkg_data_filename(lines[3][6:])
fstr350 = get_pkg_data_filename(lines[4][6:])
fstr500 = get_pkg_data_filename(lines[5][6:])


f160 = fits.PrimaryHDU(fits.getdata(fstr160),fits.getheader(fstr160))
f250 = fits.PrimaryHDU(fits.getdata(fstr250),fits.getheader(fstr250))
f350 = fits.PrimaryHDU(fits.getdata(fstr350),fits.getheader(fstr350))
f500 = fits.PrimaryHDU(fits.getdata(fstr500),fits.getheader(fstr500))

## KERNELS

# Loads kernels according to filenames of kernels provided by Aniano et al (2011).

print("Loading kernels...")
kernel160_250 = get_pkg_data_filename('kernels/Kernel_HiRes_PACS_160_to_SPIRE_250.fits')
kernel160_350 = get_pkg_data_filename('kernels/Kernel_HiRes_PACS_160_to_SPIRE_350.fits')
kernel160_500 = get_pkg_data_filename('kernels/Kernel_HiRes_PACS_160_to_SPIRE_500.fits')
kernel250_350 = get_pkg_data_filename('kernels/Kernel_HiRes_SPIRE_250_to_SPIRE_350.fits')
kernel250_500 = get_pkg_data_filename('kernels/Kernel_HiRes_SPIRE_250_to_SPIRE_500.fits')
kernel350_500 = get_pkg_data_filename('kernels/Kernel_HiRes_SPIRE_350_to_SPIRE_500.fits')

## PIXELSIZES

print("Loading pixel sizes...")
pix160 = find_pixel_scale(f160.header)
pix250 = find_pixel_scale(f250.header)
pix350 = find_pixel_scale(f350.header)
pix500 = find_pixel_scale(f500.header)

#%%

## TERM 1 COMPUTATION

# This step will prepare the images for a pixel-by-pixel fit at the resolution of the 500 micron image,
# The regridding step can introduce NaN borders in the images, and these must be inspected and trimmed off before the greybody fit,
# A provision for this is provided in the following sections.

print("Computing term 1...")

print("Processing f160...")
f160_conv500 = convolveim(f160, kernel160_500)
f160_conv500_reg500 = regridim(f160_conv500,pix500)
findat1 = f160_conv500_reg500.data

print("Processing f250...")
f250_conv500 = convolveim(f250, kernel250_500)
f250_conv500_reg500 = regridim(f250_conv500,pix500)
findat2 = f250_conv500_reg500.data

print("Processing f350...")
f350_conv500 = convolveim(f350, kernel350_500)
f350_conv500_reg500 = regridim(f350_conv500,pix500)
findat3 = f350_conv500_reg500.data

print("Processing f500...")
findat4 = f500.data

#%%

# Plot processed images to check for length of NaN borders to be trimmed off. Choose the largest NaN border width of all images and
# trim this off all images to ensure the same spatial extent is retained in all images.

plt.subplot(2,2,1)
plt.imshow(findat1)
plt.subplot(2,2,2)
plt.imshow(findat2)
plt.subplot(2,2,3)
plt.imshow(findat3)
plt.subplot(2,2,4)
plt.imshow(findat4)

#%%

# Trim border of 1 pixel (for example) from all images

findat1 = findat1[1:-1,1:-1]
findat2 = findat2[1:-1,1:-1]
findat3 = findat3[1:-1,1:-1]
findat4 = findat4[1:-1,1:-1]

#%%

# Create temporary arrays for testing alignment. In case you mess up the align arrays in any way, you can 
# always restore the original align arrays from this step.

align1 = np.copy(findat1)
align2 = np.copy(findat2)
align3 = np.copy(findat3)
align4 = np.copy(findat4)

#%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%4+1
    
#%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs

align2 = shift(align2,'right',1)
align3 = shift(align3,'up',1)

#%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, an maximal border must be trimmed off of all images.

align1 = trimbor(align1,[1,1,1,1])
align2 = trimbor(align2,[1,1,1,1])
align3 = trimbor(align3,[1,1,1,1])
align4 = trimbor(align4,[1,1,1,1])

#%%

# Once the images have been aligned, and 0 borders have been trimmed off, the alignN arrays are used for the greybody fit.
# the console-progressbar package has been used to visualize progress here, however this can be turned off by commenting out
# lines including pb in the greybody_fit.py file.

data_naxis1 = align1.shape[0]
data_naxis2 = align1.shape[1]
n_data = 4
data = np.zeros([data_naxis1,data_naxis2,n_data])
for i in range(1,n_data+1):
    data[:,:,i-1] = eval('align%d'%(i))
        
wavelengths = np.array([160,250,350,500])*1e-6 # m
omega = (pix500/3600)**2*3.046e-4
sig_vals = np.array([0.15,0.15,0.15,0.15])

result, error = greybody_fit(data, wavelengths, omega, sig_vals)
sigma500_reg500 = fits.PrimaryHDU(result[:,:,1])
sigma500_reg500.header = f500.header

print("\nRegridding column density to pix250...")
sigma500_reg250 = regridim(sigma500_reg500,pix250)
msd_term1_reg250 = sigma500_reg250

print("\nTerm 1 computed!")
fits.writeto('msd_term1_reg250.fits',msd_term1_reg250.data,msd_term1_reg250.header)

#%%

## TERM 2 COMPUTATION
# This step will prepare the images for a pixel-by-pixel fit at the resolution of the 350 micron image,
# The regridding step can introduce NaN borders in the images, and these must be inspected and trimmed off before the greybody fit,
# A provision for this is provided in the following sections.

print("Computing term 2...")

print("Processing f160...")
f160_conv350 = convolveim(f160, kernel160_350)
f160_conv350_reg350 = regridim(f160_conv350,pix350)
findat1 = f160_conv350_reg350.data

print("Processing f250...")
f250_conv350 = convolveim(f250, kernel250_350)
f250_conv350_reg350 = regridim(f250_conv350,pix350)
findat2 = f250_conv350_reg350.data

print("Processing f350...")
findat3 = f350.data

#%%

# Plot processed images to check for length of NaN borders to be trimmed off. Choose the largest NaN border width of all images and
# trim this off all images to ensure the same spatial extent is retained in all images.

plt.subplot(1,3,1)
plt.imshow(findat1)
plt.subplot(1,3,2)
plt.imshow(findat2)
plt.subplot(1,3,3)
plt.imshow(findat3)

#%%

# Trim border of 1 pixel from all images

findat1 = trimbor(findat1,[1,1,1,1])
findat2 = trimbor(findat2,[1,1,1,1])
findat3 = trimbor(findat3,[1,1,1,1])

#%%

# Create temporary arrays for testing alignment. In case you mess up the align arrays in any way, you can 
# always restore the original align arrays from this step.

align1 = np.copy(findat1)
align2 = np.copy(findat2)
align3 = np.copy(findat3)

#%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%3+1
    
#%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs

align3 = shift(align3,'left',1)

#%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, an maximal border must be trimmed off of all images.

align1 = trimbor(align1,[1,1,1,1])
align2 = trimbor(align2,[1,1,1,1])
align3 = trimbor(align3,[1,1,1,1])

#%%

# Once the images have been aligned, and 0 borders have been trimmed off, the alignN arrays are used for the greybody fit.
# the console-progressbar package has been used to visualize progress here, however this can be turned off by commenting out
# lines including pb in the greybody_fit.py file.

data_naxis1 = align1.shape[0]
data_naxis2 = align1.shape[1]
n_data = 3
data = np.zeros([data_naxis1,data_naxis2,n_data])
for i in range(1,n_data+1):
    data[:,:,i-1] = eval('align%d'%(i))
        
wavelengths = np.array([160,250,350])*1e-6 # m
omega = (pix350/3600)**2*3.046e-4
sig_vals = np.array([0.15,0.15,0.15])

result, error = greybody_fit(data, wavelengths, omega, sig_vals)
sigma350_reg350 = fits.PrimaryHDU(result[:,:,1])
sigma350_reg350.header = f350.header

print("\nPerforming unsharp masking...")
sigma350_reg350_conv500 = convolveim(sigma350_reg350,kernel350_500)
msd_term2_reg350 = fits.PrimaryHDU(sigma350_reg350.data-sigma350_reg350_conv500.data)
msd_term2_reg350.header = sigma350_reg350_conv500.header

print("\nRegridding column density to pix250...")
msd_term2_reg250 = regridim(msd_term2_reg350,pix250)
msd_term2_reg250 = msd_term2_reg250

print("\nTerm 2 computed!")
fits.writeto('msd_term2_reg250.fits',msd_term2_reg250.data,msd_term2_reg250.header)

#%%

## TERM 3 COMPUTATION
# This step will prepare the images for creating a columndensity map from a color-temperature map at the resolution of the 250 micron image,
# The regridding step can introduce NaN borders in the images, and these must be inspected and trimmed off before the greybody fit,
# A provision for this is provided in the following sections.

print("Computing term 3...")

print("Processing f160...")
f160_conv250 = convolveim(f160, kernel160_250)
f160_conv250_reg250 = regridim(f160_conv250,pix250)
findat1 = f160_conv250_reg250.data

print("Processing f250...")
findat2 = f250.data

#%%

# Plot processed images to check for length of NaN borders to be trimmed off. Choose the largest NaN border width of all images and
# trim this off all images to ensure the same spatial extent is retained in all images.

plt.subplot(1,2,1)
plt.imshow(findat1)
plt.subplot(1,2,2)
plt.imshow(findat2)

#%%

# Trim border of 1 pixel from all images (for example)

findat1 = trimbor(findat1,[1,1,1,1])
findat2 = trimbor(findat2,[1,1,1,1])

#%%

# Create temporary arrays for testing alignment. In case you mess up the align arrays in any way, you can 
# always restore the original align arrays from this step.

align1 = np.copy(findat1)
align2 = np.copy(findat2)

#%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%2+1
    
#%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs

align2 = shift(align2,'left',1)

#%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, an maximal border must be trimmed off of all images.

align1 = trimbor(align1,[1,1,1,1])
align2 = trimbor(align2,[1,1,1,1])

#%%

# Once the images have been aligned, and 0 borders have been trimmed off, the alignN arrays are used for the greybody fit.
# the console-progressbar package has been used to visualize progress here, however this can be turned off by commenting out
# lines including pb in this script.

h = 6.626e-34    # Js
c = 3e8          # m/s
k = 1.38e-23     # J/K
mu = 2.86
mH = 1.67e-24    # g
B = 2
omega = (pix250/3600)**2*3.046e-4

def factor(r,T):
    return (np.exp((h*c)/(k*T*250e-6))-1)/(np.exp((h*c)/(k*T*160e-6))-1) - r

data_ratio = align1/align2
B_ratio = data_ratio*(160/250)**(B+3)
tempmap = np.zeros(align1.shape)
pb = ProgressBar(total=align1.shape[0],prefix='Producing color-temperature map from flux ratio...',suffix='Completed',decimals=1,length=50,fill='>',zfill=' ')
for i in range(align1.shape[0]):
    pb.print_progress_bar(i)
    for j in range(align1.shape[1]):
        r = B_ratio[i,j]
        part = partial(factor,r)
        root = fsolve(part,[7])
        tempmap[i,j] = root[0]

print("\nGenerating column density map from color-temperature map...")
sigma250_reg250 = fits.PrimaryHDU(align2/(1e26*omega*ltov(250e-6)*bbf(250e-6,tempmap)*mu*mH*k_nu(250e-6)))
sigma250_reg250.header = f250.header

print("\nPerforming unsharp masking...")
sigma250_reg250_conv350 = convolveim(sigma250_reg250,kernel250_350)
msd_term3_reg250 = fits.PrimaryHDU(sigma250_reg250.data-sigma250_reg250_conv350.data)
msd_term3_reg250.header = sigma250_reg250_conv350.header

print("\nTerm 3 computed!")
fits.writeto('msd_term3_reg250.fits',msd_term3_reg250.data,msd_term3_reg250.header)

#%%

## Load msd terms

term1str = get_pkg_data_filename('msd_term1_reg250.fits')
term2str = get_pkg_data_filename('msd_term2_reg250.fits')
term3str = get_pkg_data_filename('msd_term3_reg250.fits')

term1 = fits.PrimaryHDU(fits.getdata(term1str),fits.getheader(term1str))
term2 = fits.PrimaryHDU(fits.getdata(term2str),fits.getheader(term2str))
term3 = fits.PrimaryHDU(fits.getdata(term3str),fits.getheader(term3str))

t1dat = term1.data
t2dat = term2.data
t3dat = term3.data

print("\nThe 'msd_termn_reg250.fits' data files may have NaN borders and be of different sizes.")
print("\nEnsure that the following shapes are the same post trimming: ", "Term 1: ",t1dat.shape,"Term 2: ",t2dat.shape,"Term 3: ",t3dat.shape)

#%%

# Plot processed images to check for length of borders to be trimmed off. Choose the largest border width of all images and
# trim this off all images to ensure the same spatial extent is retained in all images.

plt.subplot(1,3,1)
plt.imshow(t1dat)
plt.subplot(1,3,2)
plt.imshow(t2dat)
plt.subplot(1,3,3)
plt.imshow(t3dat)

#%%

# Trim border from all images. Here, the larger sizes of some terms must also be accounted for, and the maximal shape size
# must be selected to exclude NaN borders, and to ensure all images have equal size.

t1dat = trimbor(t1dat,[2,2,2,2])
t2dat = trimbor(t2dat,[2,2,2,2])
t3dat = trimbor(t3dat,[4,4,4,4])

#%%

# Create temporary arrays for testing alignment. In case you mess up the align arrays in any way, you can 
# always restore the original align arrays from this step.

align1 = np.copy(t1dat)
align2 = np.copy(t2dat)
align3 = np.copy(t3dat)

#%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%3+1
    
#%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs

align2 = shift(align2,'right',3)
align3 = shift(align3,'left',1)
align3 = shift(align3,'up',1)

#%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, an maximal border must be trimmed off of all images.

align1 = trimbor(align1,[3,3,3,3])
align2 = trimbor(align2,[3,3,3,3])
align3 = trimbor(align3,[3,3,3,3])

#%%

# Adding spatial scales to generate final column density map at the resolution of the SPIRE 250 map.

print("Adding terms...")

msd_cdens_reg250 = fits.PrimaryHDU(align1 + align2 + align3,term1.header)

print("Final column density map computed successfully!")
fits.writeto('msd_cdens_reg250.fits',msd_cdens_reg250.data,msd_cdens_reg250.header)

#%%

# Plot of individual terms and final column density map.

plt.subplot(2,2,1,projection=WCS(msd_cdens_reg250.header))
plt.title('Term 1')
plt.imshow(align1)
plt.subplot(2,2,2,projection=WCS(msd_cdens_reg250.header))
plt.title('Term 2')
plt.imshow(align2)
plt.subplot(2,2,3,projection=WCS(msd_cdens_reg250.header))
plt.title('Term 3')
plt.imshow(align3)
plt.subplot(2,2,4,projection=WCS(msd_cdens_reg250.header))
plt.title('Final column density map')
plt.imshow(msd_cdens_reg250.data)

#%%

# Ratio of Term 1 to final map convolved to SPIRE 500 map resolution for consistency check.

msd_cdens_reg250_conv500 = convolveim(msd_cdens_reg250,kernel250_500)

ratiohi2low = fits.PrimaryHDU(msd_cdens_reg250_conv500.data/align1,msd_cdens_reg250.header)
plt.imshow(ratiohi2low.data)

print("Mean = ",ratiohi2low.data.mean())
print("Standard deviation = ",ratiohi2low.data.std())
