# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:33:01 2022 v1.0*

@author: Akhil Lasrado, akhil.lasrado@hotmail.com, IISER Kolkata


SPATIAL FILTERING ALGORITHM FOR HERSCHEL HI-GAL IMAGES (160,250,350,500 MIC)
BASED ON THE PRESCRIPTION DESCRIBED IN PALMEIRIM ET AL (2013) (https://doi.org/10.1051/0004-6361/201220500).

> DEPENDENCIES INCLUDE: numpy, astropy, scipy, functools, reproject, console-progressbar 
  (console-progressbar is for visualizing progress in case of large images. This can be avoided by commenting out lines containing 'pb' in greybody_fit.py) [Installation: pip install console-progressbar]

> Enter 160,250,350,500 micron image (with units in MJy/sr!) file names in ./+spatial_filt.cfg file

> It is recommended to run the code section-by-section (Shift+Enter in spyder), and to run the code in a new folder 
> containing the three scripts (spatial_filt.py, greybody_fit.py, and useful_routines.py) along with the ./+spatial_filt.cfg file
> to prevent overwriting of existing files.

"""

#%%

import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import matplotlib.pyplot as plt
from greybody_fitt import * #NOQA
from useful_routinest import * #NOQA
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
# The convolution step can introduce boundary effects due to interpolation, and these must be inspected
# and trimmed off before the greybody fit. A provision for this is provided in the following sections.

print("Computing term 1...")

print("Processing f160...")
f160_conv500 = convolveim(f160, kernel160_500)
f160_conv500_reg250 = regridim(f160_conv500,f500)
findat1 = f160_conv500_reg250.data

print("Processing f250...")
f250_conv500 = convolveim(f250, kernel250_500)
f250_conv500_reg250 = regridim(f250_conv500,f500)
findat2 = f250_conv500_reg250.data

print("Processing f350...")
f350_conv500 = convolveim(f350, kernel350_500)
f350_conv500_reg500 = regridim(f350_conv500,f500)
findat3 = f350_conv500_reg500.data

print("Processing f500...")
findat4 = f500.data

#%%%

# Plot processed images to check for boundary effects in images. Choose the largest border width of all images 
# that includes boundary effects and trim this lengths off of all images to ensure the same spatial extent is 
# retained in all images.

plt.subplot(2,2,1)
plt.imshow(findat1)
plt.subplot(2,2,2)
plt.imshow(findat2)
plt.subplot(2,2,3)
plt.imshow(findat3)
plt.subplot(2,2,4)
plt.imshow(findat4)

#%%%

# Trim border of 4 pixel (for example) from all images

trim_len = 4
findat1 = trimborall(findat1,trim_len)
findat2 = trimborall(findat2,trim_len)
findat3 = trimborall(findat3,trim_len)
findat4 = trimborall(findat4,trim_len)

#%%%

# Create temporary arrays for testing alignment. In case you would like to return to the original alignment, you
# can always restore the original align arrays from this step.

align1 = np.copy(findat1)
align2 = np.copy(findat2)
align3 = np.copy(findat3)
align4 = np.copy(findat4)

#%%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%4+1
    
#%%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs.

align4 = shift(align4,'up',1)
align4 = shift(align4,'left',1)

#%%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, a border length equal to the largest shift introduced in 
# among all images should be trimmed.

max_shift = 1
align1 = trimborall(align1,max_shift)
align2 = trimborall(align2,max_shift)
align3 = trimborall(align3,max_shift)
align4 = trimborall(align4,max_shift)

#%%%

# Once the images have been aligned, and 0 borders have been trimmed off, the alignN arrays are used for the greybody fit.
# the console-progressbar package has been used to visualize progress here, however this can be turned off by commenting out
# lines containing 'pb' in the greybody_fit.py file.

sr2arcs = 4.25e10
fact = (pix500**2)*1e6/sr2arcs

align1 = align1*fact
align2 = align2*fact
align3 = align3*fact
align4 = align4*fact

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
sigma500_reg250 = regridim(sigma500_reg500,f250)
msd_term1_reg250 = sigma500_reg250

print("\nTerm 1 computed!")
fits.writeto('sigma500_reg250.fits',sigma500_reg500.data,sigma500_reg500.header)
fits.writeto('sf_term1_reg250.fits',msd_term1_reg250.data,msd_term1_reg250.header)

#%%

## TERM 2 COMPUTATION

# This step will prepare the images for a pixel-by-pixel fit at the resolution of the 350 micron image,

print("Computing term 2...")

print("Processing f160...")
f160_conv350 = convolveim(f160, kernel160_350)
f160_conv350_reg350 = regridim(f160_conv350,f350)
findat1 = f160_conv350_reg350.data

print("Processing f250...")
f250_conv350 = convolveim(f250, kernel250_350)
f250_conv350_reg350 = regridim(f250_conv350,f350)
findat2 = f250_conv350_reg350.data

print("Processing f350...")
findat3 = f350.data

#%%%

# Plot processed images to check for boundary effects in images. Choose the largest border width of all images 
# that includes boundary effects and trim this lengths off of all images to ensure the same spatial extent is 
# retained in all images.

plt.subplot(1,3,1)
plt.imshow(findat1)
plt.subplot(1,3,2)
plt.imshow(findat2)
plt.subplot(1,3,3)
plt.imshow(findat3)

#%%%

# Trim border of 1 pixel (for example) from all images

trim_len = 1
findat1 = trimborall(findat1,trim_len)
findat2 = trimborall(findat2,trim_len)
findat3 = trimborall(findat3,trim_len)
findat4 = trimborall(findat4,trim_len)

#%%%

# Create temporary arrays for testing alignment. In case you would like to return to the original alignment, you
# can always restore the original align arrays from this step.

align1 = np.copy(findat1)
align2 = np.copy(findat2)
align3 = np.copy(findat3)

#%%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%3+1
    
#%%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs.

align3 = shift(align3,'left',1)
align3 = shift(align3,'up',1)

#%%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, a border length equal to the largest shift introduced in 
# among all images should be trimmed.

max_shift = 1
align1 = trimborall(align1,max_shift)
align2 = trimborall(align2,max_shift)
align3 = trimborall(align3,max_shift)

#%%%

# Once the images have been aligned, and 0 borders have been trimmed off, the alignN arrays are used for the greybody fit.
# the console-progressbar package has been used to visualize progress here, however this can be turned off by commenting out
# lines containing 'pb' in the greybody_fit.py file.

sr2arcs = 4.25e10
fact = (pix350**2)*1e6/sr2arcs

align1 = align1*fact
align2 = align2*fact
align3 = align3*fact

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
msd_term2_reg250 = regridim(msd_term2_reg350,f250)

print("\nTerm 2 computed!")
fits.writeto('sigma350_reg250.fits',sigma350_reg350.data,sigma350_reg350.header)
fits.writeto('sf_term2_reg250.fits',msd_term2_reg250.data,msd_term2_reg250.header)

#%%

## TERM 3 COMPUTATION

# This step will prepare the images for creating a column density map from a color-temperature map at the resolution of the 250 micron image.

print("Computing term 3...")

print("Processing f160...")
f160_conv250 = convolveim(f160, kernel160_250)
f160_conv250_reg250 = regridim(f160_conv250,f250)
findat1 = f160_conv250_reg250.data

print("Processing f250...")
findat2 = f250.data

#%%%

# Plot processed images to check for boundary effects in images. Choose the largest border width of all images 
# that includes boundary effects and trim this lengths off of all images to ensure the same spatial extent is 
# retained in all images.

plt.subplot(1,2,1)
plt.imshow(findat1)
plt.subplot(1,2,2)
plt.imshow(findat2)

#%%%

# Trim border of 2 pixel from all images (for example)

trim_len = 2
findat1 = trimborall(findat1,trim_len)
findat2 = trimborall(findat2,trim_len)

#%%%

# Create temporary arrays for testing alignment. In case you would like to return to the original alignment, you
# can always restore the original align arrays from this step.

align1 = np.copy(findat1)
align2 = np.copy(findat2)

#%%%

# Blink to check alignment. In order to stop the loop, the console/kernel must be interrupted (Ctrl+C in spyder).

k=1
while True:
    plt.imshow(eval('align%d'%(k)))
    plt.title('align%d'%(k))
    plt.pause(0.5)
    plt.clf()
    k=k%2+1
    
#%%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs.

#align2 = shift(align2,'right',1)

#%%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, a border length equal to the largest shift introduced in 
# among all images should be trimmed.

max_shift = 1
align1 = trimborall(align1,max_shift)
align2 = trimborall(align2,max_shift)

#%%%

# Once the images have been aligned, and 0 borders have been trimmed off, the alignN arrays are used for the greybody fit.
# the console-progressbar package has been used to visualize progress here, however this can be turned off by commenting out
# lines containing 'pb' in the script in this section.

sr2arcs = 4.25e10
fact = (pix250**2)*1e6/sr2arcs

align1 = align1*fact
align2 = align2*fact

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
pb = ProgressBar(total=align1.shape[0],prefix='Producing color-temperature map...',suffix='Completed',decimals=1,length=50,fill='>',zfill=' ')
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
msd_term3_reg250 = regridim(msd_term3_reg250,f250)

print("\nTerm 3 computed!")
fits.writeto('sigma250_reg250.fits',sigma250_reg250.data,sigma250_reg250.header)
fits.writeto('sf_term3_reg250.fits',msd_term3_reg250.data,msd_term3_reg250.header)

#%%

## ADDING OF SPATIAL SCALES

# Load spatial scale terms. The 'sf_termn_reg250.fits' data files are likely to contain NaN values
# at the edges. These must be trimmed off before adding the scales together.

term1str = get_pkg_data_filename('sf_term1_reg250.fits')
term2str = get_pkg_data_filename('sf_term2_reg250.fits')
term3str = get_pkg_data_filename('sf_term3_reg250.fits')

term1 = fits.PrimaryHDU(fits.getdata(term1str),fits.getheader(term1str))
term2 = fits.PrimaryHDU(fits.getdata(term2str),fits.getheader(term2str))
term3 = fits.PrimaryHDU(fits.getdata(term3str),fits.getheader(term3str))

t1dat = term1.data
t2dat = term2.data
t3dat = term3.data

#%%%

# Plot spatial scale terms to check for length of borders to be trimmed off.

plt.subplot(1,3,1)
plt.imshow(t1dat)
plt.subplot(1,3,2)
plt.imshow(t2dat)
plt.subplot(1,3,3)
plt.imshow(t3dat)

#%%%

# Trim border from all images. Here, the larger sizes of some terms must also be accounted for, and the maximal shape size
# must be selected to exclude NaN borders, and to ensure all images have equal size.

trim_len = 22
t1dat = trimborall(t1dat,trim_len)
t2dat = trimborall(t2dat,trim_len)
t3dat = trimborall(t3dat,trim_len)

#%%%

# Create temporary arrays for testing alignment. In case you would like to return to the original alignment, you
# can always restore the original align arrays from this step.

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
    
#%%%

# Shift arrays for alignment if needed, ie emission peaks must be ensured to overlap in all images.
# The shift function in useful_routines.py allows images to be shifted 'up', 'down', 'left', and 'right', 
# by some pixel units. This results in the addition of a border of value 0 on the opposite side, 
# which must be trimmed off before the greybody fit is performed. 

# The following are example inputs.

align2 = shift(align2,'left',7)
align2 = shift(align2,'up',7)
align3 = shift(align3,'left',10)
align3 = shift(align3,'up',10)

#%%%

# Trim borders again to exclude zero values. The trimbor function will trim off a positive sized value from an array,
# and to preserve the spatial extent and retain alignment, an maximal border must be trimmed off of all images.

max_shift = 15
align1 = trimborall(align1,max_shift)
align2 = trimborall(align2,max_shift)
align3 = trimborall(align3,max_shift)

#%%%

# Adding spatial scales to generate final column density map at the resolution of the SPIRE 250 map.

print("Adding terms...")

msd_cdens_reg250 = fits.PrimaryHDU(align1 + align2 + align3,term1.header)

fits.writeto('sf_cdens_reg250.fits',msd_cdens_reg250.data,msd_cdens_reg250.header)

#%%

## PLOT OF INDIVIDUAL TERMS AND FINAL COLUMN DENSITY MAP

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

#%%%

# Ratio of Term 1 to final map convolved to SPIRE 500 map resolution for consistency check.
# Trimming is provided to exclude the boundary effect from convolution.

msd_cdens_reg250_conv500 = convolveim(msd_cdens_reg250,kernel250_500)

trim_len = 5
hires_dat = trimborall(msd_cdens_reg250.data,trim_len)
lowres_dat = trimborall(align1,trim_len)

ratiohi2low = fits.PrimaryHDU(hires_dat/lowres_dat,msd_cdens_reg250.header)
plt.imshow(ratiohi2low.data)

print("Mean = ",ratiohi2low.data.mean())
print("Standard deviation = ",ratiohi2low.data.std())
