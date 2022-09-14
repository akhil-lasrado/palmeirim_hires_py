# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:54:02 2022 v1.0*

@author: Akhil
"""
import astropy
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import convolve
from astropy.convolution import convolve_fft
from scipy.ndimage import zoom
from reproject import *
from reproject import reproject_adaptive
from reproject import reproject_exact
from scipy import ndimage

#%%

# Function to extract the pixel scale of an image from its header, output in arcseconds.

def find_pixel_scale(header):
    
    """Finds the value of the image pixel scale from the image headers
    Inputs:
        header: Header of the image
    Output:
        pixel_scale: Pixel scale of the image in arcsec/pixel
    """
    
    pixel_scale = None
    keys = [key for key in header.keys()]

    if ('CD1_1' in keys) and ('CD1_2' in keys):
        pixel_scale = np.sqrt(header['CD1_1']**2 + header['CD1_2']**2)*3600

    elif ('PC1_1' in keys) and ('PC1_2' in keys):
        pixel_scale = np.sqrt(header['PC1_1']**2 + header['PC1_2']**2)*3600

    elif 'PXSCAL_1' in keys:
        pixel_scale = abs(header['PXSCAL_1'])

    elif 'PIXSCALE' in keys:
        pixel_scale = header['PIXSCALE']

    elif 'SECPIX' in keys:
        pixel_scale = header['SECPIX']

    elif 'CDELT1' in keys:
        pixel_scale = abs(header['CDELT1'])*3600

    else:
        print('Unable to get pixel scale from image header')
        while True:
            pixel_scale = input('Please input the pixel scale value in \
                                arcsec per pixel')
            try:
                pixel_scale = float(pixel_scale)
                return pixel_scale
            except ValueError:
                pass

    return pixel_scale

#%%

# Function to convolve a fits image using a specified kernel. Output is the convolved fits image.

def convolveim(file,kernel):
    
    image = file.data
    kernel_dat = fits.getdata(kernel)

    hdui = file.header
    hduk = fits.getheader(kernel)
    
    pixel_scale_i = find_pixel_scale(hdui)
    pixel_scale_k = find_pixel_scale(hduk)
    
    # resize kernel_dat if necessary
    if pixel_scale_k != pixel_scale_i:
        ratio = pixel_scale_k / pixel_scale_i
        size = ratio*kernel_dat.shape[0]
        # ensure an odd kernel_dat
        if round(size) % 2 == 0:
            size += 1
            ratio = size / kernel_dat.shape[0]
        kernel_dat = zoom(kernel_dat, ratio) / ratio**2

    convolved_image = convolve_fft(image,kernel_dat,boundary='interpolate',nan_treatment='interpolate')
    
    convolved_file = fits.PrimaryHDU(convolved_image)
    convolved_file.header = hdui

    return convolved_file


#%%

# Function for adaptive resampling of a fits image to the pixel size of a template fits file. Output is the regridded fits image.

def regridim(file,template):
    
    f_data = np.copy(file.data)
    f_header = file.header.copy()
    
    cdelt_new = find_pixel_scale(template.header)
    cdelt_old = find_pixel_scale(f_header)
    
    size_ax_y = template.shape[0]#round((f_data.shape[0]*cdelt_old)/cdelt_new)
    size_ax_x = template.shape[1]#round((f_data.shape[1]*cdelt_old)/cdelt_new)
    
    data = np.random.random((size_ax_y,size_ax_x))
    hdu = fits.PrimaryHDU(data=data)
    hdu.header = f_header
    hdu.header['CRPIX1'] =  int(size_ax_x/2)
    hdu.header['CDELT1'] =  -cdelt_new/3600
    hdu.header['CRPIX2'] =  int(size_ax_y/2)
    hdu.header['CDELT2'] =  cdelt_new/3600
    hdu.header['CD1_1'] = -cdelt_new/3600
    hdu.header['CD1_2'] = 0.0
    hdu.header['CD2_1'] = 0.0
    hdu.header['CD2_2'] = cdelt_new/3600

    array, footprint = reproject_adaptive(file, hdu.header)
    
    regridded_file = fits.PrimaryHDU(array)
    regridded_file.header = hdu.header

    return regridded_file

#%%

# Function to convert the units of an image from Mjy/sr to Jy/pixel. Returns a fits image as the output.

def mjps2jpp(file):
    
    data = file.data
    header = file.header

    pixdelt = find_pixel_scale(header)
    sr2arcs = 4.25e10

    fact = (pixdelt**2)*1e6/sr2arcs
    datapix = data*fact

    output_file = fits.PrimaryHDU(datapix)
    output_file.header = header
    
    return output_file
    
#%%

# Function to convert the units of an image from Jy/beam to Jy/pixel, provided the beamarea of the map is specified. Returns a fits image as the output.

def jpb2jpp(file,beamarea):
    
    data = file.data
    header = file.header

    pixdelt = find_pixel_scale(header)
   
    sig2fwhm = np.sqrt(8*np.log(2))
    beamsize = (2*np.pi*beamarea)/(sig2fwhm**2)
    fact = pixdelt**2/beamsize
    datapix = data*fact
    
    output_file = fits.PrimaryHDU(datapix)
    output_file.header = header
    
    return output_file

#%%

# Function to shift array in any direction in pixel units.

def shift(array,shift_dir,shift_len):
    a = np.copy(array)
    if shift_dir=='up':
        a = a[shift_len:,:]
        a = np.append(a,np.zeros((shift_len,array.shape[1])),0)
        return a
    elif shift_dir=='down':
        a = a[:-shift_len,:]
        a = np.append(np.zeros((shift_len,array.shape[1])),a,0)
        return a
    elif shift_dir=='left':
        a = a[:,shift_len:]
        a = np.append(a,np.zeros((array.shape[0],shift_len)),1)
        return a
    elif shift_dir=='right':
        a = a[:,:-shift_len]
        a = np.append(np.zeros((array.shape[0],shift_len)),a,1)
        return a
    else:
        print('Enter valid direction: up, down, left, right')

#%%

def trimborall(array,trim_len):
    a = np.copy(array)
    a = a[trim_len:-trim_len,trim_len:-trim_len]
    return a
