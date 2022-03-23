# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:54:02 2022

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
from reproject import reproject_interp
from reproject import reproject_adaptive
from reproject import reproject_exact

#%%

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
            pixel_scale = input('Plesae input the pixel scale value in \
                                arcsec per pixel')
            try:
                pixel_scale = float(pixel_scale)
                return pixel_scale
            except ValueError:
                pass

    return pixel_scale
#%%

## CONVOLVE

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

    convolved_image = convolve_fft(image,kernel_dat,boundary='wrap', nan_treatment='interpolate', normalize_kernel=True, preserve_nan=True)
    
    convolved_file = fits.PrimaryHDU(convolved_image)
    convolved_file.header = hdui

    return convolved_file


#%%

## REGRID

def regridim(file,cdelt_new):
    
    f_data = file.data
    f_header = file.header
    f_hdu = fits.PrimaryHDU(f_data,f_header)
    
    cdelt_old = find_pixel_scale(f_header)
    
    size_ax_y = int((f_data.shape[0]*cdelt_old)/cdelt_new)
    size_ax_x = int((f_data.shape[1]*cdelt_old)/cdelt_new)
    
    data = np.random.random((size_ax_y,size_ax_x))
    hdu = fits.PrimaryHDU(data=data)
    header = hdu.header
    header['CTYPE1'] = 'GLON-TAN'
    header['CRPIX1'] =  int(size_ax_x/2)
    header['CRVAL1'] =  f_header['CRVAL1']
    header['CDELT1'] =  -cdelt_new/3600 # Enter (-) pixel size required in arcseconds
    header['CUNIT1'] = 'deg     '
    header['CTYPE2'] = 'GLAT-TAN'
    header['CRPIX2'] =  int(size_ax_y/2)
    header['CRVAL2'] =  f_header['CRVAL2']
    header['CDELT2'] =  cdelt_new/3600 # Enter (+) pixel size required in arcseconds
    header['CUNIT2'] = 'deg     '

    array, footprint = reproject_exact(f_hdu, header)
    
    regridded_file = fits.PrimaryHDU(array)
    regridded_file.header = header
    
    return regridded_file

#%%

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

def gen_cube(sub_data,n_data):

    data_naxis1 = sub_data.shape[0]
    data_naxis2 = sub_data.shape[1]

    data = np.zeros([data_naxis1,data_naxis2,n_data])
    for i in range(1,n_data+1):
        print(type('dat%d'%(i)))
        data[:,:,i-1] = eval('dat%d'%(i))
        
    return data
