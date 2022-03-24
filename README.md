# palmeirim_hires_py
Python program implementation of the obtaining a high resolution column density map using Herschel PACS and SPIRE multiwavelength images.

Required inputs:

> Herschel PACS and SPIRE image cutouts (.fits) of the same angular extent, including the 160, 250, 350, 500 micron cutouts of the region.
> Smoothing kernels for convolution, for the smoothing from all wavelengths to all longer wavelengths, in total this should be 6 kernel .fits files (Kernels by Aniano et al (2011) are recommended).
> Pixel sizes (in arcseconds) of each of the cutouts.
