# palmeirim_hires_py

Python program for obtaining a high resolution (18.2") column density map using Herschel PACS and SPIRE multiwavelength images first described in [Palmeirim et al (2013)](https://doi.org/10.1051/0004-6361/201220500)([pdf](https://www.aanda.org/articles/aa/pdf/2013/02/aa20500-12.pdf)). Unsharp masking for the spatial scales at 250 and 350 micron restore the high spacial frequency features lost in convolving all Herschel wavelengths to the (36.4") resolution of the 500 micron image. The resulting map is obtained at the resolution of the SPIRE 250 micron map, which is an improvement by a factor of two over the standard procedure.

- Herschel PACS and SPIRE image cutouts (.fits) of the same angular extent, including the 160, 250, 350, 500 micron cutouts of the region. Units must be in Jy/pixel. Functions for conversions from Jy/beam (beam size must be given) and MJy/sr to Jy/pixel are included in the useful_routines.py script.

- Smoothing kernels for convolution, for the smoothing from all wavelengths to all longer wavelengths, in total this should be 6 kernel .fits files (the kernels by [Aniano et al (2011)](https://doi.org/10.1086/662219)([pdf](https://iopscience.iop.org/article/10.1086/662219/pdf)) are recommended). Download links for the required kernels are provided in the kernels folder.

- DEPENDENCIES INCLUDE: numpy, astropy, scipy, functools, reproject, console-progressbar (Installation: pip install console-progressbar) (for visualization)

- Enter 160,250,350,500 micron image (with units in Jy/pixel!) file names in ./+spatial_filt.cfg file

- It is recommended to run the code section-by-section (Shift+Enter in spyder), and to run the code in a new folder 
  containing the three scripts (spatial_filt.py, greybody_fit.py, and useful_routines.py) along with the ./+spatial_filt.cfg file
  to prevent overwriting of existing files. The kernels folder must also be located here.
