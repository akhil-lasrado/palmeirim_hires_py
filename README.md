# palmeirim_hires_py
Python implementation for obtaining a high resolution (18.2") column density map using Herschel PACS and SPIRE multiwavelength images first described in [Palmeirim et al (2013)](https://doi.org/10.1051/0004-6361/201220500). Unsharp masking for the spatial scales at 250 and 350 micron restore the high spacial frequency features lost in convolving all Herschel wavelengths to the (36.4") resolution of the 500 micron image. The resulting map is obtained at the resolution of the SPIRE 250 micron map, which is an improvement by a factor of two over the standard procedure.

Required inputs:

- Herschel PACS and SPIRE image cutouts (.fits) of the same angular extent, including the 160, 250, 350, 500 micron cutouts of the region.
- Smoothing kernels for convolution, for the smoothing from all wavelengths to all longer wavelengths, in total this should be 6 kernel .fits files (the kernels by [Aniano et al (2011)](https://doi.org/10.1086/662219) are recommended).

The multiscale_decomposition.py routine depends on functions included in the useful_routines.py and greybody_fit.py files.
