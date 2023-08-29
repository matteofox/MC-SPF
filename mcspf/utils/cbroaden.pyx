#!/usr/bin/env python
import numpy as n
cimport numpy as n
from cpython cimport bool

DTYPE = float
ctypedef double DTYPE_t
ctypedef long DTYPE_i

def broaden(n.ndarray[DTYPE_t, ndim=1] flux, n.ndarray[DTYPE_t, ndim=1] sigma, bool variance=False ):
    #sigma should be the broadening in pixels to apply for each pixel, should have the same length as flux/wl

    cdef int N_elem = len(flux)
    cdef int pix, npix

    cdef n.ndarray[DTYPE_t, ndim=1] out_spec = n.zeros(N_elem, dtype=DTYPE)
    cdef n.ndarray[DTYPE_t, ndim=1] norm_spec = n.ones(N_elem, dtype=DTYPE) 
    cdef n.ndarray[DTYPE_t, ndim=1] ttpix = n.zeros(N_elem, dtype=DTYPE)

    cdef n.ndarray[DTYPE_t, ndim=1]  width = 2.355*3*sigma #sample the distribution out to 3*FWHM
    cdef n.ndarray[DTYPE_i, ndim=1]  spix = n.asarray(n.floor(sigma*3.), dtype=int)
    cdef n.ndarray[DTYPE_t, ndim=1]  s2 = sigma**2

    if variance:
        for pix in range(N_elem):
            if width[pix]/2.355 < 1.0:
                out_spec[pix] = flux[pix]
                ttpix[pix] = norm_spec[pix] 
            else:
                if pix+1 < width[pix]:
                    for npix in range(0-pix,spix[pix]):
                        out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*((1./((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1.*(npix**2/(2*s2[pix]))))**2
                        ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*((1./((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1.*(npix**2/(2*s2[pix]))))**2
                elif width[pix] <= pix+1 <= N_elem-width[pix]+1:
                    for npix in range(-1*spix[pix]+1,spix[pix]):    
                        out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*((1./((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1.*(npix**2/(2*s2[pix]))))**2
                        ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*((1./((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1.*(npix**2/(2*s2[pix]))))**2
                else:     
                    for npix in range(-1*spix[pix]+1,N_elem-pix):
                        out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*((1./((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1.*(npix**2/(2*s2[pix]))))**2
                        ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*((1./((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1.*(npix**2/(2*s2[pix]))))**2


    else:
        for pix in range(N_elem):
            if width[pix]/2.355 < 1.0:
                out_spec[pix] = flux[pix]
                ttpix[pix] = norm_spec[pix]
            else:
                if pix+1 < width[pix]:
                    for npix in range(0-pix,spix[pix]):
                        out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*(1/((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1*(npix**2/(2*s2[pix])))
                        ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*(1/((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1*(npix**2/(2*s2[pix])))
                elif width[pix] <= pix+1 <= N_elem-width[pix]+1:
                    for npix in range(-1*spix[pix]+1,spix[pix]):    
                        out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*(1/((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1*(npix**2/(2*s2[pix])))
                        ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*(1/((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1*(npix**2/(2*s2[pix])))
                else:     
                    for npix in range(-1*spix[pix]+1,N_elem-pix):
                        out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*(1/((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1*(npix**2/(2*s2[pix])))
                        ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*(1/((2*3.14159265359*s2[pix])**0.5))*2.71828182846**(-1*(npix**2/(2*s2[pix])))


    return out_spec/ttpix



