#!/usr/bin/env python
import numpy as np
cimport numpy as np
import cython
from libc.math cimport sin, floor, ceil, exp, pow, M_PI

ctypedef np.float_t DTYPE_t
ctypedef np.int_t iDTYPE_t

__all__ = ["sincrebin_single", "sincrebin_single_os"]

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void getpixelshifts(double[:] sci_wl, double[:] sky_wl, long[:] mpix, double[:] dpix):
    cdef Py_ssize_t nsci = sci_wl.shape[0]
    cdef Py_ssize_t nsky = sky_wl.shape[0]
    cdef Py_ssize_t ii
    #cdef np.ndarray[iDTYPE_t, ndim=1] mpix = np.zeros(nsky, dtype=int)
    #cdef np.ndarray[DTYPE_t, ndim=1] dpix = np.zeros(nsky, dtype=float)
    cdef long j
    cdef double shift, dlam

    j = 1
    shift = 0.
    dlam = 0.
    for ii in range(nsky):
        while (j < nsci-1) and (sci_wl[j] < sky_wl[ii]):
            j += 1
        dlam = sci_wl[j]-sci_wl[j-1]
        shift = (sky_wl[ii] - sci_wl[j]) / dlam
        mpix[ii] = int(floor(shift+0.5)) + j
        dpix[ii] = shift - float(mpix[ii]) + j

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void getpixelshifts_os(double[:] sci_wl, double[:] sky_wl, long[:] mpix, double[:] dpix, double[:] dlam):
    cdef Py_ssize_t nsci = sci_wl.shape[0]
    cdef Py_ssize_t nsky = sky_wl.shape[0]
    cdef Py_ssize_t ii
    #cdef np.ndarray[iDTYPE_t, ndim=1] mpix = np.zeros(nsky, dtype=int)
    #cdef np.ndarray[DTYPE_t, ndim=1] dpix = np.zeros(nsky, dtype=float)
    cdef long j
    cdef double shift#, dlam

    j = 1
    shift = 0.
    #dlam = 0.
    for ii in range(nsky):
        while (j < nsci-1) and (sci_wl[j] < sky_wl[ii]):
            j += 1
        #dlam = sci_wl[j]-sci_wl[j-1]
        #shift = (sky_wl[ii] - sci_wl[j]) / dlam
        shift = (sky_wl[ii] - sci_wl[j]) / dlam[j]
        mpix[ii] = int(floor(shift+0.5)) + j
        dpix[ii] = shift - float(mpix[ii]) + j


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void calcsinc(double[:] sinc, long nsinc):
    cdef double SINCRAD = 6.0
    cdef long SINCBIN = 10000
    cdef double SINCDAMP = 3.25

    cdef long nsinch = int(nsinc / 2)
    cdef double dx = float((2.*SINCRAD)/(nsinc-1.))
    cdef long kk

    for kk in range(nsinch):
        x = (kk-nsinch)*dx
        if ceil(x) == x:
            sinc[kk] = 0.
        else:
            sinc[kk] = exp(-1.*pow(x/SINCDAMP, 2)) * sin(M_PI*x)/(M_PI*x)
    sinc[nsinch] = 1.
    for kk in range(nsinch+1,nsinc):
        sinc[kk] = sinc[nsinc-kk-1]

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void calcsinc_os(double[:] sinc, long nsinc):
    cdef double SINCRAD = 4.0
    cdef long SINCBIN = 10000
    cdef double SINCDAMP = 1.15

    cdef long nsinch = int(nsinc / 2)
    cdef double dx = float((2.*SINCRAD)/(nsinc-1.))
    cdef long kk

    for kk in range(nsinch):
        x = (kk-nsinch)*dx
        if ceil(x) == x:
            sinc[kk] = 0.
        else:
            sinc[kk] = exp(-1.*pow(x/SINCDAMP, 2)) * sin(M_PI*x)/(M_PI*x)
    sinc[nsinch] = 1.
    for kk in range(nsinch+1,nsinc):
        sinc[kk] = sinc[nsinc-kk-1]




@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=1] sincrebin_single(double[:] sci_wl, double[:] sky_wl, double[:] sky_flux):
    
    #parameters for the sinc kernel
    cdef double SINCRAD = 6.0
    cdef long SINCBIN = 10000
    cdef double SINCDAMP = 3.25
    cdef long nsinc = int((2*SINCRAD) * SINCBIN + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] psinc = np.zeros(nsinc, dtype=float)

    cdef Py_ssize_t nsci = sci_wl.shape[0]
    cdef Py_ssize_t nsky = sky_wl.shape[0]
    cdef np.ndarray[iDTYPE_t, ndim=1] mpix = np.zeros(nsky, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=1] dpix = np.zeros(nsky, dtype=float)
    cdef Py_ssize_t ii, kk, hh

    cdef double radius = 5.
    cdef long nkpix = int(2*radius + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] kernel = np.zeros(nkpix, dtype=float)
    
    cdef double k_offset = (float(SINCRAD) - radius)
    cdef int sign, npix
    cdef np.ndarray[DTYPE_t, ndim=1] sky_out = np.zeros(nsci, dtype=float)
    cdef double tsum, low, high, x, rsum, shift
    cdef long j, pmin, pmax#, hh, kk

    getpixelshifts(sci_wl, sky_wl, mpix, dpix) 

    pmin = mpix[0]
    pmax = mpix[nsky-1]

    calcsinc(psinc, nsinc)

    for ii in range(nsky):
        shift = dpix[ii] - k_offset

        tsum = 0.
        for kk in range(nkpix):
            x = <double>(kk-shift)*SINCBIN
            low = (psinc[<int>x])
            high = (psinc[<int>(x+1)])
            kernel[kk] = low + (high-low)*(x-floor(x))
            tsum += kernel[kk]

        #normalize kernel
        rsum = 1./tsum
        for kk in range(nkpix):
            kernel[kk] *= rsum

        if ii == 0 and pmin > -1.*radius:
            npix = <int>(pmin + radius + 1)
            sign = -1
        elif ii == nsky-1 and pmax < nsci-1+radius:
            npix = <int>(nsci-pmax+radius)
            sign = 1
        else:
            npix = 1
            sign = 1

        for hh in range(npix):
            j = mpix[ii]-<int>radius+ (sign*hh)
            for kk in range(nkpix):
                jj = j + kk
                if jj < 0 or jj >= nsci:
                    continue
                else:
                    sky_out[jj] += sky_flux[ii]*kernel[kk]

    return sky_out

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=1] sincrebin_single_os(double[:] sky_wl, double[:] sci_wl, double[:] sci_flux):
    """modified slightly to deal with the oversampling case""" 
    #parameters for the sinc kernel
    cdef double SINCRAD = 4.0
    cdef long SINCBIN = 10000
    cdef double SINCDAMP = 1.15
    cdef long nsinc = int((2*SINCRAD) * SINCBIN + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] psinc = np.zeros(nsinc, dtype=float)

    cdef Py_ssize_t nsci = sci_wl.shape[0]
    cdef Py_ssize_t nsky = sky_wl.shape[0]
    cdef np.ndarray[iDTYPE_t, ndim=1] mpix = np.zeros(nsky, dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=1] dpix = np.zeros(nsky, dtype=float)
    cdef Py_ssize_t ii, kk, hh

    cdef double radius = 2.
    cdef long nkpix = int(2*radius + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] kernel = np.zeros(nkpix, dtype=float)
    
    cdef double k_offset = (float(SINCRAD) - radius)
    cdef int sign, npix
    #cdef np.ndarray[DTYPE_t, ndim=1] sky_out = np.zeros(nsci, dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=1] sky_out = np.zeros(nsky, dtype=float)
    cdef double tsum, low, high, x, rsum, shift
    cdef long j, pmin, pmax#, hh, kk

    getpixelshifts(sci_wl, sky_wl, mpix, dpix) 

    pmin = mpix[0]
    pmax = mpix[nsky-1]

    calcsinc_os(psinc, nsinc)

    for ii in range(nsky):
        shift = dpix[ii] - k_offset

        tsum = 0.
        for kk in range(nkpix):
            x = <double>(kk-shift)*SINCBIN
            low = (psinc[<int>x])
            high = (psinc[<int>(x+1)])
            kernel[kk] = low + (high-low)*(x-floor(x))
            tsum += kernel[kk]

        #normalize kernel
        rsum = 1./tsum
        for kk in range(nkpix):
            kernel[kk] *= rsum

        if ii == 0 and pmin > -1.*radius:
            npix = <int>(pmin + radius + 1)
            sign = -1
        elif ii == nsky-1 and pmax < nsci-1+radius:
            npix = <int>(nsci-pmax+radius)
            sign = 1
        else:
            npix = 1
            sign = 1

        for hh in range(npix):
            j = mpix[ii]-<int>radius+ (sign*hh)
            for kk in range(nkpix):
                jj = j + kk
                if jj < 0 or jj >= nsci:
                    continue
                else:
                    #sky_out[jj] += sky_flux[ii]*kernel[kk]
                    #sky_out[jj] += sci_flux[mpix[ii]]*kernel[kk]
                    sky_out[ii] += sci_flux[jj]*kernel[kk]

    return sky_out





@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=3] sincrebin_cube(double[:] sci_wl, double[:] sky_wl, double[:,:,:] sky_flux):
    cdef Py_ssize_t nx = sky_flux.shape[1]
    cdef Py_ssize_t ny = sky_flux.shape[2]
    cdef Py_ssize_t nz = sky_flux.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] sky_out = np.zeros((nz, nx, ny), dtype=float)
    cdef Py_ssize_t ii, jj
   
    for ii in range(nx):
        for jj in range(ny):
            sky_out[:,ii,jj] = sincrebin_single(sci_wl, sky_wl, sky_flux[:,ii,jj])
    return sky_out







