#!/usr/bin/env python
import numpy as np
cimport numpy as np
import cython
from libc.math cimport fabs

ctypedef np.float_t DTYPE_t

__all__ = ["getmag_spec"]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_t tsum(DTYPE_t[:] yarr, DTYPE_t[:] xarr) nogil:
    """ Simple trapezoid integration
    """
    cdef ssize_t nn = xarr.shape[0]
    cdef DTYPE_t isum = 0
    cdef ssize_t ii

    for ii in range(nn-1):
        isum += fabs(xarr[ii+1]-xarr[ii])*(yarr[ii+1]+yarr[ii])/2.
    return isum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def getmag_spec(DTYPE_t[:] wl, DTYPE_t[:,:] tnum, long nbands, DTYPE_t[:] flux_out):
    """compute flux given an input spectrum and transmission curves
    """
    cdef ssize_t ii
    #cdef np.ndarray[DTYPE_t, ndim=1] flux_out = np.zeros(nbands, dtype=np.float)

    for ii in range(nbands):
        flux_out[ii] = tsum(tnum[:,ii], wl)

    return

