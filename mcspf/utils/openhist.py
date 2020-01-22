#!/usr/bin/env python
import numpy as np
import pylab as pl
import scipy.stats as sp

def get_fraction(sample,total, npts=None):
    sample, total = float(sample), float(total)
#    sample, total = np.asarray(sample, dtype=float), np.asarray(total, dtype=float)
    if npts is None: 
        dist = sp.binom(total, sample/total)
    else:
        dist = sp.binom(npts, sample/total)
    if sample == 0:
        med_fraction=0.0
        err = 0.0
    else:
        if npts is None:
            med_fraction = dist.median()/total
            err = med_fraction - dist.ppf(0.16)/total
        else:
            med_fraction = dist.median()/npts
            err = med_fraction - dist.ppf(0.16)/npts
    return med_fraction, err

def openhist(array, nbins, range=None, norm=1., color='black', ec=None, fill=False, style='solid', linewidth=1.0, alpha=1.0, errors=False, err_color=None, log=False, weights=None, zorder=3, offset=0.0, logx=False, output_raw=True, yoffset=0.0, axis='xaxis', soften=False, no_norm=False, dens=False):
        "plot a histogram normalised by norm"
        if logx:
            nbins = np.logspace(range[0],range[1],nbins+1)
            array = 10**array
        #print nbins
        if weights is None:
            if not np.iterable(nbins):
                vals, bins = np.histogram(array, bins=nbins, range=range)
            else:
                vals, bins = np.histogram(array, bins=nbins)
        else:
            if not np.iterable(nbins):
                vals, bins = np.histogram(array, bins=nbins, range=range, weights=weights)
            else:
                vals, bins = np.histogram(array, bins=nbins, weights=weights)

        centers = (bins[1:] + bins[:-1])/2.
        
        ##dy = np.sqrt(np.maximum(vals,1.))/float(norm)
        #dy = norm*np.sqrt(vals)/float(vals.sum())
        if no_norm:
            nvals = np.copy(vals)

            total = vals.sum()
            npts = len(array)
            nerrs = np.empty(len(vals), dtype=float)
            for iter, val in enumerate(vals):
                bog, nerrs[iter] = get_fraction(val, total, npts=npts) 

        else:
            vals = norm*vals/float(vals.sum())

            #jiggery-pokery for errors and fractions
            total = vals.sum()
            npts = len(array)
        
            nvals = np.empty(len(vals), dtype=float)
            nerrs = np.empty(len(vals), dtype=float)
            for iter, val in enumerate(vals):
                nvals[iter], nerrs[iter] = get_fraction(norm*val, norm*total, npts=npts) 
            #if soften:
            #    nerrs[iter] = np.sqrt(nerrs[iter]**2 + 10/float(vals.sum())


            nvals = norm*nvals#/float(npts)
            nerrs = norm*nerrs#/float(npts)

        if dens: #convert to density, scaling by bin width
            wbin = np.diff(bins)
            nvals /= wbin
            nerrs /= wbin

        if soften:
            nerrs = np.sqrt(nerrs**2 + 10./npts**2)

        x = np.zeros(2*len(bins), np.float)
        y = np.zeros(2*len(bins), np.float)

        #print bins, nvals
        x[0::2], x[1::2] = bins, bins
        y[1:-1:2], y[2::2] = nvals, nvals

        if log:
                y = np.log10(y)
                y[~np.isfinite(y)] = -10
                if ec is None:
                        if fill:
                                pl.fill(x+offset, (y), fill=fill, facecolor=color, edgecolor=color, closed=False, alpha=alpha, zorder=zorder)
                        else:
                                pl.fill(x+offset, (y), fill=False, edgecolor=color, linestyle=style, closed=False, lw=linewidth, zorder=zorder)
                else:
                        if fill:
                                pl.fill(x+offset, (y), fill=fill, facecolor=color, edgecolor=ec, closed=False, alpha=alpha, zorder=zorder)
                        else:
                                pl.fill(x+offset, (y), fill=False, edgecolor=ec, linestyle=style, closed=False, lw=linewidth, zorder=zorder)
                
        else:
            if axis == 'yaxis':
                if ec is None:
                        if fill:
                                pl.fill(y+yoffset, x+offset, fill=fill, facecolor=color, edgecolor=color, closed=False, alpha=alpha, zorder=zorder)
                        else:
                                pl.fill(y+yoffset, x+offset, fill=False, edgecolor=color, linestyle=style, closed=False, lw=linewidth, zorder=zorder)
                else:
                        if fill:
                                pl.fill(y+yoffset, x+offset, fill=fill, facecolor=color, edgecolor=ec, closed=False, alpha=alpha, zorder=zorder)
                        else:
                                pl.fill(y+yoffset, x+offset, fill=False, edgecolor=ec, linestyle=style, closed=False, lw=linewidth, zorder=zorder)
            else:       
                if ec is None:
                        if fill:
                                pl.fill(x+offset, y+yoffset, fill=fill, facecolor=color, edgecolor=color, closed=False, alpha=alpha, zorder=zorder)
                        else:
                                pl.fill(x+offset, y+yoffset, fill=False, edgecolor=color, linestyle=style, closed=False, lw=linewidth, zorder=zorder)
                else:
                        if fill:
                                pl.fill(x+offset, y+yoffset, fill=fill, facecolor=color, edgecolor=ec, closed=False, alpha=alpha, zorder=zorder)
                        else:
                                pl.fill(x+offset, y+yoffset, fill=False, edgecolor=ec, linestyle=style, closed=False, lw=linewidth, zorder=zorder)

        if errors:
                if err_color is None:
                        pl.errorbar(centers+offset, nvals+yoffset, nerrs, fmt=None, lw=linewidth, ecolor=color, zorder=zorder, capsize=0)
                else:
                        pl.errorbar(centers+offset, nvals+yoffset, nerrs, fmt=None, lw=linewidth, ecolor=err_color, zorder=zorder, capsize=0)

        if output_raw:
            return x, y
        else:
            #print len(bins), len(nvals)
            return bins, nvals, centers

        #return bins, nvals
