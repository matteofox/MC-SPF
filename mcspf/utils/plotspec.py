#!/usr/bin/env python
import string, math, os
import numpy as n
import matplotlib.pyplot as pl

def plotspec(wl, flux, color='black', style='solid', lw=1.0, mask=None, var=None, zorder=None, noplot=False):
       
    dwl = wl[1] - wl[0]

    wl_edges = n.r_[wl - dwl/2., wl[-1] + dwl/2.]
    npts = len(wl_edges)

    spec_x = n.zeros(2*npts, float)
    spec_y = n.zeros(2*npts, float)

    spec_x[0::2], spec_x[1::2] = wl_edges, wl_edges
    spec_y[1:-1:2], spec_y[2::2] = flux, flux

    if var is not None:
        var_y = n.zeros(2*npts, float)
        var_y[1:-1:2], var_y[2::2] = var, var
        var_y = var_y[1:-1]


    if mask is not None:
        mask = (mask == False)
        mask_y = n.zeros(2*npts, bool)
        mask_y[1:-1:2], mask_y[2::2] = mask, mask
        mask_y = mask_y[1:-1]

    spec_x = spec_x[1:-1]
    spec_y = spec_y[1:-1]

    #pl.fill(spec_x[1:-1], spec_y[1:-1], fill=False, edgecolor=color, linestyle=style, closed=False, lw=lw)
    if not noplot:

        if zorder is None:
            if style in ['pts']:
                if mask is None:
                    pl.errorbar(wl, flux, var, fmt=None, color=0.5, capsize=0.0, elinewidth=1.0)
                    pl.plot(spec_x, spec_y, '.', color=color)
                else:
                    masked_y = n.ma.masked_array(spec_y, mask=mask_y)
                    pl.plot(spec_x, masked_y,'.', color=color)
            else:
                if mask is None:
                #pl.errorbar(wl, flux, var, fmt=None, color=0.5, capsize=0.0, elinewidth=1.0)
                    if var is not None:
                        pl.fill_between(spec_x, spec_y-var_y, spec_y+var_y, color='0.8')
                    pl.plot(spec_x, spec_y, color=color, linestyle=style, lw=lw)
                else:
                    masked_y = n.ma.masked_array(spec_y, mask=mask_y)
                    pl.plot(spec_x, masked_y, color=color, linestyle=style, lw=lw)
        else:
            if style in ['pts']:
                if mask is None:
                    if var is not None:
                        pl.errorbar(wl, flux, var, fmt=None, color=0.5, capsize=0.0, elinewidth=1.0, zorder=zorder-0.1)
                    pl.plot(spec_x, spec_y, '.', color=color, zorder=zorder)
                else:
                    masked_y = n.ma.masked_array(spec_y, mask=mask_y)
                    pl.plot(spec_x, masked_y,'.', color=color, zorder=zorder)
            else:
                if mask is None:
                #pl.errorbar(wl, flux, var, fmt=None, color=0.5, capsize=0.0, elinewidth=1.0)
                    if var is not None:
                        pl.fill_between(spec_x, spec_y-var_y, spec_y+var_y, color='0.8', zorder=zorder-0.1)
                    pl.plot(spec_x, spec_y, color=color, linestyle=style, lw=lw, zorder=zorder)
                else:
                    masked_y = n.ma.masked_array(spec_y, mask=mask_y)
                    pl.plot(spec_x, masked_y, color=color, linestyle=style, lw=lw, zorder=zorder)
    
    
    return spec_x, spec_y
    
