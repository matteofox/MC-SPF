#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools for working with the FSPS filter set.

This module uses filter information located in....
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["init_filters", "find_filter", "get_filter", "list_filters"]

import os
import numpy as np
from pkg_resources import resource_stream, resource_exists


# Cache for $SPS_HOME/data/filter_lambda_eff.dat parsed by numpy
LAMBDA_EFF_TABLE = None

# Cache for bandpass transmission listings: a dictionary keyed by bandpass
# name with values of wavelength, transmission tuples.
TRANS_CACHE = None

PATH = None

class Filter(object):

    def __init__(self, index, name, comment):
        self.index = index - 1
        self.name = name.lower()
        self.fullname = comment
        self.lambdaeff = 0.

    def __str__(self):
        return "<Filter({0})>".format(self.name)

    def __repr__(self):
        return "<Filter({0})>".format(self.name)

    @property
    def transmission(self):
        """Returns filter transmission: a tuple of wavelength (Angstroms) and
        an un-normalized transmission arrays.
        """
        try:
            return TRANS_CACHE[self.name]
        except KeyError as e:
            e.args += ("Could not find transmission data "
                       "for {0}".format(self.name))
            raise

    @property
    def lambda_eff(self):
        """Effective wavelength of Filter, in Angstroms."""
        if self.lambdaeff == 0 :
            self.lambdaeff = self._calc_lambda_eff()
        return float(self.lambdaeff)

    def _calc_lambda_eff(self):
        """Calculate effective wavelength of Filter, in Angstroms."""
        
        lam, trans = TRANS_CACHE[self.name]
        
        #This applies for photon counting device
        return np.sqrt(np.trapz(lam*trans, lam)/np.trapz(trans/lam, lam))
         
        

def init_filters(path):
    """
    Load the filter list, creating a dictionary of :class:`Filter` instances.
    """
    
    global TRANS_CACHE, PATH
    PATH = path
    
    # Load filter table from FSPS
    filter_list_path = path+'FILTER_LIST'
    filters = {}
    with open(filter_list_path) as f:
        for line in f:
            columns = line.strip().split()
            if len(columns) <= 1:
                continue
            else:
               fsps_id = columns[0]
               
               openbrackets = (np.array([x[0] for x in columns]) == '(')
               if openbrackets.sum() >0:
                 commentcol = np.argmax(np.array([x[0] for x in columns]) == '(')
                 key = '_'.join(columns[1:commentcol])
                 comment = ' '.join(columns[commentcol:])
               else:
                 key = '_'.join(columns[1:])
                 comment = ' '
               filters[key.lower()] = Filter(int(fsps_id), key, comment)
    
    """Parse the allfilters.dat file into the TRANS_CACHE."""
    path = os.path.expandvars(PATH+"allfilters.dat")
    names = list_filters(filters)
    TRANS_CACHE = {}
    filter_index = -1
    lambdas, trans = [], []
    with open(path) as f:
        for line in f:
            line.strip()
            if line[0].startswith("#"):
                # Close out filter
                if filter_index > -1:
                    TRANS_CACHE[names[filter_index]] = (
                        np.array(lambdas), np.array(trans))
                # Start new filter
                filter_index += 1
                lambdas, trans = [], []
            else:
                try:
                    l, t = line.split()
                    lambdas.append(float(l))
                    trans.append(float(t))
                except(ValueError):
                    pass

    
    return filters



def find_filter(filtersdb,band):
    """
    Find the FSPS name for a filter.

    Usage:

    ::
        find_filter("F555W")
        ['wfpc2_f555w', 'wfc_acs_f555w']

    :param band:
        Something like the name of the band.

    """
    b = band.lower()
    possible = []
    for k in filtersdb.keys():
        if b in k:
            possible.append(k)
    return possible


def get_filter(filtersdb, name):
    """Returns the :class:`fsps.filters.Filter` instance associated with the
    filter name.

    :param name:
        Name of the filter, as found with :func:`find_filter`.
    """
    
    try:
        return filtersdb[name.lower()]
    
    except KeyError as e:
        e.args += ("Filter {0} does not exist.".format(name),)
        raise


def list_filters(filtersdb):
    """Returns a list of all FSPS filter names.

    Filters are sorted by their FSPS index.
    """
    
    lst = [(name, f.index) for name, f in filtersdb.items()]
    lst.sort(key=lambda x: x[1])
    return [l[0] for l in lst]

