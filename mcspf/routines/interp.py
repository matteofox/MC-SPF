import numpy as np
from bisect import bisect_left

def tri_interp(data_cube, valuetpl, arraytpl):
    #locate vertices
    ilo = bisect_left(arraytpl[0], valuetpl[0])-1
    jlo = bisect_left(arraytpl[1], valuetpl[1])-1
    klo = bisect_left(arraytpl[2], valuetpl[2])-1
    
    di = (valuetpl[0] - arraytpl[0][ilo])/(arraytpl[0][ilo+1]-arraytpl[0][ilo])
    dj = (valuetpl[1] - arraytpl[1][jlo])/(arraytpl[1][jlo+1]-arraytpl[1][jlo])
    dk = (valuetpl[2] - arraytpl[2][klo])/(arraytpl[2][klo+1]-arraytpl[2][klo])

    interp_out = data_cube[ilo,jlo,klo,:]       * (1.-di)*(1.-dj)*(1.-dk) + \
                 data_cube[ilo,jlo,klo+1,:]     * (1.-di)*(1.-dj)*dk + \
                 data_cube[ilo,jlo+1,klo,:]     * (1.-di)*dj*(1.-dk) + \
                 data_cube[ilo,jlo+1,klo+1,:]   * (1.-di)*dj*dk + \
                 data_cube[ilo+1,jlo,klo,:]     * di*(1.-dj)*(1.-dk) + \
                 data_cube[ilo+1,jlo,klo+1,:]   * di*(1.-dj)*dk + \
                 data_cube[ilo+1,jlo+1,klo,:]   * di*dj*(1.-dk) + \
                 data_cube[ilo+1,jlo+1,klo+1,:] * di*dj*dk

    return interp_out

def bi_interp(data_cube, valuetpl, arraytpl):
    #locate vertices
    ilo = bisect_left(arraytpl[0], valuetpl[0])-1
    jlo = bisect_left(arraytpl[1], valuetpl[1])-1
       
    di = (valuetpl[0] - arraytpl[0][ilo])/(arraytpl[0][ilo+1]-arraytpl[0][ilo])
    dj = (valuetpl[1] - arraytpl[1][jlo])/(arraytpl[1][jlo+1]-arraytpl[1][jlo])

    interp_out = data_cube[ilo,jlo,:]     * (1.-di)*(1.-dj) + \
                 data_cube[ilo,jlo+1,:]   * (1.-di)*dj + \
                 data_cube[ilo+1,jlo,:]   * di*(1.-dj) + \
                 data_cube[ilo+1,jlo+1,:] * di*dj

    return interp_out

def interp(data_cube, valuetpl, arraytpl):
    #locate vertices
    ilo = bisect_left(arraytpl[0], valuetpl[0])-1
    
    di = (valuetpl[0] - arraytpl[0][ilo])/(arraytpl[0][ilo+1]-arraytpl[0][ilo])

    interp_out = data_cube[ilo,:]   * (1.-di) + \
                 data_cube[ilo+1,:] * di

    return interp_out
