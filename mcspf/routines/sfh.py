import numpy as np

def expsfh(time, tau):
    
    _tau = np.copy(tau)
    _time = np.copy(time)
    
    #If in Gyr go to Myr
    if _tau<20:
       _tau *= 1000
    try:
       last = _time[-1]
       if last<14:
          _time *= 1000
    except:
       if _time<14:
          _time *= 1000   
    
    #Time must be in steps of 1 Myr
    #sfh = _time*np.exp(-_time/_tau)/_tau**2
    sfh = np.exp(-_time/_tau)
    intsfh = sfh.sum() * 1E6
    return sfh/intsfh

def ssfr(time, tau, sfh):

    _tau = np.copy(tau)
    _time = np.copy(time)

    if _tau<20:
       _tau *= 1000
    
    if _time<14:
       _time *= 1000      
    
    timearr = np.arange(int(_time))
    sfharr  = sfh(timearr, tau)
    
    return sfharr[-1]
