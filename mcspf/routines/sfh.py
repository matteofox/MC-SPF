import numpy as np

def expsfh(time, tau):
    
    _tau = np.copy(tau)
    _time = np.copy(time)
        
    #Time must be in steps of 1 Myr
    sfh = np.exp(-_time/_tau)
    intsfh = sfh.sum() * 1E6
    return sfh/intsfh

def delsfh(time, tau):
    
    _tau = np.copy(tau)
    _time = np.copy(time)
        
    #Time must be in steps of 1 Myr
    sfh = _time*np.exp(-_time/_tau)/_tau**2
    intsfh = sfh.sum() * 1E6
    return sfh/intsfh

def ssfr(time, tau, sfh, timeunit='Myr', avgtime=10):

    if timeunit == 'Gyr':
       tau *= 1000
       time*= 1000

    _tau = np.copy(tau)
    _time = np.copy(time)
        
    timearr = np.arange(int(_time))
    sfharr  = sfh(timearr, tau)
    
    return np.average(sfharr[-int(avgtime):])

def ssfrfromfile(time, sfh, timeunit='Myr', avgtime=10):

    if timeunit == 'Gyr':
       time*= 1000

    _time = np.copy(time)
        
    sfharr  = np.copy(sfh)
    
    return np.average(sfharr[int(_time-avgtime):int(_time)])
