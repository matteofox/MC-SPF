import numpy as np

def expsfh(dummy, time, tau):
    
    timearr = np.arange(int(time))
    
    _tau = np.copy(tau)
    _time = np.copy(time)
        
    #Time must be in steps of 1 Myr
    sfh = np.exp(-timearr/_tau)
    intsfh = sfh.sum() * 1E6
    return sfh/intsfh

def delsfh(dummy, time, tau):

    timearr = np.arange(int(time))

    _tau = np.copy(tau)
    _time = np.copy(time)

    #Time must be in steps of 1 Myr
    sfh = timearr*np.exp(-timearr/_tau)/_tau**2
    intsfh = sfh.sum() * 1E6
    return sfh/intsfh

def exptruncsfh(sfhcustom, agetrunc, tautrunc):
    
    sfr_at_qa = sfhcustom[-int(agetrunc)]
    
    if agetrunc>0 and tautrunc!=0:
       sfhcustom[-int(agetrunc):] = sfr_at_qa*np.exp(-1*np.arange(int(agetrunc))/tautrunc)
        
    #Time must be in steps of 1 Myr
    intsfh = sfhcustom.sum() * 1E6
    return sfhcustom/intsfh


def ssfr(age, tau, sfharray, sfhfunc, timeunit='Myr', avgtime=10):

    if timeunit == 'Gyr':
       tau *= 1000
       age*= 1000

    _tau = np.copy(tau)
    _age = np.copy(age)        
    
    sfharr  = sfhfunc(sfharray, age, tau)
        
    return np.average(sfharr[-int(avgtime):])

def ssfrfromfile(time, sfh, timeunit='Myr', avgtime=10):

    if timeunit == 'Gyr':
       time*= 1000

    _time = np.copy(time)
        
    sfharr  = np.copy(sfh)
    
    return np.average(sfharr[int(_time-avgtime):int(_time)])
