#!/usr/bin/env python

import numpy as np
import os, warnings
import string, gc, time
from scipy.special import erf
from astropy.modeling import models,fitting
import astropy.io.fits as fits
from scipy.interpolate import RectBivariateSpline as rect
from bisect import bisect_left
import astropy.units as u
import sys
from contextlib import contextmanager

#Imports specific for the full version
from astropy.convolution import convolve_fft, convolve
import numpy.polynomial as poly

from ..utils.magtools  import getmag_spec
from ..utils.readfilt  import init_filters, get_filter
from ..utils.sincrebin import sincrebin_single_os as sincrebin

warnings.filterwarnings("ignore")

@contextmanager
def redir_stdout(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different



class sps_spec_fitter:

    def __init__(self, redshift, phot_mod_file, flux_obs, eflux_obs, filter_list, lim_obs, \
            cropspec=[100,20000], spec_in=None, res_in=None, polymax=20, fit_spec=True, fit_phot=True,
            priorAext=None, fesc=0., Gpriors=None, modeldir='./', filtdir='./', dl=None, cosmo=None,
            sfh_pars=['TAU','AGE']):
        
        """ Class for dealing with MultiNest fitting """
        
        #Define if photometry and or spectroscopy should be fit
        if spec_in is not None:
          self.fit_spec = fit_spec
        else:
          self.fit_spec = False  
        
        self.fit_phot = fit_phot
               
        if cosmo is None:
           from astropy.cosmology import FlatLambdaCDM
           cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
        
       #input information of the galaxy to be fitted
        self.redshift = redshift
        self.Gpriors = Gpriors
        self.modeldir = modeldir
        self.filtdir = filtdir
        self.spec_in = spec_in
        self.res_in = res_in
        self.priorAext = priorAext 
        
        #constants
        self.small_num = 1e-70
        self.lsun = 3.839e33 #erg/s
        self.pc2cm = 3.08568e18 #pc to cm
        self.clight  = 2.997925e18 #A/s
        self.h = 6.6262E-27 # Planck constant in erg s
        self.Mpc_to_cm = u.Mpc.to(u.cm)
        self.angstrom_to_km = u.angstrom.to(u.km)
        
        #Derive luminosity distance in Mpc
        if dl is None:
           dl = cosmo.luminosity_distance(redshift).value
        
        self.mag2cgs = np.log10(self.lsun/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)
        self.dist = 4.0*np.pi*(self.pc2cm*self.pc2cm)*100.
        self.lum_corr = (self.Mpc_to_cm*dl)**2*4.*np.pi #4pi*dl^2 in cm^2
        
        #some derived parameters
        self.gal_age = cosmo.age(self.redshift).value #In Gyr here, will be modified later if necessary
        self.dm = 5.*np.log10(dl*1e5) #DM brings source at 10pc
        self.fscale = 10**(-0.4*self.dm)

        ## read in the pre-computed SPS grid
        
        mfile = fits.open(phot_mod_file)
        num_ext = len(mfile)
        
        #pull wavelength information from the primary extension of the models
        wdata = np.array(mfile[0].data, dtype=np.float)
        twl = self.airtovac(wdata)
        
        #get the wavelength information for the DH02 templates to stitch the two together
        dh_wl = np.loadtxt(modeldir+'spectra_DH02.dat', usecols=(0,))*1e4
        dh_nwl = len(dh_wl)

        #expand wavelength grid to include range covered by DH templates
        dh_long = (dh_wl > twl.max())
        self.wl = np.r_[twl, dh_wl[dh_long]]
        self.n_wl = len(self.wl)
        
        #Done with wavelength, now pass through extensions to get grid parameters
        ext_tau, ext_age, ext_metal = [],[],[]
        for ii in range(1,num_ext):
            if ii==1:
             try:
                self.timeunit = mfile[ii].header['AGEUNIT']
             except:
                self.timeunit = 'Myr'
                print('AGEUNIT keyword not found. Using Myr')
            tau   = mfile[ii].header[sfh_pars[0]] 
            age   = mfile[ii].header[sfh_pars[1]] 
            ext_tau.append(np.float(tau))
            ext_age.append(np.float(age))

        #If model units are in Myr modify cosmo age accordingly
        if self.timeunit == 'Myr':
           self.gal_age *= 1000
        
        self.grid_tau = np.unique(ext_tau)
        self.grid_age = np.unique(ext_age)
        
        self.n_tau = len(self.grid_tau)
        self.n_age = len(self.grid_age)
        
        #tau_id = dict(zip(self.grid_tau, range(self.n_tau)))
        #age_id = dict(zip(self.grid_age, range(self.n_age)))
        
        #output grid
        self.mod_grid = np.zeros((self.n_tau, self.n_age, self.n_wl), dtype=np.float)
        
        #grid where the fractional flux from young populations is stored
        self.fym_grid = np.zeros_like(self.mod_grid)
        
        for ii in range(1,num_ext):            
            mdata  = np.array(mfile[ii].data,  dtype=np.float)
            mmass  = mfile[ii].header['MSTAR']
            mmetal = mfile[ii].header['METAL']
            mtau   = mfile[ii].header[sfh_pars[0]] 
            mage   = mfile[ii].header[sfh_pars[1]] 
            
            tau_idx = np.where(mtau == self.grid_tau)[0]
            age_idx = np.where(mage == self.grid_age)[0]
            
            self.mod_grid[tau_idx, age_idx, :] = np.interp(self.wl, twl, mdata[:, 0]/mmass, left=0, right=0)
            self.fym_grid[tau_idx, age_idx, :] = np.interp(self.wl, twl, mdata[:, 1], left=0, right=0)

        mfile.close() 

        #some information on the BC03 spectral resolution (in the optical)
        self.sps_res_val = np.copy(self.wl)/300.
        self.sps_res_val[(self.wl >= 3200) & (self.wl <= 9500)] = 3.
        
        self.mod_grid[~np.isfinite(self.mod_grid)] = 0.
        self.fym_grid[~np.isfinite(self.fym_grid)] = 0.

        #pre-compute attenuation curve
        self.k_cal = self._make_dusty(self.wl)
        
        #redshift the grid to be used in deriving predicted fluxes, also apply
        #flux correction to conserve energy
        self._redshift_spec()
        
        ### for nebular emission ###
        self.wl_lyman = 912.
        self.ilyman = np.searchsorted(self.wl, self.wl_lyman, side='left') #wavelength just above Lyman limit
        self.lycont_wls = np.r_[self.wl[:self.ilyman], np.array([self.wl_lyman])]
        self.clyman_young = [None, None] #A list of two elements, first is for phot, the other for spec
        self.clyman_old   = [None, None] #A list of two elements, first is for phot, the other for spec
        self.fesc = fesc                 #lyman continuum escape fraction

        self.emm_scales = np.zeros((7,10,128), dtype=np.float)
        self.emm_wls    = np.zeros(128,        dtype=np.float)
        self.emm_ages   = np.zeros(10,         dtype=np.float)
        self.emm_ions   = np.zeros(7,         dtype=np.float)
        icnt = 0
        rline = 0
        iline = 0
        
        with open(modeldir+'nebular_Byler.lines','r') as file:
            for line in file:
                if line[0] != '#':
                    temp = (line.strip()).split(None)
                    if not iline: #Read wave line
                        self.emm_wls[:] = np.array(temp, dtype=np.float)
                        iline = 1
                    else:
                        if rline: #Read line fluxes
                            self.emm_scales[icnt%7,icnt//7,:] = np.array(temp, dtype=np.float)
                            icnt += 1
                        if len(temp) == 3 and float(temp[0]) == 0.0:
                            rline = 1
                            self.emm_ages[icnt//7] = float(temp[1])/1e6
                            self.emm_ions[icnt%7]  = float(temp[2])
                        else:
                            rline = 0
        
        thb = (self.emm_wls > 4860) & (self.emm_wls < 4864)
        tha = (self.emm_wls > 6560) & (self.emm_wls < 6565)
        self.emm_scales = np.copy(self.emm_scales) / self.emm_scales[:,:,thb]
        
        mscale     = np.max(self.emm_scales, axis=(0,1))
        keep_scale = (mscale > 0.025) & (self.emm_wls<1E5)
                
        self.emm_scales = self.emm_scales[:,:,keep_scale]
        self.emm_wls    = self.emm_wls[keep_scale]
        
        #generate pattern arrays for nebular emission lines
        dpix = np.diff(self.wl)
        self.wl_edges  = np.r_[np.array([self.wl[0]-dpix[0]/2.]), np.r_[self.wl[1:]-dpix/2., np.array([self.wl[-1]+dpix[-1]/2.])]]
        self.res_lines = np.interp(self.emm_wls, self.wl, self.sps_res_val)/2.355
      
        self.emm_lines_all = np.zeros((len(self.emm_ions), len(self.emm_ages), len(self.wl)), dtype=np.float)
        for jj in range(len(self.emm_ions)):
          for ii in range(len(self.emm_ages)):
            this_scale = self.emm_scales[jj,ii,:]
            self.emm_lines_all[jj,ii,:] = np.sum(this_scale[:,None]*\
                    np.diff(0.5*(1.+erf((self.wl_edges[None,:]-self.emm_wls[:,None])/\
                    np.sqrt(2.*self.res_lines**2)[:,None])), axis=1)/np.diff(self.wl_edges), axis=0)
                
        #### LOAD DUST EMISSION TABLES ####
        #first fetch alpha values
        self.dh_alpha = np.loadtxt(modeldir+'alpha_DH02.dat', usecols=(0,))
        self.dh_nalpha = len(self.dh_alpha)

        self.dh_dustemm = np.zeros((self.dh_nalpha, self.n_wl), dtype=np.float)
        for ii in range(self.dh_nalpha):
            tdust = 10**np.loadtxt(modeldir+'spectra_DH02.dat', usecols=(ii+1,))
            self.dh_dustemm[ii,:] = np.interp(self.wl, dh_wl, tdust)/self.wl

        #normalize to Lbol = 1
        norm = np.trapz(self.dh_dustemm, self.wl)
        self.dh_dustemm /= norm[:,None]
        
        #### LOAD DUST EMISSION TABLES ####        
                
        self.filters = filter_list #should already correspond to FSPS names
        self.n_bands = len(self.filters)
        self.bands, self.pivot_wl = self._get_filters()
        
        #photometric measurements (convert from fnu to flambda)
        #Input at this stage is in erg/cm^2/s/Hz output in erg/cm2/s/A
        self.flux_obs = flux_obs * self.clight/self.pivot_wl**2
        self.eflux_obs = eflux_obs * self.clight/self.pivot_wl**2
        self.lim_obs = lim_obs

        #if an input spectrum is provided, do additional processes
        if spec_in is not None:
            
            #so it can be used in the fit           
            file = fits.open(self.spec_in)
            obj       = np.asarray(file[0].data, dtype=float)
            obj_noise = np.asarray(file[1].data, dtype=float)
            wl_obj    = np.asarray(file[2].data, dtype=float)
                    
            #open up resolution file
            rfile = fits.open(self.res_in)
            rhdr = rfile[0].header
            
            self.lsf_res = np.asarray(rfile[0].data, dtype=np.float) * 2.
            crval, cdelt, crpix, size = rhdr['CRVAL1'], rhdr['CD1_1'], 1., rhdr['NAXIS1']
            self.lsf_wl = ((np.arange(size, dtype=np.float)+1 - crpix)*cdelt + crval)
            rfile.close()

            #clip off the spectrum as requested by the user
            range_crop = (wl_obj > cropspec[0]) & (wl_obj < cropspec[1])
            rest_wl = wl_obj/(1.+self.redshift) 
            
            self.wl_obj = rest_wl[range_crop]
            self.obj = obj[range_crop]
            self.obj_noise = obj_noise[range_crop]
            
            self.dlam_lin = np.diff(self.wl_obj)[0]
            self.npix_obj = len(self.wl_obj)

            
            #rebin the object to log
            self.log_wl, self.dlam_log = np.linspace(np.log10(self.wl_obj[0]), np.log10(self.wl_obj[-1]), self.npix_obj, retstep=True)
            self.log_wl_edges = np.r_[self.log_wl-self.dlam_log/2., self.log_wl[-1:]+self.dlam_log/2.]

            mask = np.isfinite(self.obj) & np.isfinite(self.obj_noise) & (self.obj_noise > 0)
            self.log_obj = sincrebin(10**self.log_wl, self.wl_obj[mask], self.obj[mask])
            self.log_noise = np.sqrt(sincrebin(10**self.log_wl, self.wl_obj[mask], self.obj_noise[mask]**2))
           
            #masking to remove bad pixels etc.
            self.goodpix_gen = np.isfinite(self.log_obj) & (self.log_noise > 0) & np.isfinite(self.log_noise)
            self.n_good = np.count_nonzero(self.goodpix_gen)

            #include masking of 5577 sky line
            sky_mask = (self.log_wl+np.log10(1.+self.redshift) > np.log10(5567.)) & (self.log_wl+np.log10(1.+self.redshift) < np.log10(5587.))
            self.goodpix_gen[sky_mask] = False
            
            #AO mask if in A0 mode
            if (obj[(wl_obj>5800) & (wl_obj<5900)]).sum() ==0:
               ao_lims = wl_obj[obj==0]
               ao_mask = (self.log_wl+np.log10(1.+self.redshift) > np.log10(ao_lims[0])) & (self.log_wl+np.log10(1.+self.redshift) < np.log10(ao_lims[-1]))
               self.goodpix_gen[ao_mask] = False
               
            self.goodpix_spec = np.copy(self.goodpix_gen)
            
            #pl.plot(self.log_wl, self.log_obj, 'k-')
            #pl.plot(self.log_wl[~self.goodpix_spec], self.log_obj[~self.goodpix_spec], 'ro', mec='red', ms=3.0)
            #pl.show()
            
            #normalize the object and variance arrays
            self.norm_window = ((self.log_wl+np.log10(1.+self.redshift) > np.log10(6000)) & (self.log_wl+np.log10(1.+self.redshift) < np.log10(6200)))
            
            #normalize the observed spectrum
            self.spec_norm = np.nansum(self.log_obj[self.norm_window]/self.log_noise[self.norm_window]**2)/\
                    np.nansum(1./self.log_noise[self.norm_window]**2)
            
            self.log_obj = self.log_obj/self.spec_norm
            self.log_noise = self.log_noise/self.spec_norm

            #Now work through the models
            self.use_wl = (self.wl > 900) & (self.wl < cropspec[1]+100)
            self.model_spec_wl = self.wl[self.use_wl]

            #rescale the model spectra and crop down wavelength
            self.model_spec_grid = self.red_mod_grid[:,:,self.use_wl]
            self.model_fysp_grid = self.fym_grid[:,:,self.use_wl]

            self.n_wl_crop = np.count_nonzero(self.use_wl)
            self.n_model_spec = self.n_tau * self.n_age

            self.log_model_wl = np.arange(np.log10(self.model_spec_wl[0]), \
                    np.log10(self.model_spec_wl[-1])+self.dlam_log/2., self.dlam_log)
            self.log_model_wl_edges = np.r_[self.log_model_wl-self.dlam_log/2., self.log_model_wl[-1:]+self.dlam_log/2.]
            self.n_model_wl = len(self.log_model_wl)
            self.spec_k_cal = self._make_dusty(10**self.log_model_wl)

            self.npad = 2**int(np.ceil(np.log2(self.n_model_wl)))

            #mask pixels outside the template range
            self.goodpix_spec[self.log_wl < min(self.log_model_wl)] = 0
            self.goodpix_spec[self.log_wl > max(self.log_model_wl)] = 0

            #spectral resolutions
            sps_res = np.interp(10**self.log_model_wl, self.wl, self.sps_res_val)
            obs_res = 10**self.log_model_wl / np.interp(10**self.log_model_wl, self.lsf_wl, self.lsf_res)

            model_dlam = np.nanmedian(np.diff(self.model_spec_wl))

            a2kms = self.clight/1e13/10**self.log_model_wl
            self.kms2pix = np.log(10)*self.clight*self.dlam_log/1e13

            sps_kms = sps_res*a2kms/2.355
            self.obs_kms = obs_res*a2kms/2.355
            res_diff = np.sqrt(self.obs_kms**2 - sps_kms**2) #observations lower resolution

            self.kernel_kms = np.copy(res_diff)
            kernel_pix = self.kernel_kms/self.kms2pix
            
            #set up storage for log models
            self.log_spec_grid = np.zeros((self.n_tau, self.n_age, self.n_model_wl), dtype=np.float)
            self.log_fysp_grid = np.zeros((self.n_tau, self.n_age, self.n_model_wl), dtype=np.float)
            
            cnt = 0
            for jj in range(self.n_tau):
                for kk in range(self.n_age):
                     this_templ = self.model_spec_grid[jj,kk,:]
                     this_young = self.model_fysp_grid[jj,kk,:] * np.copy(this_templ)
                     log_spec = np.interp(10**self.log_model_wl,  self.model_spec_wl, this_templ)
                     log_young = np.interp(10**self.log_model_wl, self.model_spec_wl, this_young)

                     self.log_spec_grid[jj,kk,:] = log_spec
                     self.log_fysp_grid[jj,kk,:] = np.copy(log_young)/np.copy(log_spec)
        

            #cut down emission line arrays for the spectral grid
            log_emm_use = (self.emm_wls > 10**self.log_model_wl[0]) & (self.emm_wls < 10**self.log_model_wl[-1])
            self.log_emm_wls = self.emm_wls[log_emm_use]
            self.log_emm_res = np.interp(np.log10(self.log_emm_wls), self.log_model_wl, self.obs_kms/self.kms2pix/2.) #in pixels
            self.log_emm_scales = self.emm_scales[:,:,log_emm_use]
            
            self.diff_pix = (self.log_model_wl_edges[None,:] - np.log10(self.log_emm_wls)[:,None])/self.dlam_log #in pixels
            self.norm_pix = np.sqrt(2.)*self.log_emm_res[:,None]
      
            #a few additional steps to set up the fitting
            self.vsys_pix = (self.log_model_wl[0] - self.log_wl[0])*\
                   np.log(10)*self.clight/self.kms2pix/1e13

            #setup for polynomial normalization
            low, high = min(self.log_wl), max(self.log_wl)
            dlam = high-low
            mlam = (high+low)/2.
            self.scaled_lambda = (self.log_wl - mlam) * 2/dlam
            self.dlam_poly = 150.
            self.poly_max = polymax
            self.poly_deg = min(int((10**high - 10**low)/self.dlam_poly), self.poly_max)

        
        #set up parameter limits
        self.tau_lims = np.array((self.grid_tau.min(), self.grid_tau.max()))
        if sfh_pars[1]=='AGE':
          self.age_lims = np.array((self.grid_age.min(), self.gal_age)) #self.grid_age.max()))
        else:
          self.age_lims = np.array((self.grid_age.min(), self.grid_age.max()))
        self.av_lims = np.array((0., 4.))
        self.sig_lims = np.array((10., 500.))
        self.vel_lims = np.array((-250., 250.))
        self.lnf_lims = np.array((-0.5, 1.5))
        self.ext_lims = np.array((0, 4.))
        self.alpha_lims = np.array((self.dh_alpha[0], self.dh_alpha[-1]))
        self.mass_lims = np.array((3,12))
        self.sigg_lims = np.array((1.,200.)) 
        self.emmage_lims = np.array((self.emm_ages.min(), 10))
        self.emmion_lims = np.array((self.emm_ions.min(), self.emm_ions.max()))
        self.lyscale_lims = np.array((0.99, 1.01))

        self.bounds = [self.tau_lims, self.age_lims, self.av_lims, \
                self.sig_lims, self.vel_lims, self.lnf_lims, self.ext_lims, \
                self.alpha_lims, self.mass_lims, self.sigg_lims, self.emmage_lims,\
                self.emmion_lims, self.lyscale_lims]
                
        self.ndims = len(self.bounds)   

    def vactoair(self, linLam):
        """Convert vacuum wavelengths to air wavelengths using the conversion
        given by Morton (1991, ApJS, 77, 119).

        """
        wave2 = np.asarray(linLam, dtype=float)**2
        fact = 1. + 2.735182e-4 + 131.4182/wave2 + 2.76249e8/(wave2*wave2)
        return linLam/fact


    def airtovac(self, linLam):
        """Convert air wavelengths to vacuum wavelengths using the conversion
        given by Morton (1991, ApJS, 77, 119).

        """
        sigma2 = np.asarray(1E4/linLam, dtype=float)**2
        
        fact = 1. + 6.4328e-5 + 2.949281e-2/(146.-sigma2) + 2.5540e-4/(41.-sigma2)
        fact[linLam < 2000] = 1.0
        
        return linLam*fact
    
    def _scale_cube(self, cube, ndims, nparams):
        for ii in range(ndims):
            cube[ii] = cube[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])

        return

    def _losvd(self, vel, sigma):
        """generate broadening kernel given values in km/s
        """
        vel_pix = vel/self.kms2pix
        sigma_pix = sigma/self.kms2pix
        dw = int(np.ceil(np.fabs(self.vsys_pix) + np.fabs(vel_pix) + 6.*sigma_pix)) #sample to 6 sigma
        npix = 2*dw + 1 # total number of pixels to fully sample the losvd
        pix = np.linspace(-dw-0.5, dw+0.5, npix+1, endpoint=True) # pixel edges

        y = (pix - vel_pix - self.vsys_pix)/sigma_pix/np.sqrt(2)
        losvd = np.diff(0.5*(1.+erf(y)))
        return losvd, npix

    def _losvd_rfft(self, vel, sigma):
        """Generate analytic fourier transform of the LOSVD following Cappellari et al. 2016 and pPXF,
        simplified for purposes here"""
        nl = self.npad//2 + 1
        losvd_rfft = np.zeros(nl, dtype=np.complex)
        vel_pix = vel/self.kms2pix + self.vsys_pix
        sigma_pix = sigma/self.kms2pix

        #compute the FFT
        a = vel_pix / sigma_pix
        w = np.linspace(0, np.pi*sigma_pix, nl)
        losvd_rfft[:] = np.exp(1j*a*w - 0.5*w**2)

        return np.conj(losvd_rfft)

    def _vel_convolve_fft(self, spec, sigma, vel):
        
        #convolve input spectrum to given velocity
        #generate kernel (in pixels)
        fft_losvd = self._losvd_rfft(vel, sigma)

        #pre-compute fft of the continuum template library
        fft_template = np.fft.rfft(spec, n=self.npad, axis=0)

        tmpl = np.fft.irfft(fft_template*fft_losvd)

        return tmpl[:self.n_model_wl]

    def _vel_convolve(self, spec, sigma, vel):
        
        #convolve input spectrum to given velocity
        #generate kernel (in pixels)
        losvd, nl = self._losvd(vel, sigma)

        convd = convolve_fft(spec, losvd)
        return convd

    def _make_dusty(self, wl):
        
        #compute attenuation assuming Calzetti+ 2000 law
        #single component 
        n_wl = len(wl)
        R = 4.05
        div = wl.searchsorted(6300., side='left')
        k_cal = np.zeros(n_wl, dtype=float)
        
        k_cal[div:] = 2.659*( -1.857 + 1.04*(1e4/wl[div:])) + R
        k_cal[:div] = 2.659*(-2.156 + 1.509*(1e4/wl[:div]) - 0.198*(1e4/wl[:div])**2 + 0.011*(1e4/wl[:div])**3) + R
        
        zero = bisect_left(-k_cal, 0.)
        k_cal[zero:] = 0.

        #2175A bump
        #eb = 1.0
        k_bump = np.zeros(n_wl, dtype=float)
        #k_bump[:] = eb*(wl*350)**2 / ((wl**2 - 2175.**2)**2 + (wl*350)**2)
        
        #k_tot is the total selective attenuation A(lam)/E(B-V).
        #For calzetti R(V) = A(V)/E(B-V)
        k_tot = k_cal + k_bump 
                
        #Return 0.4*A(lam)/A(V)
        return 0.4*(k_cal+ k_bump)/R

    def _redshift_spec(self):
        self.red_wl       = self.wl * (1.+self.redshift)
        if self.redshift>0:
           self.red_mod_grid = self.mod_grid / (1+self.redshift)  
        else:
           self.red_mod_grid = np.copy(self.mod_grid)
 
    def _tri_interp(self, data_cube, value1, value2, value3, array1, array2, array3):
        #locate vertices
        ilo = bisect_left(array1, value1)-1
        jlo = bisect_left(array2, value2)-1
        klo = bisect_left(array3, value3)-1

        di = (value1 - array1[ilo])/(array1[ilo+1]-array1[ilo])
        dj = (value2 - array2[jlo])/(array2[jlo+1]-array2[jlo])
        dk = (value3 - array3[klo])/(array3[klo+1]-array3[klo])

        interp_out = data_cube[ilo,jlo,klo,:]       * (1.-di)*(1.-dj)*(1.-dk) + \
                     data_cube[ilo,jlo,klo+1,:]     * (1.-di)*(1.-dj)*dk + \
                     data_cube[ilo,jlo+1,klo,:]     * (1.-di)*dj*(1.-dk) + \
                     data_cube[ilo,jlo+1,klo+1,:]   * (1.-di)*dj*dk + \
                     data_cube[ilo+1,jlo,klo,:]     * di*(1.-dj)*(1.-dk) + \
                     data_cube[ilo+1,jlo,klo+1,:]   * di*(1.-dj)*dk + \
                     data_cube[ilo+1,jlo+1,klo,:]   * di*dj*(1.-dk) + \
                     data_cube[ilo+1,jlo+1,klo+1,:] * di*dj*dk

        return interp_out

    def _bi_interp(self, data_cube, value1, value2, array1, array2):
        #locate vertices
        ilo = bisect_left(array1, value1)-1
        jlo = bisect_left(array2, value2)-1

        di = (value1 - array1[ilo])/(array1[ilo+1]-array1[ilo])
        dj = (value2 - array2[jlo])/(array2[jlo+1]-array2[jlo])

        interp_out = data_cube[ilo,jlo,:]     * (1.-di)*(1.-dj) + \
                     data_cube[ilo,jlo+1,:]   * (1.-di)*dj + \
                     data_cube[ilo+1,jlo,:]   * di*(1.-dj) + \
                     data_cube[ilo+1,jlo+1,:] * di*dj

        return interp_out

    def _interp(self, data_cube, value, array):
        #locate vertices
        ilo = bisect_left(array, value)-1
        di = (value - array[ilo])/(array[ilo+1]-array[ilo])

        interp_out = data_cube[ilo,:]   * (1.-di) + \
                     data_cube[ilo+1,:] * di

        return interp_out


    def _get_filters(self):
        #fetch filter transmission curves from FSPS
        #normalize and interpolate onto standard grid
        bands = np.zeros((self.n_bands, self.n_wl), dtype=float)
        pivot = np.zeros(self.n_bands, dtype=float)
        
        #lookup for filter number given name
        filters_db = init_filters(self.filtdir)
        
        for ii, filt in enumerate(self.filters):
            
            if 'line' in filt:
             return 0,0
           
            fobj = get_filter(filters_db, filt)
            fwl, ftrans = fobj.transmission
            ftrans = np.maximum(ftrans, 0.)
            trans_interp = np.asarray(np.interp(self.red_wl, fwl, \
                    ftrans, left=0., right=0.), dtype=np.float) 

            #normalize transmission
            ttrans = np.trapz(np.copy(trans_interp)*self.red_wl, self.red_wl) #for integrating f_lambda
            if ttrans < self.small_num: ttrans = 1.
            ntrans = np.maximum(trans_interp / ttrans, 0.0)
            
            if 'mips' in filt:
                td = np.trapz(((fobj.lambda_eff/self.red_wl)**(2.))*ntrans*self.red_wl, self.red_wl)
                ntrans = ntrans/max(1e-70,td)

            if 'irac' in filt or 'pacs' in filt or 'spire' in filt or 'iras' in filt: 
                td = np.trapz(((fobj.lambda_eff/self.red_wl)**(1.))*ntrans*self.red_wl, self.red_wl)
                ntrans = ntrans/max(1e-70,td)

            bands[ii,:] = ntrans
            pivot[ii] = fobj.lambda_eff
        
        return bands, pivot

    def _get_mag_single(self, spec, ret_flux=True):
        
        #compute observed frame magnitudes and fluxes, return both
        
        tflux = np.zeros(self.n_bands, dtype=np.float)

        getmag_spec(self.red_wl, np.einsum('ji,i->ij', self.bands, \
                spec*self.red_wl), self.n_bands, tflux)
        
        if not ret_flux:
            tmag = -2.5*np.log10(tflux*self.fscale) - 48.6
            if np.all(tflux) > 0:
                return tmag
            else:
                tmag[flux <= 0] = -99.
                return tmag
        
        #Return fluxes in erg/s/cm^2/A
        if np.all(tflux) > 0:        
            return tflux*self.fscale
        else:
            flux = tflux*self.fscale
            flux[tflux <= 0] = 0.
            return flux 

    def lnprior(self, p, ndim):
        if all(b[0] <= v <= b[1] for v, b in zip(p, self.bounds)):
            
            pav = 0
            
            if self.Gpriors is not None:
              for par in range(ndim):
                if self.Gpriors[2*par] != 'none' and self.Gpriors[(2*par)+1] != 'none':
                  val = float(self.Gpriors[2*par])
                  sig = float(self.Gpriors[(2*par)+1])
                  pav  +=  -0.5*(((p[par]-val)/sig)**2 + np.log(2.*np.pi*sig**2))
            
            if self.priorAext is not None:
            
               aval = self.priorAext[0]*p[2] #[1.17,0.01]
               sav  = self.priorAext[1]
               pav  +=  -0.5*(((p[6]-aval)/sav)**2 + np.log(2.*np.pi*sav**2))
                             
            return pav

        return -np.inf

    def lnlhood(self, p, ndim, nparams):
                
        if self.fit_spec == True:
          
          model_spec = self.reconstruct_spec(p, ndim)
        
          if np.all(model_spec == 0.):
              return -np.inf

          ispec2 = 1./((self.log_noise*np.exp(p[5]))**2)

          spec_lhood = -0.5*np.nansum((ispec2*(self.log_obj-model_spec)**2 - np.log(ispec2) + np.log(2.*np.pi))[self.goodpix_spec])
        
        else:
          
          spec_lhood = 0
          
           
        model_phot, _ = self.reconstruct_phot(p, ndim)
         
        if np.all(model_phot == 0.):
           return -np.inf

        iphot2 = 1./(self.eflux_obs**2)
         
        if np.sum(self.lim_obs):
             terf = 0.5*(1.+erf((self.eflux_obs-model_phot)/np.sqrt(2.)/self.eflux_obs))[self.lim_obs == 1]
             if np.any(terf == 0):
                 return -np.inf
             else:
                 phot_lhood = np.nansum(-0.5*((iphot2*(self.flux_obs-model_phot)**2)[self.lim_obs == 0] - \
                         np.log(iphot2[self.lim_obs == 0]) + np.log(2.*np.pi))) + \
                         np.nansum(np.log(terf))
        else:
             phot_lhood = np.nansum(-0.5*((iphot2*(self.flux_obs-model_phot)**2) - \
                     np.log(iphot2) + np.log(2.*np.pi)))
        
        #### APPLY THE PRIOR HERE  #####
        pr = self.lnprior(p, ndim)
        
        if not np.isfinite(pr):
            return -np.inf
        
        return spec_lhood + phot_lhood + pr

        
    def _get_clyman(self, spec): #compute number of Lyman continuum photons
        lycont_spec = np.interp(self.lycont_wls, self.wl, spec) #spectrum in erg/s/A
        nlyman = np.trapz(lycont_spec*self.lycont_wls, self.lycont_wls)/self.h/self.clight
 
        #modify input spectrum to remove photons 
        spec[:self.ilyman] *= self.fesc
    
        return nlyman*(1.-self.fesc), spec
      
    def _get_nebular(self, emm_spec, lyscale, index): 
        
        emm_young = self.clyman_young[index] * 4.796e-13 * emm_spec * lyscale #conversion is from QH0 to Hbeta luminosity
        emm_old   = self.clyman_old[index]   * 4.796e-13 * emm_spec * lyscale
                
        return emm_young, emm_old

    def _make_spec_emm(self, vel, sigma, emm_age, emm_ion):
        vel_pix = vel/self.kms2pix
        sigma_pix = sigma/self.kms2pix

        temp_emm_scales = self._bi_interp(self.log_emm_scales, emm_ion, emm_age, self.emm_ions, self.emm_ages) 
                
        emm_grid = np.sum(temp_emm_scales[:,None]*\
                np.diff(0.5*(1.+erf((self.diff_pix-vel_pix-self.vsys_pix)/np.sqrt(2.)/sigma_pix)), axis=1)/\
                np.diff(10**self.log_model_wl_edges)[None,:], axis=0)

        return emm_grid

    def reconstruct_spec(self, p, ndim):
        
        #Parameters
        itau, iage, iav, isig, ivel, ilnf, iav_ext, _, _, isig_gas, iage_gas, iion_gas, ilyscale = [p[x] for x in range(ndim)]
        
        #interpolate the full spectroscopic grid for lyman continuum calculation
        spec_model = self._bi_interp(self.red_mod_grid, itau, iage, self.grid_tau, self.grid_age)
        frac_model = self._bi_interp(self.fym_grid, itau, iage, self.grid_tau, self.grid_age)
        
        #get number of lyman continuum photons
        self.clyman_young[1], temp_young = self._get_clyman(spec_model*frac_model)
        self.clyman_old[1],   temp_old   = self._get_clyman(spec_model*(1.-frac_model))
       
        #interpolate the model grid to given values
        hr_spec_model = self._bi_interp(self.log_spec_grid, itau, iage, self.grid_tau, self.grid_age)
        hr_frac_model = self._bi_interp(self.log_fysp_grid, itau, iage, self.grid_tau, self.grid_age)

        #build the emission line arrays and attenuate given Av and A_extra
        emm_lines_spec = self._make_spec_emm(ivel, isig_gas, iage_gas, iion_gas)
        emm_young, emm_old = self._get_nebular(emm_lines_spec, ilyscale, 1)
        
        #Fobs = Fint*10^(-0.4*Alam/Av*Av)
        dusty_emm = ((10**(-iav*self.spec_k_cal) * emm_old) + (10**(-(iav+iav_ext)*self.spec_k_cal) * emm_young))[:self.npix_obj]
        
        #attenuate the continuum model and convolve to given dispersion
        spec_young = self._vel_convolve_fft(hr_spec_model*hr_frac_model, isig, ivel)
        spec_old = self._vel_convolve_fft(hr_spec_model*(1.-hr_frac_model), isig, ivel)
                
        dusty_spec = (10**(-iav*self.spec_k_cal) * \
                (spec_old + spec_young*10**(-iav_ext*self.spec_k_cal)))[:self.npix_obj]
       
        #rescaled variance
        scale_sig = self.log_noise*np.exp(ilnf)
        
        #remove shape differences between spectrum and model
        try:
            cont_poly = self._poly_norm(self.log_obj/(dusty_spec+dusty_emm), scale_sig**2/(dusty_spec+dusty_emm)**2, self.poly_deg)
        except:
            cont_poly = 0.

        self.cont_spec = np.copy(dusty_spec)*cont_poly
        
        #Remove continuum from observed spectrum
        self.emiobj = self.log_obj-self.cont_spec
        
        #return the re-normalized model + emission lines
        return (dusty_spec + dusty_emm)*cont_poly


    def reconstruct_phot(self, p, ndim):
        #parameters
        itau, iage, iav, _, _, _, iav_ext, ialpha, ilmass, _, iage_gas, iion_gas, ilyscale = [p[x] for x in range(ndim)]
        
        #interpolate the full photometric grid
        spec_model = self._bi_interp(self.red_mod_grid, itau, iage, self.grid_tau, self.grid_age)
        frac_model = self._bi_interp(self.fym_grid,     itau, iage, self.grid_tau, self.grid_age)

        #get number of lyman continuum photons
        self.clyman_young[0], temp_young = self._get_clyman(spec_model*frac_model)
        self.clyman_old[0], temp_old     = self._get_clyman(spec_model*(1.-frac_model))

        #### Include nebular emission ####
        iemm_lines = self._bi_interp(self.emm_lines_all, iion_gas, iage_gas, self.emm_ions, self.emm_ages) 

        emm_young, emm_old = self._get_nebular(iemm_lines, 1, 0)

        #attenuate photometry spectrum, then compute fluxes given input bands
        self.dusty_phot_young = (10**(-(2.27*iav)*self.k_cal) * (temp_young + emm_young))
        self.dusty_phot_old   = (10**(-iav*self.k_cal) * (temp_old+emm_old))
        self.dusty_phot       = self.dusty_phot_young + self.dusty_phot_old
        
        #### THERMAL DUST EMISSION ####
        lbol_init = np.trapz(temp_young+temp_old+emm_young+emm_old, self.wl)
        lbol_att = np.trapz(self.dusty_phot, self.wl)

        dust_emm = (lbol_init - lbol_att)
        tdust_phot = self._interp(self.dh_dustemm, ialpha, self.dh_alpha)

        #remove stellar component
        mask_pixels = (self.wl >= 2.5e4) & (self.wl <= 3e4)
        scale = np.sum(spec_model[mask_pixels]*tdust_phot[mask_pixels]) / np.sum(spec_model[mask_pixels]*spec_model[mask_pixels])
        
        tdust_phot -= scale*spec_model
        tdust_phot[(self.wl < 2.5e4) | (tdust_phot < 0.)] = 0.
        
        norm = np.trapz(tdust_phot, self.wl) 

        dust_phot = np.copy(tdust_phot) * dust_emm / norm
        tdust_phot = 0.
        icnt = 0.
        lboln, lbolo = 0, 1e5
        
        while (lbolo-lboln) > 1e-15 and icnt < 5:
            idust_phot = np.copy(dust_phot)
            dust_phot = dust_phot * 10**(-iav*self.k_cal)

            tdust_phot += dust_phot
            
            lboln = np.trapz(dust_phot, self.wl)
            lbolo = np.trapz(idust_phot, self.wl)
            dust_phot = np.maximum(tdust_phot*(lbolo-lboln)/norm, self.small_num) 
            icnt += 1

        self.dusty_phot_dust = tdust_phot
        flux_model = self._get_mag_single(self.dusty_phot+tdust_phot)

        return flux_model*(10**ilmass), (self.dusty_phot+tdust_phot)*(10**ilmass)

    def _poly_norm(self, spectrum, noise, degree):
        good = np.isfinite(noise) & np.isfinite(spectrum) & (noise > 0) & self.goodpix_spec
        coeff = np.polyfit(self.scaled_lambda[good], spectrum[good], degree, w=1./np.sqrt(noise[good]))
        return np.polyval(coeff, self.scaled_lambda)

    
    def __call__(self, p):
        lp = self.lnprior(p, ndim)
        if not np.isfinite(lp):
            return -np.inf

        lh = self.lnlhood(p)
        if not np.isfinite(lh):
            return -np.inf

        return lh + lp

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, trace):
        gc.collect()


