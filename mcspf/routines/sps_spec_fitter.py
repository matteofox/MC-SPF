#!/usr/bin/env python

import numpy as np
import os, warnings
import string, gc, time
from scipy.special import erf
from astropy.modeling import models,fitting
import astropy.io.fits as fits
from scipy.interpolate import RectBivariateSpline as rect
from scipy.ndimage import minimum_filter as minfilt
from bisect import bisect_left
import astropy.units as u
import astropy.stats as stats
import sys
from contextlib import contextmanager

#Imports specific for the full version
from astropy.convolution import convolve_fft, convolve
import numpy.polynomial as poly

from ..utils.magtools  import getmag_spec
from ..utils.readfilt  import init_filters, get_filter
from ..utils.sincrebin import sincrebin_single_os as sincrebin
from ..utils.cbroaden  import broaden

import matplotlib.pyplot as mp

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
            cropspec=[100,20000], spec_in=[None], res_in=[None], polymax=[20,20,20,20,20], \
            fit_spec=True, fit_phot=True, priorAext=None,  Gpriors=None, modeldir='./', filtdir='./', dl=None, cosmo=None, \
            sfh_pars=['TAU','AGE'], sfh_type='exp', sfh_age_par = -1, sfhpar1range = None, sfhpar2range=None, emimetal=0.0, \
            velrange=[-250.,250.], sigrange = [1,500.], fescrange=[0.5,2.0]):
        
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
        self.cropspec = cropspec
        self.polymax = polymax
        self.emimetal = emimetal
        
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
        self.flux_to_lum = (self.Mpc_to_cm*dl)**2*4.*np.pi #4pi*dl^2 in cm^2
        
        #some derived parameters
        self.gal_age = cosmo.age(self.redshift).value #In Gyr here, will be modified later if necessary
        self.dm = 5.*np.log10(dl*1e5) #DM brings source at 10pc
        self.fscale = 10**(-0.4*self.dm)

        ## read in the pre-computed SPS grid
        
        mfile = fits.open(phot_mod_file)
        num_ext = len(mfile)
        try:
          self.mod_lib = mfile[0].header['MODLIB']
        except:
          self.mod_lib = 'unk'
        
        #pull wavelength information from the primary extension of the models
        wdata = np.array(mfile[0].data, dtype=float)
        twl = self.airtovac(wdata)
        
        #get the wavelength information for the DH02 templates to stitch the two together
        dh_wl = np.loadtxt(modeldir+'spectra_DH02.dat', usecols=(0,))*1e4
        dh_nwl = len(dh_wl)

        #expand wavelength grid to include range covered by DH templates
        dh_long = (dh_wl > twl.max())
        self.wl = np.r_[twl, dh_wl[dh_long]]
        self.red_wl = self.wl * (1.+self.redshift)
        self.n_wl = len(self.wl)
        
        #Done with wavelength, now pass through extensions to get grid parameters
        ext_tau, ext_age, ext_metal = [],[],[]
        for ii in range(1,num_ext):
            if ii==1:
             try:
                self.timeunit = mfile[ii].header['AGEUNIT']
             except:
                self.timeunit = 'Myr'
                print('      WARNING: AGEUNIT keyword not found. Using Myr')
            
            tau   = mfile[ii].header[sfh_pars[0]] 
            age   = mfile[ii].header[sfh_pars[1]] 
            ext_tau.append(float(tau))
            ext_age.append(float(age))

        #If model units are in Myr modify cosmo age accordingly
        if self.timeunit == 'Myr':
           self.gal_age *= 1000
        
        self.grid_tau = np.unique(ext_tau)
        self.grid_age = np.unique(ext_age)
                
        self.n_tau = len(self.grid_tau)
        self.n_age = len(self.grid_age)
        
        self.mod_flux_to_lum = 1.196495E40
        
        #output grid
        self.mod_grid = np.zeros((self.n_tau, self.n_age, self.n_wl), dtype=float)
        self.age_grid = np.zeros((self.n_tau, self.n_age, 2),     dtype=float)
        
        #grid where the fractional flux from young populations is stored
        self.fym_grid = np.zeros_like(self.mod_grid)
        
        for ii in range(1,num_ext):            
            mdata  = np.array(mfile[ii].data,  dtype=float)
            mmass  = mfile[ii].header['MSTAR']
            mmetal = mfile[ii].header['METAL']
            if sfh_age_par == -1:
               mage = mfile[ii].header['AGE']
            elif sfh_age_par >=0 and sfh_age_par<=10:
               mage = mfile[ii].header[sfh_pars[sfh_age_par]]
            else:
               mage = sfh_age_par      
            
            mpar0   = mfile[ii].header[sfh_pars[0]] 
            mpar1   = mfile[ii].header[sfh_pars[1]] 
            
            par0_idx = np.where(mpar0 == self.grid_tau)[0]
            par1_idx = np.where(mpar1 == self.grid_age)[0]
            
            self.mod_grid[par0_idx, par1_idx, :] = np.interp(self.wl, twl, mdata[:, 0]/mmass, left=0, right=0)
            self.fym_grid[par0_idx, par1_idx, :] = np.interp(self.wl, twl, mdata[:, 1], left=0, right=0)
            self.age_grid[par0_idx, par1_idx, :] = mage

        mfile.close() 
        
        self.age_max = int(np.nanmax(self.age_grid))
        self.sfh_grid = np.zeros((self.n_tau, self.n_age, self.age_max), dtype=float)
        self.sfh_array = np.ones((self.age_max), dtype=float)
        
        if sfh_type=='custom':
           
           sfh_mod_file = phot_mod_file.replace('.fits','_sfh.fits')
           print('      INFO: Reading SFH file: {}'.format(sfh_mod_file))
           if os.path.isfile(sfh_mod_file):
             sfile = fits.open(sfh_mod_file)
             
             for ii in range(1,num_ext):            
             
               sdata   = np.array(sfile[ii].data,  dtype=float)
               slength = sfile[ii].header['NAXIS2']
               spar0    = sfile[ii].header[sfh_pars[0]] 
               spar1    = sfile[ii].header[sfh_pars[1]] 
               
               par0_idx = np.where(spar0 == self.grid_tau)[0]
               par1_idx = np.where(spar1 == self.grid_age)[0]
               
               if slength>=self.age_max:
                 self.sfh_grid[par0_idx, par1_idx, :] = sdata[:self.age_max,1]
               else:
                 self.sfh_grid[par0_idx, par1_idx, :slength] = sdata[:,1]
                 ratio_extrap = sdata[-1,1]/sdata[-2,1]
                 for ss in range(slength, np.min((slength+1000,self.age_max-1))):
                    self.sfh_grid[par0_idx, par1_idx,ss] = self.sfh_grid[par0_idx, par1_idx,ss-1]*ratio_extrap
               
               self.sfh_array = self.sfh_grid[0, 0,:]
               #self.sfh_grid[par0_idx, par1_idx, slength:] = sdata[-1,1]   
               
               #mp.plot(np.arange(14000),self.sfh_grid[tau_idx,age_idx,:])
               #mp.plot(sdata[:,1])
               #mp.show()           
        
        if (self.mod_lib).lower() == 'bc03':
            #some information on the BC03 spectral resolution (FWHM is 3A in the optical)
            self.sps_res_val = np.copy(self.wl)/300.
            self.sps_res_val[(self.wl >= 3200) & (self.wl <= 9500)] = 3.
        elif (self.mod_lib).lower() == 'cb19':
            self.sps_res_val = np.zeros_like(self.wl)+2.
            self.sps_res_val[(self.wl >= 912) & (self.wl <= 3540)] = 1.
            self.sps_res_val[(self.wl >= 3540) & (self.wl <= 7350)] = 2.5
            self.sps_res_val[(self.wl >= 7350) & (self.wl <= 9400)] = 1.
        else: 
            print('WARNING: Spectral library {} not understood, BC03 resolution will be used for spectral fits.'.format(self.mod_lib))
            self.sps_res_val = np.copy(self.wl)/300.
            self.sps_res_val[(self.wl >= 3200) & (self.wl <= 9500)] = 3.

        self.mod_grid[~np.isfinite(self.mod_grid)] = 0.
        self.fym_grid[~np.isfinite(self.fym_grid)] = 0.

        #pre-compute attenuation curve for photometry
        self.k_cal = self._make_dusty(self.wl)
        
        #redshift the grid to be used in deriving predicted fluxes, also apply
        #flux correction to conserve energy
        #self._redshift_spec()
        
        ### for nebular emission ###
        self.wl_lyman = 912.
        self.ilyman = np.searchsorted(self.wl, self.wl_lyman, side='left') #wavelength just above Lyman limit
        self.lycont_wls = np.r_[self.wl[:self.ilyman], np.array([self.wl_lyman])]
        self.clyman_young = None #A list of two elements, first is for phot, the other for spec
        self.clyman_old   = None #A list of two elements, first is for phot, the other for spec

        self.emm_scales = np.zeros((7,10,128), dtype=float)
        self.emm_wls    = np.zeros(128,        dtype=float)
        self.emm_ages   = np.zeros(10,         dtype=float)
        self.emm_ions   = np.zeros(7,         dtype=float)
        icnt = 0
        rline = 0
        iline = 0
        
        metallist = np.array([0.00020,0.00063246,0.00209426,0.00526054,0.00814761,0.01002374,0.01261915,0.01588656,0.02,0.02517851,0.03169786])
        metalstrg = np.array(['-1.0000e-01','-1.5000e+00','-1.9800e+00','-2.0000e-01','-3.0000e-01','-3.9000e-01','-5.8000e-01','-9.8000e-01','0.0000e+00','1.0000e-01','2.0000e-01'])
        
        metind = np.argmin(np.abs(metallist-self.emimetal))
        print('      INFO: Emission line metallicity requested {}, found {:5.4f}'.format(self.emimetal,metallist[metind]))
        
        with open(modeldir+'nebular_Byler.lines','r') as file:
            for line in file:
                if line[0] != '#':
                    temp = (line.strip()).split(None)
                    if not iline: #Read wave line
                        self.emm_wls[:] = np.array(temp, dtype=float)
                        iline = 1
                    else:
                        if rline: #Read line fluxes
                            self.emm_scales[icnt%7,icnt//7,:] = np.array(temp, dtype=float)*self.lsun #output should be in erg/s/QHO
                            icnt += 1
                        if len(temp) == 3 and temp[0] == metalstrg[metind]:
                            rline = 1
                            self.emm_ages[icnt//7] = float(temp[1])/1e6
                            self.emm_ions[icnt%7]  = float(temp[2])
                        else:
                            rline = 0
        
        #thb = (self.emm_wls > 4860) & (self.emm_wls < 4864)
        #tha = (self.emm_wls > 6560) & (self.emm_wls < 6565)
        #self.emm_scales = np.copy(self.emm_scales) / self.emm_scales[:,:,thb]
                
        #mscale     = np.max(self.emm_scales, axis=(0,1))
        #keep_scale = (mscale > 0.025) & (self.emm_wls<1E5)
        keep_scale = (self.emm_wls<1E5)
                
        self.emm_scales = self.emm_scales[:,:,keep_scale]
        self.emm_wls    = self.emm_wls[keep_scale]
                
        #generate pattern arrays for nebular emission lines
        dpix = np.diff(self.wl)
        self.wl_edges  = np.r_[np.array([self.wl[0]-dpix[0]/2.]), np.r_[self.wl[1:]-dpix/2., np.array([self.wl[-1]+dpix[-1]/2.])]]
        self.res_lines = np.interp(self.emm_wls, self.wl, self.sps_res_val)/2.355
      
        self.emm_lines_all = np.zeros((len(self.emm_ions), len(self.emm_ages), len(self.wl)), dtype=float)
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

        self.dh_dustemm = np.zeros((self.dh_nalpha, self.n_wl), dtype=float)
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
        if spec_in[0] is not None:
            
            self.n_spec = len(spec_in)
            
            self.wl_obj       = []
            self.obj          = []
            self.obj_noise    = []
            self.npix_obj     = []
            self.log_wl       = []
            self.log_wl_edges = []
            self.log_obj      = []
            self.log_noise    = []
            self.goodpix_spec = []
            self.log_model_wl = []
            self.log_model_wl_edges = []
            self.n_model_wl         = []
            self.spec_k_cal         = []
            self.spec_norm          = []
            self.npad               = []
            self.log_spec_grid      = []
            self.log_fysp_grid      = []
            self.log_emm_wls    = []
            self.log_emm_scales = []
            self.diff_pix       = []
            self.vsys_pix       = []
            self.obs_kms        = []
            self.kernel_kms     = []
            self.kms2pix        = []
            self.poly_deg       = []
            self.scaled_lambda  = []
            

            
            for ind in range(self.n_spec):
                self.spec_init(self.spec_in[ind], self.res_in[ind], self.polymax[ind], self.cropspec[2*ind:2*ind+2])
                print('      INFO: Polynomial degree used for spectrum {} is: {}'.format(ind+1,self.poly_deg[ind]))

        else:
          self.n_spec = 0
        
        if sfhpar1range is None:
            if sfh_pars[0].upper() =='AGE':
              maxval_valid = np.min((self.grid_tau.max(), self.gal_age))
            else:
              maxval_valid = self.grid_tau.max()
            self.sfhpar1_lims = np.array((self.grid_tau.min(), maxval_valid))
        else:
            #Verify validity of user requested values
            minval_valid = np.max((self.grid_tau.min(), sfhpar1range[0]))
            if sfh_pars[0].upper() =='AGE':
              maxval_valid = np.min((self.grid_tau.max(), sfhpar1range[1], self.gal_age))
            else:
              maxval_valid = np.min((self.grid_tau.max(), sfhpar1range[1]))
            self.sfhpar1_lims = np.array((minval_valid, maxval_valid))
        print('      INFO: SFH PAR1 range is: {:.1f} - {:.1f}'.format(self.sfhpar1_lims[0], self.sfhpar1_lims[1]))

        if sfhpar2range is None:
            if sfh_pars[1].upper() =='AGE':
              maxval_valid = np.min((self.grid_tau.max(), self.gal_age))
            else:
              maxval_valid = self.grid_tau.max()
            self.sfhpar2_lims = np.array((self.grid_age.min(), maxval_valid))
        else:
            #Verify validity of user requested values
            minval_valid = np.max((self.grid_age.min(), sfhpar2range[0]))
            if sfh_pars[1].upper() =='AGE':
              maxval_valid = np.min((self.grid_age.max(), sfhpar2range[1], self.gal_age))
            else:
              maxval_valid = np.min((self.grid_age.max(), sfhpar2range[1]))
            self.sfhpar2_lims = np.array((minval_valid, maxval_valid))
        print('      INFO: SFH PAR2 range is: {:.1f} - {:.1f}'.format(self.sfhpar2_lims[0], self.sfhpar2_lims[1]))
            
        self.av_lims  = np.array((0., 5.))
        self.ext_lims = np.array((0., 5.))
        self.alpha_lims = np.array((self.dh_alpha[0], self.dh_alpha[-1]))
        self.mass_lims = np.array((3,12))
        self.sig_lims = np.array((sigrange[0], sigrange[1]))
        self.vel_lims = np.array((velrange[0], velrange[1]))
        self.emmsig_lims = np.array((1.,200.)) 
        self.emmage_lims = np.array((self.emm_ages.min(), 10))
        self.emmion_lims = np.array((self.emm_ions.min(), self.emm_ions.max()))
        self.fesc_lims = np.array((fescrange[0], fescrange[1]))
        self.lnf_lims = np.array((-2, 2))
        
        if (sigrange[0] != 1) or (sigrange[1] != 500):
            print('      INFO: Custom stellar sigma range is: {} - {}'.format(sigrange[0], sigrange[1]))

        if (velrange[0] != -250) or (velrange[1] != 250):
            print('      INFO: Custom stellar velocity range is: {} - {}'.format(velrange[0], velrange[1]))

        
        self.bounds = [self.sfhpar1_lims, self.sfhpar2_lims, self.av_lims, \
                self.ext_lims, self.alpha_lims,  self.mass_lims, \
                self.sig_lims, self.vel_lims, self.emmsig_lims, self.emmage_lims,\
                self.emmion_lims, self.fesc_lims, self.lnf_lims, self.lnf_lims]
                
        self.ndims = len(self.bounds)   
        
    def spec_init(self, specfile, resfile, polymax, cropspec):

        #so it can be used in the fit           
        file = fits.open(specfile)
        obj       = np.asarray(file[0].data, dtype=float)
        obj_noise = np.asarray(file[1].data, dtype=float)
        wl_obj    = np.asarray(file[2].data, dtype=float)
                
        #open up resolution file
        rfile = fits.open(resfile)
        rhdr = rfile[0].header
        
        lsf_res = np.asarray(rfile[0].data, dtype=float) #This is R
        crval, cdelt, crpix, size = rhdr['CRVAL1'], rhdr['CD1_1'], 1., rhdr['NAXIS1']
        lsf_wl = ((np.arange(size, dtype=float)+1 - crpix)*cdelt + crval)
        lsf_res = lsf_wl / lsf_res #Now in FWHM in A
        rfile.close()
        
        #clip off the spectrum as requested by the user
        range_crop = (wl_obj > cropspec[0]) & (wl_obj < cropspec[1])
        
        wl_obj = wl_obj[range_crop]
        obj = obj[range_crop] * 1E-20
        obj_noise = obj_noise[range_crop] * 1E-20
        npix_obj = len(wl_obj)
        
        goodpix_lin =  np.isfinite(obj) & np.isfinite(obj_noise)
        if goodpix_lin.sum() <10:
           print('    WARNING: Too few spectral valid points. Spectrum will not be fit.')
           self.fit_spec=False
        
        #rebin the object to log
        log_wl, dlam_log = np.linspace(np.log10(wl_obj[0]), np.log10(wl_obj[-1]), npix_obj, retstep=True)
        log_wl_edges = np.r_[log_wl-dlam_log/2., log_wl[-1:]+dlam_log/2.]
        
        log_obj = sincrebin(10**log_wl, wl_obj, obj)
        log_noise = np.sqrt(sincrebin(10**log_wl, wl_obj, obj_noise**2))
        
        #mask nans, pixels with zero noise and data equal to zero
        goodpix_spec = np.isfinite(log_obj) & np.isfinite(log_noise) & (log_noise>0) & (log_obj!=0)

        #Filter to avoid edge effects due to sincrebin
        goodpix_spec = minfilt(goodpix_spec, size=3)
                  
        #mp.plot(log_wl, log_obj, 'k-')
        #mp.plot(log_wl[~goodpix_spec], log_obj[~goodpix_spec], 'ro', mec='red', ms=3.0)
        #mp.show()
        
        #setup for polynomial normalization
        low_log, high_log = min(log_wl), max(log_wl)
        slam_log = high_log-low_log
        mlam_log = (high_log+low_log)/2.
        
        low_lin, high_lin = min(wl_obj), max(wl_obj)
        mlam_lin = (high_lin+low_lin)/2.
        
        scaled_lambda = (log_wl - mlam_log) * 2/slam_log
        poly_deg = min(int((high_lin - low_lin)/150.), polymax)
        
        #normalize the observed spectrum
        #norm_window = ((log_wl > np.log10(mlam_lin-125)) & (log_wl < np.log10(mlam_lin+125)))
        #spec_norm = np.nansum(log_obj[norm_window]/log_noise[norm_window]**2)/np.nansum(1./log_noise[norm_window]**2)
        
        dummy, spec_norm, dummy = stats.sigma_clipped_stats(log_obj[goodpix_spec])
        
        log_obj   = log_obj/spec_norm
        log_noise = log_noise/spec_norm
                
        #Prepare absorption profile (wave must be rest)    
        spec_k_cal = self._make_dusty((10**log_wl)/(1+self.redshift))

        #Now work through the models
        use_wl = (self.wl > 900) & (self.wl < 20000)
        model_spec_wl = self.wl[use_wl] * (1+self.redshift)
        log_model_wl = np.arange(np.log10(model_spec_wl[0]),np.log10(model_spec_wl[-1])+dlam_log/2., dlam_log)
        log_model_wl_edges = np.r_[log_model_wl-dlam_log/2., log_model_wl[-1:]+dlam_log/2.]
        n_model_wl = len(log_model_wl)
        npad = 2**int(np.ceil(np.log2(n_model_wl)))

        #mask pixels outside the template range
        goodpix_spec[log_wl < min(log_model_wl)] = False
        goodpix_spec[log_wl > max(log_model_wl)] = False
                
        #spectral resolutions in FWHM in A, for observations interpolate at observed-frame waves
        sps_res = np.interp(10**log_wl, self.red_wl, self.sps_res_val) 
        obs_res = np.interp(10**log_wl, lsf_wl, lsf_res)
                
        kms2pix = np.log(10)*self.clight*dlam_log/1e13

        sps_kms = (sps_res/2.355)*(self.clight/1e13)/((10**log_wl)/(1+self.redshift))
        obs_kms = (obs_res/2.355)*(self.clight/1e13)/((10**log_wl)) #Wave here has to be obs frame
        kernel_kms = np.sqrt(obs_kms**2 - sps_kms**2) #if positive then observations have lower resolution

        #fig, ax = mp.subplots()
        #ax.plot(10**log_wl, sps_kms, '-k')
        #ax.plot(10**log_wl, obs_kms, '-r')
        #mp.show()
                
        #set up storage for log models
        log_spec_grid = np.zeros((self.n_tau, self.n_age, n_model_wl), dtype=float)
        log_fysp_grid = np.zeros((self.n_tau, self.n_age, n_model_wl), dtype=float)
        
        cnt = 0
        for jj in range(self.n_tau):
            for kk in range(self.n_age):
                 this_templ = self.mod_grid[jj,kk,use_wl]  
                 this_young = self.fym_grid[jj,kk,use_wl] * np.copy(this_templ)
                 log_spec  = np.interp(10**log_model_wl,  model_spec_wl, this_templ/(1+self.redshift))
                 log_young = np.interp(10**log_model_wl,  model_spec_wl, this_young/(1+self.redshift))

                 log_spec_grid[jj,kk,:] = log_spec
                 log_fysp_grid[jj,kk,:] = np.copy(log_young)/np.copy(log_spec)
        

        #cut down emission line arrays for the spectral grid
        log_emm_use = (self.emm_wls*(1+self.redshift) > 10**log_model_wl[0]) & (self.emm_wls*(1+self.redshift) < 10**log_model_wl[-1])
        log_emm_wls = self.emm_wls[log_emm_use] *(1+self.redshift)
        log_emm_scales = self.emm_scales[:,:,log_emm_use]
        
        tha = np.where((log_emm_wls/(1+self.redshift) > 6560) & (log_emm_wls/(1+self.redshift) < 6565))
        thb = np.where((log_emm_wls/(1+self.redshift) > 4858) & (log_emm_wls/(1+self.redshift) < 4864))
        #print tha, thb
        
        diff_pix = (log_model_wl_edges[None,:] - np.log10(log_emm_wls)[:,None])/dlam_log #in pixels
        vsys_pix = (log_model_wl[0] - log_wl[0])*np.log(10)*self.clight/kms2pix/1e13
        
        #Add relevant values to arrays
        self.wl_obj.append(wl_obj)
        self.obj.append(obj)
        self.obj_noise.append(obj_noise)
        self.npix_obj.append(npix_obj)
        self.log_wl.append(log_wl)
        self.log_wl_edges.append(log_wl_edges)
        self.log_obj.append(log_obj)
        self.log_noise.append(log_noise) 
        self.goodpix_spec.append(goodpix_spec) 
        self.log_model_wl.append(log_model_wl)
        self.log_model_wl_edges.append(log_model_wl_edges)
        self.n_model_wl.append(n_model_wl)
        self.spec_norm.append(spec_norm)
        self.spec_k_cal.append(spec_k_cal)
        self.npad.append(npad)
        self.poly_deg.append(poly_deg)
        self.kms2pix.append(kms2pix)
        self.kernel_kms.append(kernel_kms)
        self.obs_kms.append(obs_kms)
        self.log_spec_grid.append(log_spec_grid)
        self.log_fysp_grid.append(log_fysp_grid)
        self.log_emm_wls.append(log_emm_wls)
        self.log_emm_scales.append(log_emm_scales)
        self.scaled_lambda.append(scaled_lambda)
        self.diff_pix.append(diff_pix)
        self.vsys_pix.append(vsys_pix)
    
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
    
    #This should be used with MultiNest 
    def _scale_cube_mn(self, cube, ndims, nparams):
        for ii in range(ndims):
            cube[ii] = cube[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])

        return
    
    #This should be used with UltraNest
    def _scale_cube_un(self, cube):
        sc_cube = np.copy(cube)
        for ii in range(self.ndims):
            sc_cube[ii] = sc_cube[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])
        return sc_cube

    def _losvd_rfft(self, vel, sigma, spid):
        """Generate analytic fourier transform of the LOSVD following Cappellari et al. 2016 and pPXF,
        simplified for purposes here"""
        nl = self.npad[spid]//2 + 1
        losvd_rfft = np.zeros(nl, dtype=complex)
        vel_pix = vel/self.kms2pix[spid] + self.vsys_pix[spid]
        sigma_pix = sigma/self.kms2pix[spid] #20230718 Removed sigma**2 due to issue with units

        #compute the FFT
        a = vel_pix / sigma_pix
        w = np.linspace(0, np.pi*sigma_pix, nl)
        losvd_rfft[:] = np.exp(1j*a*w - 0.5*w**2)

        return np.conj(losvd_rfft)

    def _vel_convolve_fft(self, spec, sigma, vel, spid):
        
        #convolve input spectrum to given velocity
        #generate kernel (in pixels)
        fft_losvd = self._losvd_rfft(vel, sigma, spid)

        #pre-compute fft of the continuum template library
        fft_template = np.fft.rfft(spec, n=self.npad[spid], axis=0)

        tmpl = np.fft.irfft(fft_template*fft_losvd)

        return tmpl[:self.n_model_wl[spid]]

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
                    ftrans, left=0., right=0.), dtype=float) 

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
        
        tflux = np.zeros(self.n_bands, dtype=float)

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
               pav  +=  -0.5*(((p[3]-aval)/sav)**2 + np.log(2.*np.pi*sav**2))
                             
            return pav

        return -np.inf

    def lnlhood_worker(self, p):
        
        spec_lhood = 0
        phot_lhood = 0
                
        if self.fit_spec == True:
          
          for ss in range(self.n_spec):
            model_spec = self.reconstruct_spec(p, self.ndims, ss)
        
            if np.all(model_spec == 0.):
                return -np.inf

            ispec2 = 1./((self.log_noise[ss]*np.exp(p[12+ss]))**2)

            spec_lhood += -0.5*np.nansum((ispec2*(self.log_obj[ss]-model_spec)**2 - np.log(ispec2) + np.log(2.*np.pi))[self.goodpix_spec[ss]])
                  
           
        model_phot, _ = self.reconstruct_phot(p, self.ndims)
         
        if np.all(model_phot == 0.) or np.any(~np.isfinite(model_phot)):
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
             phot_lhood = -0.5*np.nansum(((iphot2*(self.flux_obs-model_phot)**2) - np.log(iphot2) + np.log(2.*np.pi)))
        
        #### APPLY THE PRIOR HERE  #####
        pr = self.lnprior(p, self.ndims)
        
        if not np.isfinite(pr):
            return -np.inf
        
        return spec_lhood + phot_lhood + pr
    
    #This should be used with Multinest
    def lnlhood_mn(self, p, ndim, nparams):
        
        return self.lnlhood_worker(p)
    
    #This should be used with UltraNest    
    def lnlhood_un(self, p):
        
        val = self.lnlhood_worker(p)
        if np.isfinite(val):
          return val
        else:
         return -1e70  
        

        
    def _get_clyman(self, spec, fesc): #compute number of Lyman continuum photons
        #spectrum in erg/s/cm2/A so it returns the flux of ionizing photons not the rate
        lycont_spec = np.interp(self.lycont_wls, self.wl, spec) 
        nlyman = np.trapz(lycont_spec*self.lycont_wls, self.lycont_wls)/self.h/self.clight
 
        #modify input spectrum to remove photons 
        if fesc>0:
           spec[:self.ilyman] *= fesc
    
        return nlyman*(1.-fesc), spec
      
    def _get_nebular(self, emm_spec): 
        
        emm_young = self.clyman_young * emm_spec 
        emm_old   = self.clyman_old   * emm_spec 
        
        return emm_young, emm_old

    def _make_spec_emm(self, vel, sigma, emm_age, emm_ion, spid):
        
        vel_pix = vel/self.kms2pix[spid] + self.vsys_pix[spid]
        
        #Here assume the narrowest line is defined by the resolution of the observations
        #Need to extend the array by one element because of the diff functions below. 
        #Resolution is a smooth function so it should be ok
        obs_kms = np.r_[self.obs_kms[spid],self.obs_kms[spid][-1]]
        sigma_pix = np.sqrt(sigma**2+obs_kms**2)/self.kms2pix[spid]

        temp_emm_scales = self._bi_interp(self.log_emm_scales[spid], emm_ion, emm_age, self.emm_ions, self.emm_ages) 
                
        emm_grid = np.sum(temp_emm_scales[:,None]*\
                np.diff(0.5*(1.+erf((self.diff_pix[spid]-vel_pix)[:,:self.npix_obj[spid]+1]/np.sqrt(2.)/sigma_pix)), axis=1)/\
                np.diff(10**self.log_wl_edges[spid])[None,:], axis=0)
        
        return emm_grid

    def reconstruct_spec(self, p, ndim, spid, retall=False):
        
        #Parameters
        itau, iage, iav, iav_ext, _, ilmass, isig, ivel,  isig_gas, iage_gas, iion_gas, ifesc, ilnf0, ilnf1 = [p[x] for x in range(ndim)]
        
        #interpolate the full photometric grid (non redshifted for lyman continuum photons)
        spec_model = self._bi_interp(self.mod_grid, itau, iage, self.grid_tau, self.grid_age)
        frac_model = self._bi_interp(self.fym_grid, itau, iage, self.grid_tau, self.grid_age)

        #get number of lyman continuum photons
        self.clyman_young, temp_young = self._get_clyman(spec_model*frac_model, ifesc)
        self.clyman_old,   temp_old   = self._get_clyman(spec_model*(1.-frac_model), ifesc)
        
        #build the emission line arrays and attenuate given Av and A_extra
        emm_lines_spec     = self._make_spec_emm(ivel, isig_gas, iage_gas, iion_gas, spid)
        emm_young, emm_old = self._get_nebular(emm_lines_spec)
        
        #Fobs = Fint*10^(-0.4*Alam/Av*Av)
        dusty_emm = ((10**(-iav*self.spec_k_cal[spid]) * emm_old) + (10**(-(iav+iav_ext)*self.spec_k_cal[spid]) * emm_young))
        
        #interpolate the spectral model grid to given values (this is in the observed frame, i.e. redshifted)
        hr_spec_model = self._bi_interp(self.log_spec_grid[spid], itau, iage, self.grid_tau, self.grid_age)
        hr_frac_model = self._bi_interp(self.log_fysp_grid[spid], itau, iage, self.grid_tau, self.grid_age)
                
        #attenuate the continuum model and convolve to given dispersion
        spec_young = self._vel_convolve_fft(hr_spec_model*hr_frac_model, isig, ivel, spid)[:self.npix_obj[spid]]
        spec_old   = self._vel_convolve_fft(hr_spec_model*(1.-hr_frac_model), isig, ivel, spid)[:self.npix_obj[spid]]
        
        #Now convolve to intrumental resolution only if observations are lower res than model
        if np.all(np.isfinite(self.kernel_kms[spid])):
              spec_young = broaden(spec_young, self.kernel_kms[spid]/self.kms2pix[spid])
              spec_old   = broaden(spec_old,   self.kernel_kms[spid]/self.kms2pix[spid])
                        
        dusty_spec = ((10**(-iav*self.spec_k_cal[spid]) * spec_old) + (10**(-(iav+iav_ext)*self.spec_k_cal[spid]) * spec_young))
                
        #rescaled variance
        if spid==0:
           scale_sig = self.log_noise[spid]*np.exp(ilnf0)
        else:
           scale_sig = self.log_noise[spid]*np.exp(ilnf1)        
        
        #remove shape differences between spectrum and model
        try:
           if self.poly_deg[spid]>0:
             cont_poly = self._poly_norm(self.log_obj[spid]/(dusty_spec+dusty_emm), scale_sig**2/(dusty_spec+dusty_emm)**2, self.poly_deg[spid], spid)
           else:
             print('     WARNING: Using fixed scaling of spectra')
             cont_poly = np.ones_like(dusty_spec) * self.fscale * 10**ilmass / self.spec_norm[spid]
        except:
           return np.zeros((2))
        
        #fig, ax = mp.subplots()
        #ok = (self.goodpix_spec[spid]) 
        #ax.scatter(10**self.log_wl[spid][ok], self.log_obj[spid][ok], marker='o', color='red')
        #ax.plot(10**self.log_wl[spid][ok], self.log_obj[spid][ok], '-k')
        #ax.plot(10**self.log_wl[spid][ok], ((dusty_spec+dusty_emm)*cont_poly)[ok], '-r')
        #ax.plot(10**self.log_wl[spid][ok], ((dusty_spec)/np.median(dusty_spec))[ok], '-b')
        #mp.show()
                        
        totspec = (dusty_spec + dusty_emm)*cont_poly
        contspec = dusty_spec*cont_poly
        emispec  = dusty_emm*cont_poly
                
        if retall:
          return totspec, contspec, emispec
        else:
          #return the re-normalized model + emission lines
          return  totspec


    def reconstruct_phot(self, p, ndim):
        #parameters
        itau, iage, iav, iav_ext, ialpha, ilmass, _, _, _, iage_gas, iion_gas, ifesc, _, _ = [p[x] for x in range(ndim)]
        
        #interpolate the full photometric grid (non redshifted for lyman continuum photons)
        spec_model = self._bi_interp(self.mod_grid, itau, iage, self.grid_tau, self.grid_age)
        frac_model = self._bi_interp(self.fym_grid, itau, iage, self.grid_tau, self.grid_age)

        #get number of lyman continuum photons
        self.clyman_young, temp_young   = self._get_clyman(spec_model*frac_model, ifesc)
        self.clyman_old,   temp_old     = self._get_clyman(spec_model*(1.-frac_model), ifesc)
        
        #### Include nebular emission ####
        iemm_lines = self._bi_interp(self.emm_lines_all, iion_gas, iage_gas, self.emm_ions, self.emm_ages) 
        emm_young, emm_old = self._get_nebular(iemm_lines)

        #Send model to observed frame, scale flux down and wave up (wave is pre computed as self.wl_red)
        tot_young = (temp_young+emm_young) / (1+self.redshift)
        tot_old   = (temp_old  +emm_old) / (1+self.redshift)
        
        #attenuate photometry spectrum, then compute fluxes given input bands
        self.dusty_phot_young = (10**(-(iav+iav_ext)*self.k_cal) * (tot_young))
        self.dusty_phot_old   = (10**(-iav*self.k_cal) * (tot_old))
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

    def _poly_norm(self, spectrum, noise, degree, spid):
        good = np.isfinite(noise) & np.isfinite(spectrum) & (noise > 0) & self.goodpix_spec[spid]
        coeff = np.polyfit(self.scaled_lambda[spid][good], spectrum[good], degree, w=1./np.sqrt(noise[good]))
        return np.polyval(coeff, self.scaled_lambda[spid])

    def reconstruct_sfh(self, p, ndim):
        #parameters
        ipar0, ipar1, _, _, _, _, _, _, _, _, _, _, _, _ = [p[x] for x in range(ndim)]
        
        #interpolate the sfh grid
        sfh_interp = self._bi_interp(self.sfh_grid, ipar0, ipar1, self.grid_tau, self.grid_age)
        age_interp = self._bi_interp(self.age_grid, ipar0, ipar1, self.grid_tau, self.grid_age)

        return sfh_interp, age_interp[0]
    
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


