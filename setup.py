#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob
import subprocess

try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.build import build

except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.build import build

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

        for root, dirs, files in list(os.walk('mcspf')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build', 'dist', ):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("MC-SPF requires cython to install")


class build_ext(_build_ext):
    def build_extension(self, ext):
        _build_ext.build_extension(self, ext)
 

if __name__ == "__main__":

    include_dirs = ["include", numpy.get_include(),]

    cmodules = []
    cmodules += [Extension("mcspf.utils.magtools", ["mcspf/utils/magtools.pyx"], 
                           include_dirs=include_dirs)]
    cmodules += [Extension("mcspf.utils.sincrebin", ["mcspf/utils/sincrebin.pyx"], 
                           include_dirs=include_dirs)]
    cmodules += [Extension("mcspf.utils.cbroaden", ["mcspf/utils/cbroaden.pyx"], 
                           include_dirs=include_dirs)]
                           
    ext_modules = cythonize(cmodules)


    scripts = ['scripts/'+file for file in os.listdir('scripts/')]  

    cmdclass = {'clean': CleanCommand,
                'build_ext': build_ext}
    
     
    with open('mcspf/_version.py') as f:
      exec(f.read())
      
    setup(
        name = "mcspf",
        url="NO_URL",
        version= __version__,
        author="Matteo Fossati",
        author_email="matteo.fossati@unimib.it",
        ext_modules = ext_modules,
	cmdclass = cmdclass,
        scripts = scripts, 
        packages=["mcspf", 
	          "mcspf.routines", 
		  "mcspf.utils"],
        license="LICENSE",
        description="Monte-Carlo Stellar Population Fitter, (FULL version)",
        install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mpi4py',
          'astropy',
	  'corner',
          'dynesty'],
        package_data={"": ["README.md", "LICENSE"],
	              "mcspf": ["models/Dust_Emi_models/alpha_DH02.dat",
		      "models/Dust_Emi_models/spectra_DH02.dat",
                      "models/Dust_Emi_models/alpha_DL14.dat",
		      "models/Dust_Emi_models/spectra_DL14.dat",
		      "models/Dust_Emi_models/nebular_Byler_mist_2017.lines",
		      "models/Dust_Emi_models/nebular_Byler_mist_2018.lines",
		      "models/Filters/FILTER_LIST",
		      "models/Filters/allfilters.dat",
		      "models/Filters/allindices.dat",
		      "models/Filters/filter_lambda_eff.dat",
		      "models/SPS/Models_exp_bc03hr.fits",
		      "models/SPS/Models_del_bc03hr.fits",
		      ]},
        include_package_data=True,
        zip_safe=False,
    )

