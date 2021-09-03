#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import setuptools  
import sys
import warnings

NAME = 'particles'
DESCRIPTION = 'Sequential Monte Carlo in Python'

with open('README.md') as f:
    long_description = f.read()

METADATA = dict(
    name=NAME, 
    version='0.2', 
    url='http://github.com/nchopin/particles/',
    license='MIT', 
    author='Nicolas Chopin',
    install_requires=['numpy>=1.18',
                      'scipy>=1.7',
                      'numba',
                      'joblib'
                      ],
    author_email='nicolas.chopin@ensae.fr',
    description=DESCRIPTION, 
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=[NAME], 
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
    ]
)

fortran_warning = """
lowdiscrepancy fortran module could not be built (missing compiler? see INSTALL
notes). Package should work as expected, except for the parts related to QMC
(quasi-Monte Carlo). 
"""

# detect that Read the docs is trying to build the package
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # RTD does not have a fortran compiler
    from setuptools import setup
else:
    # Try to install using Fortran, if the compiler
    # cannot be found, switch to traditional installation.
    try:
        from numpy.distutils.core import setup
        from numpy.distutils.extension import Extension
        ext = Extension(name=NAME + ".lowdiscrepancy", 
                        sources=["src/LowDiscrepancy.f"])
        METADATA['ext_modules'] = [ext,]
    except:
        from setuptools import setup
        warnings.warn(fortran_warning)

setup(**METADATA)
