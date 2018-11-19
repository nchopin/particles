#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import setuptools  
import sys

import particles

NAME = 'particles'
DESCRIPTION = 'Sequential Monte Carlo in Python'

with open('README.md') as f:
    long_description = f.read()

METADATA = dict(
    name=NAME, 
    version='0.1', 
    url='http://github.com/nchopin/particles/',
    license='MIT', 
    author='Nicolas Chopin',
    install_requires=['numpy',
                      'scipy',
                      'numba'
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

# Readthedocs uses the --force option, and does not work if we ask to 
# compile fortran code, so we disable this in that case.
if '--force' in sys.argv: 
    from setuptools import setup 
else:
    from numpy.distutils.core import setup
    from numpy.distutils.extension import Extension
    ext = Extension(name=NAME + ".lowdiscrepancy", 
                    sources=["src/LowDiscrepancy.f"])
    METADATA['ext_modules'] = [ext,]
    
setup(**METADATA)
