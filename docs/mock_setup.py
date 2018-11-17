#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This mock setup may be used to build the documentation on "Read the docs". 
The actual setup needs to compile fortran files, and RTD does not like it.
"""

from setuptools import setup, find_packages
import io

import particles

with open('README.md') as f:
    long_description = f.read()

setup(
    name='particles',
    version='0.1', 
    url = 'https://github.com/nchopin/particles', 
    license='MIT', 
    author='Nicolas Chopin',
    install_requires=['numpy',
                      'scipy', 
                      'numba'
                     ],
    author_email='nicolas.chopin@ensae.fr',
    description='Sequential Monte Carlo in python', 
    long_description=long_description,
    packages=['particles'],
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
