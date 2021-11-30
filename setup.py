#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import setup
import sys
import warnings

NAME = 'particles'
DESCRIPTION = 'Sequential Monte Carlo in Python'

with open('README.md') as f:
    long_description = f.read()

METADATA = dict(
    name=NAME,
    version='0.3',
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

setup(**METADATA)
