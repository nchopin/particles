#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from setuptools import setup, find_packages
import io
import setuptools  # only mention of setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

import particles

NAME = 'particles'

with open('README.md') as f:
    long_description = f.read()

ext = Extension(name=NAME + ".lowdiscrepancy", 
              sources=["src/LowDiscrepancy.f"])
extensions = [ext,]

setup(
    name=NAME, 
    version='0.1', 
     url='http://github.com/nchopin/particles/',
    license='MIT', 
    author='Nicolas Chopin',
    # tests_require=['pytest'],
    # install_requires=['numpy>=1.9',
    #                   'scipy'
    #                   ],
    #cmdclass={'test': PyTest},
    author_email='nicolas.chopin@ensae.fr',
    description='Sequential Monte Carlo in python', 
    # long_description=long_description,
    packages=['particles'],
    include_package_data=True,
    platforms='any',
    # test_suite='sandman.test.test_sandman',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    ext_modules=extensions
)
