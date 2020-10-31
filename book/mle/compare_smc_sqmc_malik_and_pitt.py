#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Produces Figure 13.7 that compares likelihood estimates obtained from:

    * a standard particle filter
    * a standard pf, but with "common random numbers" (RNG seed is kept
    constant)
    * SQMC 
    * the interpolation method of Malik & Pitt (2011) 

    See Section 13.3 (derivative-free approaches to compute the MLE) for more 
    details. 

    Note: takes 2-3 hrs.
"""

from __future__ import division, print_function

import numpy as np
from numpy import random
from matplotlib import pyplot as plt

import particles
from particles import datasets as dta
from particles import state_space_models as ssms

from malikpitt_interpolation import MalikPitt_SMC

# data 
T = 200
data = dta.GBP_vs_USD_9798().data[:(T + 1)]

def fkmod(theta):
    mu = theta[0]; rho = theta[1]; sigma = theta[2] 
    return ssms.Bootstrap(ssm=ssms.StochVol(mu=mu, rho=rho, sigma=sigma), 
                         data=data)

def loglik(theta, seed=None, qmc=False, N=10**4, verbose=False, interpol=False):
    if seed is not None:
        random.seed(seed)
    if interpol:
        alg = MalikPitt_SMC(fk=fkmod(theta), N=N) 
    else:
        alg = particles.SMC(fk=fkmod(theta), N=N, qmc=qmc)
    alg.run()
    out = alg.logLt
    if verbose:
        print(theta, out)
    return out

sig_min = 0.2; sig_max = 0.5
mu = -1.; rho = 0.9 
sigmas = np.linspace(sig_min, sig_max, 1000)

# SMC 
ll = [loglik([mu, rho, sigma]) for sigma in sigmas]
# SMC frozen seed
ll_frozen = [loglik([mu, rho, sigma], seed=4) for sigma in sigmas]
# interpolation
ll_pol = [loglik([mu, rho, sigma], seed=4, interpol=True) for sigma in sigmas]
# QMC 
ll_qmc  = [loglik([mu, rho, sigma], qmc=True) for sigma in sigmas]

# PLOT
# ====
savefigs = True  # False if you don't want to save plots as pdfs

plt.figure()
plt.style.use('seaborn-dark')
plt.grid('on') 
plt.plot(sigmas, ll, '+', color='gray', label='standard')
plt.plot(sigmas, ll_frozen, 'o', color='gray', label='fixed seed')
plt.plot(sigmas, ll_pol, color='gray', label='interpolation')
plt.plot(sigmas, ll_qmc,'k.', lw=2, label='SQMC')
plt.xlabel(r'$\sigma$')
plt.ylabel('log-likelihood')
plt.legend()
if savefigs:
    plt.savefig('loglik_interpolated_frozen_sqmc.pdf') 

plt.show()
