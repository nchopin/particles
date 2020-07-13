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
from malikpitt_interpolation import MalikPitt_SMC
from matplotlib import pyplot as plt
from numpy import random

import particles
from particles import ensemble_transform as et
from particles import state_space_models as ssms

# data 
raw_data = np.loadtxt('../../datasets/GBP_vs_USD_9798.txt',
                      skiprows=2, usecols=(3,), comments='(C)')
T = 200
data = 100. * np.diff(np.log(raw_data[:(T + 1)]))


def fkmod(theta):
    mu = theta[0];
    rho = theta[1];
    sigma = theta[2]
    return ssms.Bootstrap(ssm=ssms.StochVol(mu=mu, rho=rho, sigma=sigma),
                          data=data)


def loglik(theta, seed=None, qmc=False, N=10 ** 4, verbose=False, interpol=False, transform=False, epsilon=0.,
           corrected=False):
    if seed is not None:
        random.seed(seed)
    if interpol:
        alg = MalikPitt_SMC(fk=fkmod(theta), N=N)
    elif transform:
        if epsilon:
            if corrected:
                et_instance = et.CorrectedEntropyRegularizedEnsembleTransform(epsilon)
            else:
                et_instance = et.EntropyRegularizedEnsembleTransform(epsilon)

        else:
            et_instance = et.EnsembleTransform()
        alg = et.EnsembleTransformFilter(et_instance,
                                         fk=fkmod(theta),
                                         N=N,
                                         qmc=qmc)

    else:
        alg = particles.SMC(fk=fkmod(theta), N=N, qmc=qmc)
    alg.run()
    out = alg.logLt
    if verbose:
        print(theta, out)
    return out


if __name__ == "__main__":
    # theta0 = [-1., 0.9, 0.3]
    # mle = fmin(objfunc, theta0, xtol=.1, ftol=.1, maxfun=1000)

    sig_min = 0.2
    sig_max = 0.5
    mu = -1.
    rho = 0.9
    sigmas = np.linspace(sig_min, sig_max, 20)
    print("SMC")
    # SMC
    ll = [loglik([mu, rho, sigma]) for sigma in sigmas]
    # SMC frozen seed
    print("SMC frozen")
    ll_frozen = [loglik([mu, rho, sigma], seed=4) for sigma in sigmas]
    # interpolation
    print("SMC interpolated")
    ll_pol = [loglik([mu, rho, sigma], seed=4, interpol=True) for sigma in sigmas]
    print("ET")
    # ensemble transform
    ll_et = [loglik([mu, rho, sigma], seed=4, transform=True) for sigma in sigmas]
    # entropy ensemble transform
    print("Reg-ET")
    ll_reg_et = [loglik([mu, rho, sigma], seed=4, transform=True, epsilon=0.1) for sigma in sigmas]
    print("Corrected-ET")
    ll_reg_corr = [loglik([mu, rho, sigma], seed=4, transform=True, epsilon=0.1, corrected=True) for sigma in sigmas]

    # QMC
    ll_qmc = [loglik([mu, rho, sigma], qmc=True) for sigma in sigmas]
    # ET QMC
    ll_et_qmc = [loglik([mu, rho, sigma], qmc=True, transform=True) for sigma in sigmas]
    # REG ET QMC
    ll_reg_et_qmc = [loglik([mu, rho, sigma], qmc=True, transform=True, epsilon=0.1) for sigma in sigmas]
    # ET QMC
    ll_corr_et_qmc = [loglik([mu, rho, sigma], qmc=True, transform=True, epsilon=0.1, corrected=True) for sigma in sigmas]

    # PLOT
    # ====
    savefigs = True  #  False if you don't want to save plots as pdfs

    plt.figure()
    plt.style.use('seaborn-dark')
    plt.grid('on')
    plt.plot(sigmas, ll, '+', color='gray', label='standard')
    plt.plot(sigmas, ll_frozen, 'o', color='gray', label='fixed seed')
    plt.plot(sigmas, ll_pol, color='gray', label='interpolation')
    plt.plot(sigmas, ll_qmc, 'k.', lw=2, label='SQMC')
    plt.plot(sigmas, ll_et, 'k.', lw=2, label='ET')
    plt.plot(sigmas, ll_reg_et, 'k*', lw=2, label='Reg-ET')
    plt.plot(sigmas, ll_reg_corr, 'k--', lw=2, label='Corrected-ET')
    plt.plot(sigmas, ll_et_qmc, 'ko', lw=2, label='ET-QMC')
    plt.plot(sigmas, ll_reg_et_qmc, 'k.-', lw=2, label='REG-QMC')
    plt.plot(sigmas, ll_corr_et_qmc, 'k..', lw=2, label='CORR-QMC')
    plt.xlabel('sigma')
    plt.ylabel('log-likelihood')
    plt.legend()
    if savefigs:
        plt.savefig('loglik_interpolated_frozen_sqmc.pdf')

    plt.show()
