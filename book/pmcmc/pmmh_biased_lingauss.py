#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Follow-up on: pmmh_lingauss.py

Shows an instance where PMMH may be biased because it fails to explore a
"corner" of the parameter space, where the performance of the bootstrap filter
deteriorates too much.

See the end of the first numerical experiment in Chapter 16 (Figure 16.7 and the
surrounding discussion).

"""
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as random 
import pandas
import seaborn as sb
from scipy import stats
from statsmodels.tsa.stattools import acf

import particles
from particles import distributions as dists
from particles import kalman
from particles import mcmc
from particles import smc_samplers

# prior 
dict_prior = {'varX': dists.Gamma(a=.5, b=1.),
              'varY': dists.Gamma(a=.5, b=.1),
              'rho':dists.Uniform(a=-1., b=1.)
             }
prior = dists.StructDist(dict_prior)

# State-space model 
class ReparamLinGauss(kalman.LinearGauss):
    def __init__(self, varX=1., varY=1., rho=0.):
        sigmaY = np.sqrt(varY) if varY > 0. else 0.
        sigmaX = np.sqrt(varX) if varX > 0. else 0.
        sigma0 = sigmaX
        # Note: We take X_0 ~ N(0, sigmaX^2) so that Gibbs step is tractable
        kalman.LinearGauss.__init__(self, sigmaX=sigmaX, sigmaY=sigmaY, rho=rho,
                                    sigma0=sigma0)
 
data = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')

niter = 10 ** 5
burnin = int(niter/ 10)
algos = OrderedDict()
rw_cov = (0.15)**2 * np.eye(3)

# Basic Metropolis sampler
class StaticLGModel(smc_samplers.StaticModel):
    def loglik(self, theta, t=None): 
        # Note: for simplicity we ignore argument t here,
        # and compute the full log-likelihood
        ll = np.zeros(theta.shape[0])
        for n, th in enumerate(theta): 
            mod = ReparamLinGauss(**smc_samplers.rec_to_dict(th))
            kf = kalman.Kalman(data=data, ssm=mod)
            kf.filter()
            ll[n] = np.sum(kf.logpyt)
        return ll

sm = StaticLGModel(data=data, prior=prior)
algos['mh'] = mcmc.BasicRWHM(model=sm, niter=niter, adaptive=False, 
                             rw_cov=rw_cov, verbose=10)

algos['pmmh'] = mcmc.PMMH(ssm_cls=ReparamLinGauss, prior=prior, data=data,
                          Nx=100, niter=niter, adaptive=False, rw_cov=rw_cov, 
                          verbose=10)
# Run the algorithms 
####################

for alg_name, alg in algos.items():
    print('\nRunning ' + alg_name)
    alg.run()
    print('CPU time: %.2f min' % (alg.cpu_time / 60))

# Plots
#######
plt.style.use('ggplot')
savefigs = True  # False if you don't want to save plots as pdfs

# compare marginals of varY
plt.figure()
plt.hist(algos['mh'].chain.theta['varY'][burnin:], 35, alpha=0.7, density=True, 
         histtype='stepfilled', color='black', label='mh', range=(0., .4))
plt.hist(algos['pmmh'].chain.theta['varY'][burnin:], 35, alpha=0.7, density=True, 
         histtype='stepfilled', color='gray', label='pmmh-100', range=(0., .4))
plt.xlabel(r'$\sigma_Y^2$')
plt.legend()
if savefigs:
    plt.savefig('pmmh_lingauss_biased.pdf')


# compare all the marginals
plt.figure()
for i, param in enumerate(dict_prior.keys()):
    plt.subplot(2, 2, i + 1)    
    for alg_name, alg in algos.items():
        if isinstance(alg, particles.SMC):
            w, par = alg.W, alg.X.theta[param]
        else:
            w, par = None, alg.chain.theta[param][burnin:]
        plt.hist(par, 30, density=True, alpha=0.5, weights=w,
                 label=alg_name, histtype='stepfilled')
    ax = plt.axis()
    xx = np.linspace(ax[0], ax[1], 100)
    plt.plot(xx, np.exp(dict_prior[param].logpdf(xx)), 'k:')
    plt.xlabel(param)
plt.legend()

# MCMC traces
plt.figure()
for i, p in enumerate(dict_prior.keys()):
    plt.subplot(2, 3, i + 1)
    for alg_name, alg in algos.items():
        if hasattr(alg, 'chain'):
            th = alg.chain.theta[p]
            plt.plot(th, label=alg_name)
    plt.xlabel('iter')
    plt.ylabel(p)

# just the MCMC trace of sigmaY^2 for the "biased" algorithm
plt.figure()
alg = algos['pmmh']
th = alg.chain.theta['varY']
plt.plot(th, 'k')
plt.xlabel('iter')
plt.ylabel(r'$\sigma_Y^2$')
if savefigs:
    plt.savefig('pmmh_lingauss_biased_trace_sigmaY2.pdf')

# ACFs (of MCMC algorithms)
nlags = 300
plt.figure()
for i, param in enumerate(dict_prior.keys()):
    plt.subplot(2, 2, i + 1)
    for alg_name, alg in algos.items():
        if not isinstance(alg, particles.SMC):
            plt.plot(acf(alg.chain.theta[param][burnin:], nlags=nlags, fft=True),
                     label=alg_name)
            plt.title(param)
plt.legend()

plt.show()

