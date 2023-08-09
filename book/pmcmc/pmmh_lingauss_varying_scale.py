#!/usr/bin/env python

"""

Follow-up on: pmmh_lingauss.py

This script generates Figure 16.3 in the book, which compares performance
of PMMH for N=100 and different random walk scales, for a linear Gaussian
example.

Warning: takes about 5 hrs to complete.
"""

from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf

import particles
from particles import distributions as dists
from particles import kalman, mcmc


def msjd(theta):
    """Mean squared jumping distance.
    """
    s = 0.
    for p in theta.dtype.names:
        s += np.sum(np.diff(theta[p], axis=0) ** 2)
    return s

# prior
dict_prior = {'varX': dists.InvGamma(a=2., b=2.),
              'varY': dists.InvGamma(a=2., b=2.),
              'rho':dists.Uniform(a=-1., b=1.)
             }
prior = dists.StructDist(dict_prior)

# State-space model
class ReparamLinGauss(kalman.LinearGauss):
    def __init__(self, varX=1., varY=1., rho=0.):
        sigmaX = np.sqrt(varX)
        sigmaY = np.sqrt(varY)
        sigma0 = sigmaX
        # Note: We take X_0 ~ N(0, sigmaX^2) so that Gibbs step is tractable
        kalman.LinearGauss.__init__(self, sigmaX=sigmaX, sigmaY=sigmaY, rho=rho,
                                    sigma0=sigma0)

data = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')


niter = 10 ** 5
burnin = int(niter/ 10)
algos = OrderedDict()

# PMMH algorithms
scales = list(reversed([0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.7, 1.]))
for scale in scales:
    key = 'scale=%g' % scale
    algos[key] = mcmc.PMMH(ssm_cls=ReparamLinGauss, prior=prior,
                           data=data, Nx=100, niter=niter,
                           adaptive=False, rw_cov=scale ** 2 * np.eye(3),
                           verbose=10, scale=scale)
# Run the algorithms
####################

for alg_name, alg in algos.items():
    print('\nRunning ' + alg_name)
    alg.run()
    print('CPU time: %.2f min' % (alg.cpu_time / 60))
    print('mean squared jumping distance: %f' % msjd(alg.chain.theta[burnin:]))

# Plots
#######
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')

# distance vs ar
plt.figure()
x = [alg.acc_rate for alg in algos.values()]
y = [msjd(alg.chain.theta[burnin:]) for alg in algos.values()]
plt.plot(x, y, '-ok')
for i, s in enumerate(scales):
    plt.text(x[i] + 3e-3, y[i] + 5, '%.3g' % s)
plt.ylim(0., 400.)
plt.xlim(0., 0.26)
plt.xlabel('acceptance rate')
plt.ylabel('mean squared jumping distance')
if savefigs:
    plt.savefig('pmmh_lingauss_varying_scales.pdf')

# ACFs (of MCMC algorithms)
nlags = 100
plt.figure()
for i, param in enumerate(dict_prior.keys()):
    plt.subplot(2, 2, i + 1)
    for alg_name in algos.keys():
        if not isinstance(alg, particles.SMC):
            plt.plot(acf(algos[alg_name].chain.theta[param][burnin:],
                         nlags=nlags, fft=True), label=alg_name)
            plt.title(param)
plt.legend()

plt.show()
