#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPU time vs N for a PMMH kernel; same setting, model, and so on as in 
pmmh_lingauss.py 

This script was used to generate the plot in the Python corner of Chapter 16
(on Bayesian inference and PMCMC algorithms). 
"""

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np

from particles import distributions as dists
from particles import kalman
from particles import mcmc

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
 
# data was simulated as follows: 
# _, data = ReparamLinGauss(varX=1., varY=(0.2)**2, rho=.9).simulate(100)
data = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')


niter = 10 ** 3
algos = OrderedDict()
rw_cov = (0.15)**2 * np.eye(3)

# PMMH algorithms 
Nxs =  list(range(100, 1100, 100))  # needs list for Python 3 

for Nx in Nxs:
    algos[Nx] = mcmc.PMMH(ssm_cls=ReparamLinGauss, prior=prior, data=data,
                           Nx=Nx, niter=niter, adaptive=False, rw_cov=rw_cov, 
                           verbose=10)

# Run the algorithms 
####################

for Nx, alg in algos.items(): 
    print('Nx = %i' % Nx)
    alg.run()
    print('CPU time: %.2f min' % (alg.cpu_time / 60))

# PLOTS
#######
savefigs = False
plt.style.use('ggplot')

plt.figure()
plt.plot(algos.keys(), [alg.cpu_time for alg in algos.values()], 'k')
plt.xlabel(r'$N$')
plt.ylim(bottom=0.)
plt.ylabel('CPU time (s)')
if savefigs: 
    plt.savefig('pmmh_cost_vs_N.pdf')

plt.show()
