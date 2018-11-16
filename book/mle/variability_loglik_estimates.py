#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figures 13.3 and 14 (MLE chapter): variability of the log-likelihood estimate:
    1. as a function of time (N fixed)
    2. when T=O(N) 

The considered model is an univariate linear Gaussian model (so we may use
Kalman to compute the exact likelihood)

Warning: takes 2 days and a half to complete without multi-processing! 
"""

from __future__ import division, print_function

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
sb.set_palette("dark")

import particles
from particles import distributions as dists
from particles import state_space_models as ssm

# set up models, simulate data
maxT = 10**4
my_ssm = ssm.LinearGauss(sigmaX=1., sigmaY=.2, rho=0.9)
_, data = my_ssm.simulate(maxT)
exact = my_ssm.kalman_filter(data)
true_loglik = np.cumsum(exact.logpyts)


def fk_mod(T):
    "FeynmanKac object, given T"
    return ssm.Bootstrap(ssm=my_ssm, data=data[:T])

# Plot the simulated data
plt.style.use('ggplot')
plt.figure()
plt.plot(data)
plt.title('data')
plt.xlabel('t')

savefigs = False  # change this if you want to save the plots as PDFs

# N fixed, log-error as a function of t
# =====================================

nruns = 1000
Ns = [100, 1000]
T = 100
results = particles.multiSMC(fk=fk_mod(T), N=Ns, nruns=nruns)

for N in Ns:
    # plot MSE of log-lik estimate vs time
    ll = np.array([r['output'].summaries.logLts for r in results if r['N'] == N])
    for t in range(T):
        ll[:, t] -= true_loglik[t]

    # confidence intervals + most extreme paths
    plt.figure()
    pctl = lambda x: np.percentile(ll, x, axis=0)
    plt.fill_between(np.arange(T), pctl(10), pctl(90), color='gray', alpha=0.5,
                     label='80% range')
    plt.fill_between(np.arange(T), pctl(25), pctl(75), color='black', alpha=0.5,
                     label='50% range')
    plt.plot(pctl(0), 'k:', label='min-max')
    plt.plot(pctl(100), 'k:')
    plt.xlabel('t')
    plt.ylabel('log-lik error')
    plt.title('N=%i' % N)
    plt.legend(loc=3)
    if savefigs:
        plt.savefig('loglik_error_vs_time_N=%i.pdf' % N)

# N grows with T
################

Ts = [10**k for k in range(1, 5)]
results = []
for T in Ts:
    res_for_T = particles.multiSMC(fk=fk_mod(T), N=10 * T, nruns=nruns)
    for r in res_for_T:
        r['T'] = T
        r['log_error'] = r['output'].summaries.logLts[-1] - true_loglik[T - 1]
    results.extend(res_for_T)

plt.figure()
sb.boxplot(x=[r['T'] for r in results],
           y=[r['log_error'] for r in results],
           color='gray', flierprops={'markerfacecolor': '0.75',
                                     'markersize': 5,
                                     'linestyle': 'none'})
plt.xlabel('T')
plt.ylabel('log-lik error')
plt.ylim((-5., 5.))  # more readable

if savefigs:
    plt.savefig('boxplots_T_grows_with_N.pdf')

plt.show()
