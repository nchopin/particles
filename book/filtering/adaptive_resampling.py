#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive resampling means that we resample every time the ESS gets smaller 
than some threshold ESS_min. This script assesses the impact of ESS_min on the
variance of particle estimates (specifically log-likelihood estimates). 

The considered algorithm is the (perfect) guided filter associated to a
univariate linear Gaussian model. By perfect, I mean that the 'optimal'
proposal is used, i.e. the distribution of X_t given X_{t-1} and Y_t. 

The resulting plot appears in Figure 10.2 in the book. 

Warning: takes about 30 min to complete (on a single core; change nprocs to 
zero to use all cores). 
"""

from __future__ import division, print_function

from matplotlib import pyplot as plt
import numpy as np

import particles
from particles import kalman
from particles import state_space_models

# parameter values  
sigmaX = 1.
sigmaY = .2
rho = 0.9 
T = 1000
N = 200 

# define ss model, simulate data
ssm = kalman.LinearGauss(sigmaX=sigmaX, sigmaY=sigmaY, rho=rho)
true_states, data = ssm.simulate(T)

# computes true log-likelihood
kf = kalman.Kalman(ssm=ssm, data=data)
kf.filter()
true_loglik = np.sum(kf.logpyt)

# FK model 
fk_model = state_space_models.GuidedPF(ssm=ssm, data=data)

# Run SMC algorithm for different values of ESS_min
alphas = list(np.linspace(0., .1, 11)) + list(np.linspace(0.15, 1., 18))
results = particles.multiSMC(fk=fk_model, N=N, ESSrmin=alphas, nruns=200,
                          nprocs=1) 

# PLOTS
#======
plt.style.use('ggplot')
savefigs = False

# inter-quartile range of log-likelihood estimate as a function of ESSmin
plt.figure()    
quartiles = np.zeros((2, len(alphas)))
for i, q in enumerate([25, 75]): 
    for j, alpha in enumerate(alphas):
        ll = [r['output'].logLt for r in results if r['ESSrmin']==alpha]
        quartiles[i, j] = np.percentile(np.array(ll), q)
plt.fill_between([a*N for a in alphas], quartiles[0], quartiles[1],
                 facecolor='darkgray', alpha=0.8)
plt.xlabel(r'ESS$_\min$')
plt.ylabel(r'log-lik')
plt.axhline(y=true_loglik, ls=':', color='k')  # true value 
if savefigs:
    plt.savefig('impact_threshold_in_adaptrs.pdf')

plt.show()
