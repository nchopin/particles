#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare bootstrap, guided and auxiliary particle filters on a linear Gaussian
model.  Results are compared to true values computed by a Kalman filter. 

This generates three of the plots of Chapter 10 on particle filtering (Figure
10.2). 

"""

from __future__ import division, print_function

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

import particles
from particles import kalman
from particles import state_space_models as ssms 

# Define ssm, simulate data 
T = 100
my_ssm = kalman.LinearGauss(sigmaX=1., sigmaY=.2, rho=.9)
true_states, data = my_ssm.simulate(T)

# FK models
models = OrderedDict()
models['bootstrap'] = ssms.Bootstrap(ssm=my_ssm, data=data)
models['guided'] = ssms.GuidedPF(ssm=my_ssm, data=data)
models['APF'] = ssms.AuxiliaryPF(ssm=my_ssm, data=data)
# Uncomment line below if you want to include the "Boostrap APF"
# (APF with proposal set to dist. of X_t|X_{t-1}) in the comparison
#models['bootAPF'] = ssm.AuxiliaryBootstrap(ssm=my_ssm, data=data)

# Compute "truth" 
kf = kalman.Kalman(ssm=my_ssm, data=data)
kf.filter()
true_loglik = np.cumsum(kf.logpyt)
true_filt_means = [f.mean for f in kf.filt]

# Get results 
N = 10**3
results = particles.multiSMC(fk=models, N=N, nruns=1000, moments=True)

# PLOTS
# =====
plt.style.use('ggplot')
savefigs = True  # False if you don't want to save plots as pdfs

# black and white 
sb.set_palette(sb.dark_palette("lightgray", n_colors=len(models), 
                               reverse=True)) 

# box-plots for log-likelihood evaluation
plt.figure()
sb.boxplot(x=[r['fk'] for r in results],
           y=[r['output'].logLt for r in results]
          )
plt.ylabel('log-likelihood estimate')
plt.axhline(y=true_loglik[-1], ls=':', color='k')
# adapt line below if you want to zoom (if boxplots for guided/APF are too tiny)
#v = plt.axis(); plt.axis([v[0],v[1],-152.5,-150.5]) 
if savefigs:
    plt.savefig('lingauss_boxplot_loglik_boot_guided_APF.pdf') 

# MSE of particle estimate of E(X_t|Y_{0:t}) vs time 
plt.figure()
for mod in models.keys():
    errors = np.array( [ [mom['mean']-truemean for mom, truemean in 
                           zip(r['output'].summaries.moments, true_filt_means)] 
                                for r in results if r['fk']==mod] ).squeeze()
    plt.plot(np.sqrt(np.mean(errors**2, axis=0)), label=mod, linewidth=2)
plt.xlabel(r'$t$')
plt.ylabel(r'MSE$^{1/2}$ of estimate of $E(X_t|Y_{0:t})$')
plt.legend()
if savefigs:
    plt.savefig('lingauss_std_filtexpectation_boot_guided_APF.pdf') 

# ESS vs time (from a single run)
plt.figure()
for model in models.keys():
    pf = next(r['output'] for r in results if r['fk']==model)
    plt.plot(pf.summaries.ESSs,label=model, linewidth=2)
plt.legend(loc=(0.02,0.3))
plt.axis([0,T,0,N])
plt.xlabel(r'$t$')
plt.ylabel('ESS')
if savefigs:
    plt.savefig('lingauss_ESSvst_boot_guided_APF.pdf') 

plt.show()
