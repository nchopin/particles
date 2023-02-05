#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Single-run variance estimates in waste-free IBIS. 

In IBIS, you approximate at time t the posterior distribution based on
data y_{0:t}. With waste-free IBIS, it then becomes possible to estimate the
(Monte Carlo) variance of estimates computed at each time t, such as the
posterior expectation of a given parameter (again given the data up to time t),
or the current log-evidence (log of marginal likelihood). Module smc_samplers
now implements collectors (see module `collectors`) that collect such variance
estimates. 

This script gives an example on how these collectors may be used. It is based
on the same example as in Chapter 17 of the book: logistic regression, Gaussian
prior for the coefficients, pima or EEG datasets. 

Reference
=========

Dau, H.D. and Chopin, N. (2022). Waste-free Sequential Monte Carlo,
  Journal of the Royal Statistical Society: Series B (Statistical Methodology),
  vol. 84, p. 114-148. <https://doi.org/10.1111/rssb.12475>, see also on arxiv: 
  <https://arxiv.org/abs/2011.02328>. 

"""

from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import seaborn as sb

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssps
from particles.collectors import Moments

datasets = {'pima': dts.Pima, 'eeg': dts.Eeg}
dataset_name = 'pima'
data = datasets[dataset_name]().data
T, p = data.shape

# Standard SMC: N is number of particles, K is number of MCMC steps
# Waste-free SMC: M is number of resampled particles, P is length of MCMC
# chains (same notations as in the paper)
# All of the runs are such that N*K or M*P equal N0

if dataset_name == 'pima':
    nruns = 300
    N0 = 10**4
    M = 25
elif dataset_name == 'eeg':
    nruns = 50
    N0 = 10 ** 4
    M = 25

# prior & model
scales = 5. * np.ones(p)
scales[0] = 20.  # intercept has a larger scale
prior = dists.StructDist({'beta':dists.MvNormal(scale=scales,
                                                cov=np.eye(p))})

class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], data[t, :])
        return - np.logaddexp(0., -lin)

model = LogisticRegression(data=data, prior=prior)
phi = lambda x: x.theta['beta'][:, 0]  # intercept
results = []

# runs
print('Dataset: %s' % dataset_name)
N, lc = M, N0 // M
res = {'M': M, 'P': lc}

fk = ssps.IBIS(model=model, len_chain=lc)

results = particles.multiSMC(fk=fk, N=N, nruns=nruns,
                             collect=[Moments, ssps.Var_logLt, 
                                      ssps.Var_phi(phi=phi)])

# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use('ggplot')
pal = sb.dark_palette('white', n_colors=2)

plt.figure()
r0 = results[0]['output'].summaries
plt.plot(r0.logLts)
plt.ylabel(r'log-likelihood')
plt.xlabel(r'$t$')

plt.figure()
varest = np.array([r['output'].summaries.var_logLt for r in results]) / N0
lower = np.percentile(varest, 5, axis=0)
upper = np.percentile(varest, 95, axis=0)
label_fill = f'single-run var estimates (5-95% quantiles)'
plt.fill_between(np.arange(varest.shape[1]), lower, upper, alpha=0.8,
                 color='gray', label=label_fill)
plt.plot(np.var([r['output'].summaries.logLts for r in results], axis=0),
         label=f'empirical variance over the {nruns} runs')
plt.legend(loc='lower right')
plt.ylabel(r'var log-likelihood')
plt.xlabel(r'$t$')
if savefigs:
    plt.savefig(f'var_logLt_{dataset_name}.pdf')

plt.figure()
est_int = np.array([[m['mean']['beta'][0] for m in r['output'].summaries.moments]
                    for r in results])
plt.plot(est_int[0, :])
plt.ylabel(r'intercept post expectation')
plt.xlabel(r'$t$')

fig = plt.figure()
varest_int = np.array([r['output'].summaries.var_phi for r in results]) / N0
lower = np.percentile(varest_int, 5, axis=0)
upper = np.percentile(varest_int, 95, axis=0)
label_fill = f'single-run variance estimates (5-95% quantiles)'
plt.fill_between(np.arange(est_int.shape[1]), lower, upper, alpha=0.8,
                 color='gray', label=label_fill)
plt.plot(np.var(est_int, axis=0), 
         label=f'empirical variance over the {nruns} runs')
plt.legend(loc='upper right')
plt.ylabel(r'var intercept post expectation')
plt.xlabel(r'$t$')
max_when_cropped = upper[50:].max()
plt.ylim(top=max_when_cropped, bottom=-max_when_cropped / 10)
if savefigs:
    plt.savefig(f'var_post_{dataset_name}.pdf')

