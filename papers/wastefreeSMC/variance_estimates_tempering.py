#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Single-run variance estimates in waste-free tempering. 

TODO

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
import seaborn as sb

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import smc_samplers as ssps

datasets = {'pima': dts.Pima, 'sonar': dts.Sonar}
dataset_name = 'pima'
data = datasets[dataset_name]().data
T, p = data.shape

# Standard SMC: N is number of particles, K is number of MCMC steps
# Waste-free SMC: M is number of resampled particles, P is length of MCMC
# chains (same notations as in the paper)
# All of the runs are such that N*K or M*P equal N0

if dataset_name == 'pima':
    nruns = 20_000
    N0 = 10**4
    M = 25
elif dataset_name == 'sonar':
    nruns = 300
    N0 = 2 * 10 ** 5
    M = 200

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
def phi(x):
    return x.theta['beta'][:, 0]  # intercept
results = []

# runs
print('Dataset: %s' % dataset_name)
N, lc = M, N0 // M

fk = ssps.AdaptiveTempering(model=model, len_chain=lc)
results = particles.multiSMC(fk=fk, N=N, nruns=nruns, 
                             collect=[ssps.Var_logLt, ssps.Var_phi(phi=phi)])

# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use('ggplot')
pal = sb.dark_palette('white', n_colors=2)

plt.figure()
varest = np.array([r['output'].summaries.var_logLt[-1] for r in results]) / N0
sb.displot(varest)
var_emp = np.var([r['output'].summaries.logLts[-1] for r in results])
plt.axvline(var_emp, 0., 1., color='black')
plt.xlabel(r'single-run variance estimate')
plt.savefig(f'hist_varest_logLT_{dataset_name}.pdf')

