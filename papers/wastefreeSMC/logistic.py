#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reproduces first numerical experiment of the paper.

Compare standard and waste-free tempering SMC for a logistic 
posterior. Plot the boxplots (over 100 runs) of the following
estimates: 

* log normalising constant (marginal likelihood)
* posterior expectation of the average of the p coefficients

Considered dataset: sonar (but see below for other options).

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

datasets = {'pima': dts.Pima, 'eeg': dts.Eeg, 'sonar': dts.Sonar}
dataset_name = 'sonar'  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape

# Standard SMC: N is number of particles, K is number of MCMC steps
# Waste-free SMC: M is number of resampled particles, P is length of MCMC
# chains (same notations as in the paper)
# All of the runs are such that N*K or M*P equal N0

if dataset_name == 'sonar':
    alg_type = 'tempering'
    N0 = 2 * 10**5
    Ks = [5, 20, 100, 500, 1000]
    Ms = [50, 100, 200, 400, 800]
elif dataset_name == 'pima':
    alg_type = 'ibis'
    N0 = 10**4
    Ks = [1, 4, 16]
    Ms = [25, 100, 400]
elif dataset_name == 'eeg':
    alg_type = 'ibis'
    N0 = 10 ** 4
    Ks = [1, 4, 16]
    Ms = [25, 100, 400]

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

nruns = 100
results = []

# runs
print('Dataset: %s' % dataset_name)
for M, K in zip(Ms, Ks):
    for i in range(nruns):
        # need to shuffle the data for IBIS
        random.shuffle(data)
        model = LogisticRegression(data=data, prior=prior)
        for waste in [True, False]:
            if waste:
                P = N0 // M
                N, nsteps = M, P - 1
                res = {'M': M, 'P': P}
            else:
                N = N0 // K
                nsteps = K
                res = {'N': N, 'K': K}
            if alg_type == 'ibis':
                fk = ssps.IBIS(model=model, nsteps=nsteps, wastefree=waste)
            else:
                fk = ssps.AdaptiveTempering(model=model, nsteps=nsteps, 
                                            wastefree=waste)
            pf = particles.SMC(fk=fk, N=N, collect=[Moments], verbose=False)
            print('%s, waste:%i, nsteps=%i, run %i' % (alg_type, waste,
                                                       nsteps, i))
            pf.run()
            print('CPU time (min): %.2f' % (pf.cpu_time / 60))
            print('loglik: %f' % pf.logLt)
            res.update({'type': alg_type, 
                        'out': pf.summaries,
                        'waste': waste,
                        'cpu': pf.cpu_time})
            results.append(res)


# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use('ggplot')
pal = sb.dark_palette('white', n_colors=2)


titles = ['standard SMC', 'waste-free SMC']
plots = {'log marginal likelihood': lambda rout: rout.logLts[-1],
         'post expectation average pred': 
         lambda rout: np.mean(rout.moments[-1]['mean']['beta'])
        }

for plot, func in plots.items():
    fig, axs = plt.subplots(1, 2, sharey=True)
    for title, ax in zip(titles, axs):
        if title == 'waste-free SMC':
            rez = [r for r in results if r['waste']]
            xlab = 'M'
            ylab = ''
        else:
            rez = [r for r in results if not r['waste']]
            xlab = 'K'
            ylab = plot
        sb.boxplot(x=[r[xlab] for r in rez],
                   y=[func(r['out']) for r in rez],
                   hue=[r['waste'] for r in rez],
                   palette=pal, ax=ax)
        ax.set(xlabel=xlab, title=title, ylabel=ylab)
        fig.tight_layout()
    if savefigs:
        fig.savefig('%s_boxplots_%s.pdf' % (dataset_name, plot))

