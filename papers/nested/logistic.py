#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

TODO 

Reference
=========

Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." arXiv
preprint arXiv:2011.02328 (2020).  

"""

from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas
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

nruns = 200
results = []
model = LogisticRegression(data=data, prior=prior)

lc = 100
N = 1_000
rhoess = [.1, .3, .5, .7, .9]

def out_func(pf):
    try:
        est = pf.X.shared['log_evid'][-1]
    except: # not an NS SMC algorithm
        est = pf.logLt
    return {'nevals': N * ((lc - 1) * pf.t + 1) , 'est': est}

results = []
algs = ['nested', 'tempering']
for re in rhoess:
    fks = {'nested': ssps.NestedSampling(model=model, len_chain=lc, rho=re),
           'tempering': ssps.AdaptiveTempering(model=model, len_chain=lc,
                                               ESSrmin=re)}
    res = particles.multiSMC(fk=fks, N=N, verbose=True, nruns=nruns,
                             out_func=out_func)
    for r in res:
        r['rhoess'] = re
    results += res

grand_mean = np.mean([r['est'] for r in results])
for r in results:
    r['mse'] = (r['est'] - grand_mean)**2

df = pandas.DataFrame(results)
dfm = df.groupby(['fk', 'rhoess']).mean()  # variance as a function of fk and N
dfm = dfm.reset_index()

# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use('ggplot')
pal = sb.dark_palette('white', n_colors=2)
colors = {'nested': 'red',
          'tempering': 'green'}
plt.figure()
for alg in algs:
    dfma = dfm[dfm['fk'] == alg]
    plt.plot(dfma['nevals'], dfma['mse'], color=colors[alg], label=alg)
    plt.xlabel(r'nr evals')
    plt.xscale('log')
    plt.ylabel('MSE')
    plt.yscale('log')
plt.legend()
plt.savefig('nested_vs_tempering_mse_vs_evals.pdf')

plt.figure()
for alg in algs:
    dfma = dfm[dfm['fk'] == alg]
    plt.plot(dfma['rhoess'], dfma['nevals'] * dfma['mse'], color=colors[alg], label=alg)
    plt.xlabel(r'$\rho$')
    plt.ylabel('work-normalised MSE')
    plt.yscale('log')
plt.legend()
plt.savefig('nested_vs_tempering_work_normalised.pdf')

#titles = ['standard SMC', 'waste-free SMC']
#plots = {'log marginal likelihood': lambda rout: rout.logLts[-1],
#         'post expectation average pred': 
#         lambda rout: np.mean(rout.moments[-1]['mean']['beta'])
#        }

#for plot, func in plots.items():
#    fig, axs = plt.subplots(1, 2, sharey=True)
#    for title, ax in zip(titles, axs):
#        if title == 'waste-free SMC':
#            rez = [r for r in results if r['waste']]
#            xlab = 'M'
#            ylab = ''
#        else:
#            rez = [r for r in results if not r['waste']]
#            xlab = 'K'
#            ylab = plot
#        sb.boxplot(x=[r[xlab] for r in rez],
#                   y=[func(r['out']) for r in rez],
#                   hue=[r['waste'] for r in rez],
#                   palette=pal, ax=ax)
#        ax.set(xlabel=xlab, title=title, ylabel=ylab)
#        fig.tight_layout()
#    if savefigs:
#        fig.savefig('%s_boxplots_%s.pdf' % (dataset_name, plot))

