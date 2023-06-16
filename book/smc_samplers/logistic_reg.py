#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numerical experiment of Chapter 17 (SMC samplers).

Compare IBIS and SMC tempering for approximating:

* the normalising constant (marginal likelihood)
* the posterior expectation of the p coefficients

for a logistic regression model.

See below for how to select the data-set.

Note: the SMC samplers implemented in module smc_samplers are now "waste-free"
by default, see Dau & Chopin (2021), and the documentation of `smc_samplers`
(plus the corresponding jupyter notebook). This script still performs exactly
the same numerical experiments as in the book, based on standard (non
waste-free) SMC samplers. To do so, we added ``wastefree=False`` to the
definition of the corresponding `Feynman-Kac` object. Again, see the
documentation of `smc_samplers` for more details.

"""

from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import seaborn as sb

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import smc_samplers as ssps
from particles.collectors import Moments

datasets = {'pima': dts.Pima, 'eeg': dts.Eeg, 'sonar': dts.Sonar}
dataset_name = 'pima'  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape

# for each dataset, we adapt:
# * N: number of particles
# * Ks = list of Ks (nr MCMC steps)
# * typK: value of M used for plots on "typical" run

if dataset_name == 'sonar':
    N = 10 ** 4
    Ks = [10, 20, 30, 40, 50, 60]
    typK = 50
elif dataset_name == 'pima':
    N = 10 ** 3
    Ks = [1, 3, 5]
    typK = 3
elif dataset_name == 'eeg':
    N = 10 ** 3
    Ks = [1, 3, 5, 7, 10, 15, 20]
    typK = 5

# prior & model
prior = dists.StructDist({'beta':dists.MvNormal(scale=5.,
                                                cov=np.eye(p))})

class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], data[t, :])
        return - np.logaddexp(0., -lin)

# algorithms
# N and values of K set above according to dataset
ESSrmin = 0.5
nruns = 16
results = []

# runs
print('Dataset: %s' % dataset_name)
for K in Ks:
    for i in range(nruns):
        # need to shuffle the data for IBIS
        random.shuffle(data)
        model = LogisticRegression(data=data, prior=prior)
        for alg_type in ['tempering', 'ibis']:
            if alg_type=='ibis':
                fk = ssps.IBIS(model=model, wastefree=False, len_chain=K + 1)
                pf = particles.SMC(N=N, fk=fk, ESSrmin=ESSrmin,
                                collect=[Moments], verbose=False)
            else:
                fk = ssps.AdaptiveTempering(model=model, ESSrmin=ESSrmin,
                                            wastefree=False, len_chain = K + 1)
                pf = particles.SMC(N=N, fk=fk, ESSrmin=1., collect=[Moments],
                                verbose=True)
                # must resample at every time step when doing adaptive
                # tempering
            print('%s, K=%i, run %i' % (alg_type, K, i))
            pf.run()
            print('CPU time (min): %.2f' % (pf.cpu_time / 60))
            print('loglik: %f' % pf.logLt)
            res = {'K': K, 'type': alg_type, 'out': pf.summaries,
                   'cpu': pf.cpu_time}
            if alg_type=='ibis':
                n_eval = N * (T + K * sum([t for t in range(T) if
                                           pf.summaries.rs_flags[t]]))
            else:
                n_eval = N * T * (1. + K * (len(pf.summaries.ESSs) - 1))
                res['path_sampling'] = pf.X.shared['path_sampling'][-1]
                res['exponents'] = pf.X.shared['exponents']
            res['n_eval'] = n_eval
            results.append(res)

# save results
##############
# import pickle
# pickle.dump({'results':results, 'Ks': Ks, 'N': N, 'nruns': nruns, 'T': T, 'p':p,
#              'ESSrmin' : ESSrmin, 'data': data},
#             open('%s_N%i.pickle' % (dataset, N), 'wb'))
# Then reload like this
# dp = pickle.load(file_name)
# locals().update(dp)

# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use('ggplot')
pal = sb.dark_palette('white', n_colors=2)

# Compare standard and path sampling estimates of the log-normalising cst
plt.figure()
diff_est = [(r['out'].logLts[-1] - r['path_sampling'])
            for r in results if r['type']=='tempering']
sb.histplot(diff_est)

# Figure 17.1: typical behaviour of IBIS
typ_ibis = [r for r in results if r['type']=='ibis' and r['K'] == typK][0]
typ_ess = typ_ibis['out'].ESSs
typ_rs_times = np.nonzero(typ_ibis['out'].rs_flags)[0]

# Left panel: evolution of ESS
fig, ax = plt.subplots()
ax.plot(typ_ess, 'k')
ax.set(xlabel=r'$t$', ylabel='ESS')
if savefigs:
    plt.savefig(dataset_name + '_typical_ibis_ess.pdf')

# Right panel: evolution of resampling times
fig, ax = plt.subplots()
ax.plot(typ_rs_times[:-1], np.diff(typ_rs_times), 'ko-')
ax.set(xlabel=r'$t$', ylabel='duration between successive rs')
if savefigs:
    plt.savefig(dataset_name + '_typical_ibis_rs_times.pdf')

# Figure 17.2: evolution of temperature in a typical tempering run
typ_temp = [r for r in results if r['type']=='tempering' and r['K'] == typK][0]
expnts = typ_temp['exponents']
plt.figure()
plt.plot(expnts, 'k')
plt.xlabel(r'$t$')
plt.ylabel('tempering exponent')
if savefigs:
    plt.savefig(dataset_name + '_typical_tempering_temperatures.pdf')

# nr evals vs K for both algorithms
plt.figure()
sb.boxplot(x=[r['K'] for r in results],
           y=[r['n_eval'] for r in results],
           hue=[r['type'] for r in results],
           palette=pal)
plt.xlabel('number MCMC steps')
plt.ylabel('number likelihood evaluations')
if savefigs:
    plt.savefig(dataset_name + '_boxplots_nevals_vs_K.pdf')

# Figure 17.3: Box-plots estimate versus number of MCMC steps
# Left panel: marginal likelihood
plt.figure()
sb.boxplot(x=[r['K'] for r in results],
           y=[r['out'].logLts[-1] for r in results],
           hue=[r['type'] for r in results],
           palette=pal)
plt.xlabel('number MCMC steps')
plt.ylabel('marginal likelihood')
if savefigs:
    plt.savefig(dataset_name + '_boxplots_marglik_vs_K.pdf')

# Right panel: post expectation 1st pred
plt.figure()
sb.boxplot(x=[r['K'] for r in results],
           y=[r['out'].moments[-1]['mean']['beta'][1] for r in results],
           hue=[r['type'] for r in results],
           palette=pal)
plt.xlabel('number MCMC steps')
plt.ylabel('posterior expectation first predictor')
if savefigs:
    plt.savefig(dataset_name + '_boxplots_postexp1_vs_K.pdf')

# Figure 17.4: variance vs CPU trade-off
# variance times K, as a function of K
plt.figure()
cols = {'ibis': 'gray', 'tempering':'black'}
lsts = {'ibis': '--', 'tempering': '-'}
for i in range(p):
    for alg_type in ['ibis', 'tempering']:
        adj_var = []
        for K in Ks:
            mts = [r['out'].moments[-1]
                   for r in results if r['K']==K and r['type']==alg_type]
            av = (K * np.var([m['mean']['beta'][i] for m in mts]) /
                             np.mean([m['var']['beta'][i] for m in mts]))
            adj_var.append(av)
        if i==0:
            plt.plot(Ks, adj_var, color=cols[alg_type], label=alg_type,
                     alpha=.8, linewidth=2, linestyle=lsts[alg_type])
        else:
            plt.plot(Ks, adj_var, color=cols[alg_type], alpha=.8, linewidth=2,
                     linestyle=lsts[alg_type])
plt.legend()
plt.xticks(Ks, ['%i' % K for K in Ks])  # force int ticks
plt.xlabel('number MCMC steps')
plt.ylabel(r'variance times number MCMC steps')
if savefigs:
    plt.savefig(dataset_name + '_postexp_var_vs_K.pdf')
