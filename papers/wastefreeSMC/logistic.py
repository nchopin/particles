#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
First numerical experiment in Waste-Free SMC paper (TODO)

Compare IBIS and SMC tempering for approximating:

* the normalising constant (marginal likelihood)
* the posterior expectation of the p coefficients

for a logistic regression model.

See below for how to select the dataset.
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
dataset_name = 'pima'  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape

# Standard SMC: N is number of particles, K is number of MCMC steps
# Waste-free SMC: M is number of resampled particles, P is length of MCMC
# chains (same notations as in the paper)
# All of the runs are such that N*K or M*P equal 10^5

if dataset_name == 'sonar':
    MPs = [(50, 2_000), (200, 500), (500, 200)]
    NKs = [(10_000, 10), (2_500, 40), (1_000, 100)]
elif dataset_name == 'pima':
    MPs = [(20, 500), (50, 200), (100, 100)]
    NKs = [(10_000, 1), (2000, 5), (500, 20)]
elif dataset_name == 'eeg':
    N = 10 ** 3
    Ms = [1, 3, 5, 7, 10, 15, 20]
    # TODO

# prior & model
prior = dists.StructDist({'beta':dists.MvNormal(scale=5.,
                                                cov=np.eye(p))})

class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], data[t, :])
        return - np.logaddexp(0., -lin)

# algorithms
# N and values of M set above according to dataset
nruns = 20  # TODO
results = []

# runs
print('Dataset: %s' % dataset_name)
for mp, nk in zip(MPs, NKs):
    for i in range(nruns):
        # need to shuffle the data for IBIS
        random.shuffle(data)
        model = LogisticRegression(data=data, prior=prior)
        for alg_type in ['tempering', 'ibis']:
            for waste in [True, False]:
                if waste:
                    M, P = mp
                    res = {'M': M, 'P': P}
                    N = M
                    nsteps = P -1
                else:
                    N, K = nk
                    res = {'N': N, 'K': K}
                if alg_type == 'ibis':
                    fk = ssps.IBIS(model=model, nsteps=nsteps, wastefree=waste)
                else:
                    fk = ssps.AdaptiveTempering(model=model, nsteps=nsteps, 
                                                wastefree=waste)
                pf = particles.SMC(fk=fk, N=N, collect=[Moments], verbose=True)
                print('%s, waste:%i, nsteps=%i, run %i' % (alg_type, waste,
                                                           nsteps, i))
                pf.run()
                print('CPU time (min): %.2f' % (pf.cpu_time / 60))
                print('loglik: %f' % pf.logLt)
                res.update({'type': alg_type, 
                            'out': pf.summaries,
                            'waste': waste,
                            'cpu': pf.cpu_time})
            # TODO M > N
            # if alg_type=='ibis':
            #     n_eval = N * (T + M * sum([t for t in range(T) if
            #                                pf.summaries.rs_flags[t]]))
            # else:
            #     n_eval = N * T * (1. + M * (len(pf.summaries.ESSs) - 1))
            #     res['path_sampling'] = pf.X.shared['path_sampling'][-1]
            #     res['exponents'] = pf.X.shared['exponents']
            # res['n_eval'] = n_eval
            results.append(res)

# save results
##############
# import pickle
# pickle.dump({'results':results, 'Ms': Ms, 'N': N, 'nruns': nruns, 'T': T, 'p':p,
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
# plt.figure()
# diff_est = [(r['out'].logLts[-1] - r['path_sampling'])
#             for r in results if r['type']=='waste-temp']
# sb.distplot(diff_est)

# # Figure 17.1: typical behaviour of IBIS
# typ_ibis = [r for r in results if r['type']=='ibis' and r['M'] == typM][0]
# typ_ess = typ_ibis['out'].ESSs
# typ_rs_times = np.nonzero(typ_ibis['out'].rs_flags)[0]

# # Left panel: evolution of ESS
# fig, ax = plt.subplots()
# ax.plot(typ_ess, 'k')
# ax.set(xlabel=r'$t$', ylabel='ESS')
# if savefigs:
#     plt.savefig(dataset_name + '_typical_ibis_ess.pdf')

# # Right panel: evolution of resampling times
# fig, ax = plt.subplots()
# ax.plot(typ_rs_times[:-1], np.diff(typ_rs_times), 'ko-')
# ax.set(xlabel=r'$t$', ylabel='duration between successive rs')
# if savefigs:
#     plt.savefig(dataset_name + '_typical_ibis_rs_times.pdf')

# # Figure 17.2: evolution of temperature in a typical tempering run
# typ_temp = [r for r in results if r['type']=='tempering' and r['M'] == typM][0]
# expnts = typ_temp['exponents']
# plt.figure()
# plt.plot(expnts, 'k')
# plt.xlabel(r'$t$')
# plt.ylabel('tempering exponent')
# if savefigs:
#     plt.savefig(dataset_name + '_typical_tempering_temperatures.pdf')

# # nr evals vs M for both algorithms
# plt.figure()
# sb.boxplot(x=[r['M'] for r in results],
#            y=[r['n_eval'] for r in results],
#            hue=[r['type'] for r in results],
#            palette=pal)
# plt.xlabel('number MCMC steps')
# plt.ylabel('number likelihood evaluations')
# if savefigs:
#     plt.savefig(dataset_name + '_boxplots_nevals_vs_M.pdf')

# Figure 17.3: Box-plots estimate versus number of MCMC steps
# Left panel: marginal likelihood

titles = ['standard SMC', 'waste-free SMC']
plots = {'log marginal likelihood': lambda rout: rout.logLts[-1],
         'post expectation first pred': 
         lambda rout: rout.moments[-1]['mean']['beta'][1]
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
                   hue=[r['type'] for r in rez],
                   palette=pal, ax=ax)
        ax.set(xlabel=xlab, title=title, ylabel=ylab)
        fig.tight_layout()
    if savefigs:
        fig.savefig('%s_boxplots_%s.pdf' % (dataset_name, plot))

