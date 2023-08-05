#!/usr/bin/env python

"""
Numerical example of Chapter 18 on SMC^2.

Runs several instances of the SMC^2 algorithm; the likelihood may be
approximated:
    * either by a standard bootstrap filter
    * or its SQMC version

The considered state-space model is a stochastic volatility model
with leverage.

Notes: for N=10^3, a single run takes about 2 hrs; since runs are executed
in parallel, the total CPU time depends on the number of cores.
(With 50+ cores, it should take 2 hrs as well.)
"""


import warnings
warnings.filterwarnings("ignore")  # disable silly matplotlib warnings

from matplotlib import pyplot as plt
import numpy as np
from numpy import random

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssp
from particles import state_space_models
from particles.collectors import Moments

# data
data = dts.GBP_vs_USD_9798().data

# prior
dictprior = {'mu':dists.Normal(scale=2.),
             'sigma':dists.Gamma(a=2., b=2.),
             'rho': dists.Beta(a=9., b=1.),
             'phi': dists.Uniform(a=-1., b=1.)
            }
prior = dists.StructDist(dictprior)

# moment function
def qtiles(W, x):
    alphas = np.linspace(0.05, 0.95, 19)
    return rs.wquantiles_str_array(W, x.theta, alphas=alphas)

# algorithms
N = 10 ** 3  # re-run with N= 10^4 for the second CDF plots
fks = {}
fk_opts = {'ssm_cls': state_space_models.StochVolLeverage, 'prior': prior,
           'data': data, 'init_Nx': 100, 'smc_options': {'qmc': False},
           'ar_to_increase_Nx': 0.1, 'wastefree': False, 'len_chain': 6}
fks['smc2'] = ssp.SMC2(**fk_opts)
fk_opts['smc_options']['qmc'] = True
fks['smc2_qmc'] = ssp.SMC2(**fk_opts)
fk_opts['ssm_cls'] = state_space_models.StochVol
fks['smc2_sv'] = ssp.SMC2(**fk_opts)

# runs
runs = particles.multiSMC(fk=fks, N=N, collect=[Moments(mom_func=qtiles)],
                          verbose=True, nprocs=0, nruns=25)

# plots
#######
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')

colors = {'smc2': 'gray', 'smc2_qmc': 'black'}
lsts = {'smc2': '--', 'smc2_qmc': '-'}
prefix = 'smc2_sv_lvg_N%i' % N
T = data.shape[0]

# Figure 18.2: variance marginal likelihood vs time
plt.figure()
for k, c in colors.items():
    plt.plot(np.var(np.array([r['output'].summaries.logLts
                                   for r in runs if r['fk']==k]), axis=0),
             color=c, label=k, linewidth=2, linestyle=lsts[k])
plt.xlabel(r'$t$')
plt.ylabel('var log-lik')
plt.legend()
if savefigs:
    plt.savefig('%s_var_loglik.pdf' % prefix)

# Figure 18.3: model evidence leverage vs basic SV model
plt.figure()
evidence_lvg = np.mean([r['output'].summaries.logLts
                        for r in runs if r['fk']=='smc2_qmc'], axis=0)
evidence_sv = np.mean([r['output'].summaries.logLts
                        for r in runs if r['fk']=='smc2_sv'], axis=0)
plt.plot(evidence_lvg - evidence_sv, 'k')
plt.xlabel(r'$t$')
plt.ylabel('diff marginal likelihoods')
if savefigs:
    plt.savefig('%s_model_comparison.pdf' % prefix)

# Figure 18.4: sequential learning of parameters
typical_run = [r for r in runs if r['fk']=='smc2_qmc'][0]['output']
plt.figure()
for i, p in enumerate(prior.laws.keys()):
    plt.subplot(2, 2, i + 1)
    q25, q50, q75 = ([typical_run.summaries.moments[t][p][j] for t in range(T)]
                for j in [5, 10, 15])
    plt.fill_between(range(T), q25, q75, color='gray')
    plt.plot(range(T), q50, 'k')
    plt.title(r'$\%s$' % p)
    plt.xlabel(r'$t$')
plt.tight_layout()
if savefigs:
    plt.savefig('%s_seq_inference.pdf' % prefix)

# Figure 18.1 (left panel): ESS vs time for a typical run
plt.figure()
typ_run = runs[0]
print(typ_run['fk'])
plt.plot(typ_run['output'].summaries.ESSs, 'k')
plt.xlabel(r'$t$')
plt.ylabel('ESS')
if savefigs:
    plt.savefig('%s_ESS.pdf' % prefix)

# Figure 18.5 (and 18.6 with N=10^4): marginal CDFs
def cdf(x, w):
    a = np.argsort(x)
    cw = np.cumsum(w[a])
    return x[a], cw

include_prior = False

smc2s = [r['output'] for r in runs if r['fk']=='smc2']
smc2qmcs = [r['output'] for r in runs if r['fk']=='smc2_qmc']

for i, p in enumerate(['mu', 'rho', 'sigma', 'phi']):
    plt.subplot(2, 2, i + 1)
    for r in smc2qmcs:
        xx, yy = cdf(r.X.theta[p], r.W)
        plt.plot(xx, yy, color='black', alpha=0.2)
        if include_prior:
            xx = np.linspace(xx.min(), xx.max(), 100)
            plt.plot(xx, prior.laws[p].cdf(xx), ':')
    plt.xlabel(r'$\%s$' % p)
    typ_run = smc2qmcs[0]
    m = np.average(typ_run.X.theta[p], weights=typ_run.W)
    s = np.std(typ_run.X.theta[p])
    plt.xlim(m - 2.5 * s, m + 2.5 * s)
plt.tight_layout()
if savefigs:
    plt.savefig('%s_cdfs.pdf' % prefix)

# Figure 18.1 (right panel): Nx vs time
plt.figure()
for r in smc2s:
    jitter = 5 * random.randn()
    plt.plot(np.array(r.X.shared['Nxs']) + jitter, 'k', alpha=0.5)
plt.ylim(bottom=0)
plt.xlabel(r'$t$')
plt.ylabel(r'$N_x$ + jitter')
if savefigs:
    plt.savefig('%s_Nx_vs_t.pdf' % prefix)
