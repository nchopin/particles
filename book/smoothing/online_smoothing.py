#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Illustrates the different on-line smoothing algorithms using the
bootstrap filter of the following model:

X_t|X_{t-1}=x_{t-1} ~ N(mu+phi(x_{t-1}-mu),sigma^2)
Y_t|X_t=x_t ~ Poisson(exp(x_t))
as in first example in Chopin and Singh (2014, Bernoulli)

More precisely, we compare different smoothing algorithms for approximating
the smoothing expectation of additive function phit, defined as
phi_t(x_0:t) = sum_{s=0}^t psi_s(x_{s-1},x_s)
see below for a definition of psi_s.

See Chapter 12 (smoothing) for more details; in particular Figures 12.2 and
12.3 which were produced by this script.

"""

from __future__ import division, print_function

from functools import partial

import numpy as np
from matplotlib import pyplot as plt

import particles
from particles import collectors as cols
from particles import state_space_models


def psit(t, xp, x, mu, phi, sigma):
    """ score of the model (gradient of log-likelihood at theta=theta_0)
    """
    if t == 0:
        return -0.5 / sigma ** 2 + \
               (0.5 * (1. - phi ** 2) / sigma ** 4) * (x - mu) ** 2
    else:
        return -0.5 / sigma ** 2 + (0.5 / sigma ** 4) * \
               ((x - mu) - phi * (xp - mu)) ** 2


class DiscreteCox_with_addf(state_space_models.DiscreteCox):
    """ A discrete Cox model:
    Y_t ~ Poisson(e^{X_t})
    X_t - mu = phi(X_{t-1}-mu)+U_t,   U_t ~ N(0,1)
    X_0 ~ N(mu,sigma^2/(1-phi**2))
    """

    def upper_bound_log_pt(self, t):
        return -0.5 * np.log(2 * np.pi) - np.log(self.sigma)

    def add_func(self, t, xp, x):
        return psit(t, xp, x, self.mu, self.phi, self.sigma)


def outf(pf, method):
    return {'result': getattr(pf.summaries, method),
            'cpu': pf.cpu_time}


# set up models, simulate data
nruns = 25  # how many runs for each algorithm
T = 10 ** 4  #  sample size
mu0 = 0.  # true parameters
phi0 = 0.9
sigma0 = .5

ssm = DiscreteCox_with_addf(mu=mu0, phi=phi0, sigma=sigma0)
true_states, data = ssm.simulate(T)
fkmod = state_space_models.Bootstrap(ssm=ssm, data=data)

# plot data
plt.figure()
plt.plot(data)
plt.title('data')

methods = ['ON2', 'naive']  # in that order: ON2 must be run first
collectors = {'ON2': cols.Online_smooth_ON2(),
              'naive': cols.Online_smooth_naive()}
long_names = {'ON2': r'$O(N^2)$ forward-only',
              'naive': r'naive, $O(N)$ forward-only'}
runs = {}
avg_cpu = {}
Ns = {'ON2': 100, 'naive': 10 ** 4}  #  for naive N is rescaled later

for method in methods:
    col = collectors[method]
    if method == 'naive':
        # rescale N to match CPU time
        pf = particles.SMC(fk=fkmod, N=Ns['naive'], collect=[col])
        pf.run()
        Ns['naive'] = int(Ns['naive'] * avg_cpu['ON2'] / pf.cpu_time)
        print('rescaling N to %i to match CPU time' % Ns['naive'])
    long_names[method] += r', N=%i' % Ns[method]
    print(long_names[method])

    runs[method] = particles.multiSMC(fk= fkmod, N=Ns[method],
                      collect=[col], nruns=nruns, nprocs=0,
                      out_func=partial(outf, method=col.summary_name))
    avg_cpu[method] = np.mean([r['cpu'] for r in runs[method]])
    print('average cpu time (across %i runs): %f' % 
          (nruns, avg_cpu[method]))

# Plots
# =====
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')
colors = {'ON2': 'gray', 'naive': 'black'}

# IQR (inter-quartile ranges) as a function of time: Figure 11.3
plt.figure()
estimates = {method: np.array([r['result'] for r in results])
             for method, results in runs.items()}
plt.xlabel(r'$t$')
plt.ylabel('IQR (smoothing estimate)')
plt.yscale('log')
plt.xscale('log')
for method in methods:
    est = estimates[method]
    delta = np.percentile(est, 75., axis=0) - np.percentile(est, 25., axis=0)
    plt.plot(np.arange(T), delta, colors[method], linewidth=2, 
             label=long_names[method])
plt.legend(loc=4)
if savefigs:
    plt.savefig('online_iqr_vs_t_logscale.pdf')

# actual estimates
plt.figure()
mint, maxt = 0, T
miny = np.min([est[:, mint:maxt].min() for est in estimates.values()])
maxy = np.max([est[:, mint:maxt].max() for est in estimates.values()])
inflat = 1.1
ax = [mint, maxt, maxy - inflat * (maxy - miny), miny + inflat * (maxy - miny)]
for i, method in enumerate(methods):
    plt.subplot(1, len(methods), i + 1)
    plt.axis(ax)
    plt.xlabel(r'$t$')
    plt.ylabel('smoothing estimate')
    plt.title(long_names[method])
    est = estimates[method]
    for j in range(nruns):
        plt.plot(est[j, :])
if savefigs:
    plt.savefig('online_est_vs_t.pdf')

plt.show()
