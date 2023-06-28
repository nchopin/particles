#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapts the offline smoothing example in the book (see Figs 12.4 and 12.5) to
compare various FFBS algorithms (including the new variants, introduced and
discussed in Dau & Chopin, 2022):

    1. standard, O(N^2) variant
    2. pure rejection
    3. hybrid (use rejection up to N attempts)
    4. MCMC


The considered state-space model is:

X_t|X_{t-1}=x_{t-1} ~ N(mu+phi(x_{t-1}-mu),sigma^2)
Y_t|X_t=x_t ~ Poisson(exp(x_t))

and the objective is to evaluate the expectation (w.r.t. the smoothing
distribution) of the score of the model (at theta=theta_0), which is an
additive function of the form:
sum_{t=0}^{T-2} \psi_t(X_t, X_{t+1})
see the book or below for a definition of psi_t.

Compared to the book, the range of N values is slightly different (from 200 to
51200, vs 50 to 12800). Also, this time we use a box-plot for CPU vs N; this
shows more clearly that the CPU time of variant 2 has a large variability.

"""

from __future__ import division, print_function

import numpy as np
import seaborn as sb  # box-plots
from matplotlib import pyplot as plt
from matplotlib import rc  # tex
from scipy import stats
from functools import partial

from particles import state_space_models as ssms
from particles import utils
from particles.smoothing import smoothing_worker


# considered class of models
class DiscreteCox_with_add_f(ssms.DiscreteCox):
    """ A discrete Cox model:
    Y_t ~ Poisson(e^{X_t})
    X_t - mu = phi(X_{t-1}-mu)+U_t,   U_t ~ N(0,1)
    X_0 ~ N(mu,sigma^2/(1-phi**2))
    """

    def upper_bound_log_pt(self, t):
        return -0.5 * np.log(2 * np.pi * self.sigma ** 2)


# Aim is to compute the smoothing expectation of
# sum_{t=0}^{T-2} \psi(t, X_t, X_{t+1})
# here, this is the score at theta=theta_0
def psi0(x, mu, phi, sigma):
    return -0.5 / sigma**2 + (0.5 * (1. - phi**2) / sigma**4) * (x - mu)**2

def psit(t, x, xf, mu, phi, sigma):
    """ A function of t, X_t and X_{t+1} (f=future) """
    if t == 0:
        return psi0(x, mu, phi, sigma) + psit(1, x, xf, mu, phi, sigma)
    else:
        return -0.5 / sigma**2 + (0.5 / sigma**4) * ((xf - mu)
                                                     - phi * (x - mu))**2

# logpdf of gamma_{t}(dx_t), the 'prior' of the information filter
def log_gamma(x, mu, phi, sigma):
    return stats.norm.logpdf(x, loc=mu,
                             scale=sigma / np.sqrt(1. - phi ** 2))


# set up model, simulate data
T = 100
mu0 = 0.
phi0 = .9
sigma0 = .5  # true parameters
my_ssm = DiscreteCox_with_add_f(mu=mu0, phi=phi0, sigma=sigma0)
_, data = my_ssm.simulate(T)

# FK models
fkmod = ssms.Bootstrap(ssm=my_ssm, data=data)
# FK model for information filter: same model with data in reverse
fk_info = ssms.Bootstrap(ssm=my_ssm, data=data[::-1])

nruns = 100 # run each algo 100 times
Ns = [200, 800, 3200, 12800, 51200]  # note: in book, from 50 to 12800
methods = ['FFBS_ON2', 'FFBS_purereject', 'FFBS_MCMC', 'FFBS_hybrid']

add_func = partial(psit, mu=mu0, phi=phi0, sigma=sigma0)
log_gamma_func = partial(log_gamma, mu=mu0, phi=phi0, sigma=sigma0)
results = utils.multiplexer(f=smoothing_worker, method=methods, N=Ns,
                            fk=fkmod, fk_info=fk_info, add_func=add_func,
                            log_gamma=log_gamma_func, nprocs=0, nruns=nruns) 

# Plots
# =====
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')
rc('text', usetex=True)  # latex

pretty_names = {}
ON = r'$\mathcal{O}(N)$'
ON2 = r'$\mathcal{O}(N^2)$'
pretty_names['FFBS_ON2'] = ON2 + r' FFBS'
pretty_names['FFBS_purereject'] = 'FFBS-reject'
pretty_names['FFBS_hybrid'] = 'FFBS hybrid'
pretty_names['FFBS_MCMC'] = 'FFBS MCMC'

# box-plot of est. errors vs N and method (Figure 11.4)
plt.figure()
plt.xlabel(r'$N$')
plt.ylabel('smoothing estimate')
# remove FFBS_reject, since estimate has the same distribution as for FFBS ON2
res_nofon = [r for r in results if r['method'] != 'FFBS_purereject']
sb.violinplot(y=[np.mean(r['est']) for r in res_nofon],
              x=[r['N'] for r in res_nofon],
              hue=[pretty_names[r['method']] for r in res_nofon],
              flierprops={'marker': 'o',
                          'markersize': 4,
                          'markerfacecolor': 'k'})
if savefigs:
    plt.savefig('offline_boxplots_est_vs_N.pdf')

# CPU times as a function of N (Figure 11.5)
plt.figure()
# in case we want to plot the mean vs N instead
# lsts = {'FFBS_ON2': '-', 'FFBS_purereject': '--', 
#         'FFBS_hybrid': '-.', 'FFBS_MCMC': ':'}
sb.boxplot(y=[r['cpu'] for r in results],
        x=[r['N'] for r in results],
        hue=[pretty_names[r['method']] for r in results],
        flierprops={'marker': 'o',
            'markersize': 4,
            'markerfacecolor': 'k'})
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$N$')
plt.ylabel('cpu time (s)')
plt.legend(loc=2)
if savefigs:
    plt.savefig('offline_cpu_vs_N.pdf')

# and finally
plt.show()
