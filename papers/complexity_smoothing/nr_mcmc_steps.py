#!/usr/bin/env python

r"""
Same settings as ffbs_cox_model.py, extra experiment to evaluate the impact of the
number of MCMC steps on the results. 

Bottom line: nsteps=1 works already well for this example.

"""

import time
from functools import partial

import numpy as np
import seaborn as sb  # box-plots
from matplotlib import pyplot as plt
from matplotlib import rc  # tex
from scipy import stats

import particles
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

nruns = 1000 # number of times each algo is run
N = 200
nsteps = [1, 2, 3]  # number of MCMC steps
add_func = partial(psit, mu=mu0, phi=phi0, sigma=sigma0)

def mcmc_smoothing_worker(fk=None, N=10, add_func=None, nsteps=1):
    T = fk.T
    pf = particles.SMC(fk=fk, N=N, store_history=True)
    tic = time.perf_counter()
    pf.run()
    z = pf.hist.backward_sampling_mcmc(N, nsteps=nsteps)
    est = np.zeros(T - 1)
    for t in range(T - 1):
        est[t] = np.mean(add_func(t, z[t], z[t + 1]))
    cpu = time.perf_counter() - tic
    print(f'mcmc worker took {cpu} s to complete, with nsteps={nsteps}')
    return {'est': est, 'cpu': cpu}

results = utils.multiplexer(f=mcmc_smoothing_worker, N=N, fk=fkmod,
                            nsteps=nsteps, add_func=add_func, 
                            nprocs=0, nruns=nruns) 

# for reference, "exact" method
ref_method = 'hybrid'
ref_results = utils.multiplexer(f=smoothing_worker, N=N,
                                method=f'FFBS_{ref_method}',
                                fk=fkmod, add_func=add_func,
                                nprocs=0, nruns=nruns)

for r in ref_results:
    r['nsteps'] = ref_method

results += ref_results

# Plots
# =====
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')
palette = sb.dark_palette("lightgray", n_colors=5, reverse=False)
sb.set_palette(palette)
rc('text', usetex=True)  # latex

# box-plot of est. errors vs nr of steps
plt.figure()
plt.xlabel('nr MCMC steps')
plt.ylabel('smoothing estimate')
# remove FFBS_reject, since estimate has the same distribution as for FFBS ON2
sb.violinplot(y=[np.mean(r['est']) for r in results],
              x=[r['nsteps'] for r in results],
              palette=palette,
              flierprops={'marker': 'o',
                          'markersize': 4,
                          'markerfacecolor': 'k'})
if savefigs:
    plt.savefig('offline_mcmc_boxplots_est_vs_nsteps.pdf')

# CPU times as a function of nr steps
plt.figure()
# in case we want to plot the mean vs N instead
# lsts = {'FFBS_ON2': '-', 'FFBS_purereject': '--', 
#         'FFBS_hybrid': '-.', 'FFBS_MCMC': ':'}
sb.boxplot(y=[r['cpu'] for r in results],
        x=[r['nsteps'] for r in results],
        palette=palette,
        flierprops={'marker': 'o',
            'markersize': 4,
            'markerfacecolor': 'k'})
#plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'$N$')
plt.ylabel('cpu time (s)')
if savefigs:
    plt.savefig('offline_mcmc_cpu_vs_nsteps.pdf')

# and finally
plt.show()
