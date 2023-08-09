#!/usr/bin/env python

"""
Particle Gibbs (with or without the backward step) for the Theta-logistic
model (2nd numerical example in Chapter 16 on PMCMC, Figures 16.8 to 16.10).
"""


from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, stats
from statsmodels.tsa.stattools import acf

from particles import datasets as dts
from particles import distributions as dists
from particles import mcmc
from particles import state_space_models as ssms


# state-space model
class ThetaLogisticReparametrised(ssms.ThetaLogistic):
    default_params = {'precX': 4., 'precY': 6.25, 'tau0': 0.15,
                          'tau1': 0.12, 'tau2': 0.1}
    def __init__(self, **kwargs):
        ssms.ThetaLogistic.__init__(self, **kwargs)
        self.sigmaX = 1. / np.sqrt(self.precX)
        self.sigmaY = 1. / np.sqrt(self.precY)

ssm_cls = ThetaLogisticReparametrised

# data
data = dts.Nutria().data

# prior
dict_prior = OrderedDict()
dict_prior['tau0'] = dists.TruncNormal(b=3.)
dict_prior['tau1'] = dists.TruncNormal(b=3.)
dict_prior['tau2'] = dists.TruncNormal(b=3.)
dict_prior['precX'] = dists.Gamma(a=2., b=1.)
dict_prior['precY'] = dists.Gamma(a=2., b=1.)
prior = dists.StructDist(dict_prior)
# parameter names on plots
pretty_par_names = {'tau0': r'$\tau_0$', 'tau1': r'$\tau_1$', 'tau2': r'$\tau_2$',
                    'precX': r'$1/\sigma_X^2$', 'precY': r'$1/\sigma_Y^2$',
                    'x_0': r'$x_0$'}

# Particle Gibbs
class PGibbs(mcmc.ParticleGibbs):
    def update_theta(self, theta, x):
        new_theta = theta.copy()
        ax = np.array(x)
        dax = np.diff(ax)
        ay = np.array(data).flatten()
        tau0 = theta['tau0']
        tau1 = theta['tau1']
        tau2 = theta['tau2']

        # update precisions
        deltaX = dax - tau0 + tau1 * np.exp(tau2 * ax[:-1])
        new_theta['precX'] = prior.laws['precX'].posterior(deltaX).rvs()
        deltaY = ax - ay
        new_theta['precY'] = prior.laws['precY'].posterior(deltaY).rvs()

        # Metropolis step for tau2
        scale_tau2 = 0.2
        tau2_prop = stats.norm.rvs(loc=tau2, scale=scale_tau2)
        if tau2_prop < 0.:
            log_prob = -np.inf
        else:
            new_deltaX = dax - tau0 + tau1 * np.exp(tau2_prop * ax[:-1])
            log_prob = 0.5 * new_theta['precX']* (np.sum(deltaX**2)
                                                  -np.sum(new_deltaX**2))
            log_prob += (prior.laws['tau2'].logpdf(tau2_prop)
                         - prior.laws['tau2'].logpdf(theta['tau2']))
        if np.log(stats.uniform.rvs()) < log_prob:
            new_theta['tau2'] = tau2_prop
            # ugly hack to store the number of accepted steps
            if hasattr(self, 'nacc_tau2'):
                self.nacc_tau2 += 1
            else:
                self.nacc_tau2 = 1

        # update of tau0 (tau1 kept fixed)
        # delta0 = (dax - tau0 + tau1 * np.exp(new_theta['tau2'] * ax[:-1]))
        # sigX = 1. / np.sqrt(new_theta['precX'])
        # new_theta['tau0'] = prior.laws['tau0'].posterior(delta0,
        #                                                  s=sigX).rvs()

        # joint update of tau0 and tau1
        T = ay.shape[0]
        features = np.ones((T - 1, 2))
        features[:, 1] = - np.exp(new_theta['tau2'] * ax[:-1])
        xtx = np.dot(features.T, features)
        beta_ols = linalg.solve(xtx, np.matmul(features.T, dax))
        muprior = np.array([prior.laws[p].mu for p in ['tau0', 'tau1']])
        Qprior = np.diag([prior.laws[p].sigma**(-2) for p in ['tau0', 'tau1']])
        Qpost = Qprior + new_theta['precX'] * xtx
        Sigpost = linalg.inv(Qpost)
        mpost = (np.matmul(Qprior, muprior)
                 + np.matmul(Sigpost, new_theta['precX']
                                 * np.matmul(xtx, beta_ols)))
        while True:
            # reject until tau0 and tau1 are > 0
            v = stats.multivariate_normal.rvs(mean=mpost, cov=Sigpost)
            if np.all(v > 0.):
                break
        new_theta['tau0'] = v[0]
        new_theta['tau1'] = v[1]

        return new_theta

algos = OrderedDict()
niter = 10 ** 5
burnin = int(niter / 10)
for name, opt in zip(['pg-back', 'pg'], [True, False]):
    algos[name] = PGibbs(ssm_cls=ssm_cls, data=data, prior=prior, Nx=50,
                         niter=niter, backward_step=opt, store_x=True,
                         verbose=10)

for alg_name, alg in algos.items():
    print('\nRunning ' + alg_name)
    alg.run()
    print('CPU time: %.2f min' % (alg.cpu_time / 60))

# Update rates
def update_rate(x):
    """Update rate.

    Parameters
    ----------
    x: (N,T) or (N,T,d) array

    Returns
    -------
    a (T,) or (T,d) array containing the frequency at which each
    component was updated (along axis 0)
    """
    return np.mean(x[1:] != x[:-1], axis=0)

# PLOTS
# =====
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')
colors = {'pg-back': 'black', 'pg': 'gray'}
linestyles = {'pg-back': '-', 'pg': '--'}

# Update rates of PG samplers
plt.figure()
for alg_name, alg in algos.items():
    plt.plot(update_rate(alg.chain.x[burnin:]), label=alg_name, linewidth=2,
             color=colors[alg_name], linestyle=linestyles[alg_name])
plt.axis([0, data.shape[0], 0., 1.0])
plt.xlabel('t')
plt.ylabel('update rate')
plt.legend(loc=6)  # center left
if savefigs:
    plt.savefig('ecological_update_rates.pdf')  # Figure 16.8

# pair plots from PG-back
plt.figure()
thin = int(niter / min(1000, niter))
# at most 1000 points, so that file is not too heavy
th = algos['pg-back'].chain.theta[(thin - 1):niter:thin]
i = 1
for p1, p2 in [('tau1', 'tau0'), ('tau2', 'tau0')]:
    plt.subplot(1, 2, i)
    plt.scatter(th[p1], th[p2], c='k')
    plt.axis([0., 2., 0., 2.])
    plt.xlabel(pretty_par_names[p1])
    if i == 1:
        plt.ylabel(pretty_par_names[p2])
    i += 1
if savefigs:
    plt.savefig('ecological_pairplot_taus.pdf')  # Figure 16.10

# MCMC traces
plt.figure()
for i, p in enumerate(dict_prior.keys()):
    plt.subplot(2, 3, i + 1)
    for alg_name, alg in algos.items():
        th = alg.chain.theta[p]
        plt.plot(th, label=alg_name, color=colors[alg_name])
    plt.xlabel('iter')
    plt.ylabel(pretty_par_names[p])
plt.tight_layout()

# compare marginals of parameter components
plt.figure()
for i, p in enumerate(dict_prior.keys()):
    plt.subplot(2, 3, i + 1)
    for alg_name, alg in algos.items():
        th = alg.chain.theta[p][burnin:]
        plt.hist(th, 40, density=True, alpha=0.5, label=alg_name,
                 histtype='stepfilled')
        plt.ylabel(pretty_par_names[p])
    plt.legend()
plt.tight_layout()

# ACFs
plt.figure()
nlags = 100
for i, p in enumerate(list(dict_prior.keys()) + ['x_0']):
    plt.subplot(2, 3, i + 1)
    for alg_name, alg in algos.items():
        th = alg.chain.x[:, 0] if p=='x_0' else alg.chain.theta[p]
        acf_th = acf(th[burnin:], nlags=nlags, fft=True)
        plt.plot(acf_th, label=alg_name, color=colors[alg_name],
                linestyle=linestyles[alg_name])
    plt.axis([0, nlags, -0.03, 1.])
    plt.xlabel('lag')
    plt.ylabel(pretty_par_names[p])
    if i == 2:
        plt.legend()
plt.tight_layout()
if savefigs:
    plt.savefig('ecological_acfs.pdf')  # Figure 16.9

plt.show()
