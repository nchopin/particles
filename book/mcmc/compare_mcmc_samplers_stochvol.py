#!/usr/bin/env python
"""
Compare MCMC samplers to sample from the smoothing distribution of a basic
stochastic volatility model; see the numerical example at the end of Chapter 15
(MCMC).

Considered algorithms are:
* a basic, one-at-a-time Gibbs sampler
* the marginal sampler of  of Titsias and Papaspiliopoulos
* (optionally) Particle Gibbs

In addition, we use QMC-FFBS as a gold standard (since the target distribution
is a smoothing distribution, i.e. the distribution of X_{0:T} given Y_{0:T}).

"""


from collections import OrderedDict
import time

from matplotlib import pyplot as plt
import numpy as np
import numpy.random as random
from scipy import linalg
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.gofplots import qqplot_2samples

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import mcmc
from particles import smc_samplers
from particles import state_space_models

# data
T = 200
data = dts.GBP_vs_USD_9798().data[:T]

# prior
# Note: the MCMC samplers require a prior as an argument, but since we are
# not estimating the parameter theta, these samplers are designed so as
# to keep theta constant
dict_prior = {
    'mu': dists.Normal(scale=2.),
    'sigma': dists.Gamma(a=2., b=2.),
    'rho': dists.Beta(a=9., b=1.)
}
prior = dists.StructDist(dict_prior)
mu0, sigma0, rho0 = -1.02, 0.178, 0.9702
theta0 = np.array([(mu0, rho0, sigma0)],
                  dtype=[('mu', float), ('rho', float), ('sigma', float)])

ssm_cls = state_space_models.StochVol
ssm = ssm_cls(mu=mu0, sigma=sigma0, rho=rho0)

# (QMC-)FFBS as a reference
N = 3000
tic = time.time()
fk = state_space_models.Bootstrap(ssm=ssm, data=data)
pf = particles.SMC(fk=fk, N=N, qmc=True, store_history=True)
pf.run()
smth_traj = pf.hist.backward_sampling_qmc(M=N)
cpu_time_fbbs = time.time() - tic
print('FFBS-QMC: run completed, took %f min' % (cpu_time_fbbs / 60.))


def reject_sv(m, s, y):
    """ Sample from N(m, s^2) times SV likelihood using rejection.

    SV likelihood (in x) corresponds to y ~ N(0, exp(x)).
    """
    mp = m + 0.5 * s**2 * (-1. + y**2 * np.exp(-m))
    ntries = 0
    while True:
        ntries += 1
        x = stats.norm.rvs(loc=mp, scale=s)
        u = stats.uniform.rvs()
        if np.log(u) < -0.5 * y**2 * (np.exp(-x) - np.exp(-m) * (1. + m - x)):
            break
        if ntries > 1000:
            print('1000 failed attempt, m,s,y=%f, %f, %f' % (m, s, y))
            break
    return x


class SVmixin:
    def update_theta(self, theta, x):
        return theta


# one-at-a-time Gibbs
#####################


class Gibbs_SV(SVmixin, mcmc.GenericGibbs):
    def update_states(self, theta, x):
        dt = smc_samplers.rec_to_dict(theta)
        T = len(self.data)
        if x is None:
            new_x, _ = ssm_cls(**dt).simulate(T)
        else:
            new_x = x[:]
        mu, sigma, rho = dt['mu'], dt['sigma'], dt['rho']
        a = 1. + rho**2
        sp = sigma / np.sqrt(a)
        c = rho / a
        for t, yt in enumerate(self.data):
            if t == 0:
                m = mu + rho * (new_x[1] - mu)
                s = sigma
            elif t == T - 1:
                m = mu + rho * (new_x[-2] - mu)
                s = sigma
            else:
                m = mu + c * (new_x[t - 1] + new_x[t + 1] - 2. * mu)
                s = sp
            new_x[t] = reject_sv(m, s, yt)
        return new_x


# Marginal sampler
##################

class Marginal(mcmc.GenericGibbs):
    def __init__(self, niter=10, verbose=0, theta0=None, ssm_cls=None,
                 prior=None, data=None, store_x=False, delta=0.3, Cinv=None):
        mcmc.GenericGibbs.__init__(self, niter=niter, verbose=verbose,
                                   theta0=theta0, ssm_cls=ssm_cls,
                                   prior=prior, data=data, store_x=store_x)
        self.delta = delta
        self.tod = 2. / delta  # tod = two over delta
        self.T = len(self.data)
        self.A = np.linalg.inv(Cinv + self.tod * np.eye(T))
        self.cov = self.tod * np.dot(self.A, self.A) + self.A
        self.nacc = 0
        self.data_arr = np.array(self.data)

    def log_lik(self, x):
        xm = x + mu0
        return -0.5 * np.sum(xm + self.data_arr**2 * np.exp(-xm))

    def grad_log_lik(self, x):
        xm = x + mu0
        return -0.5 * (1. - self.data_arr**2 * np.exp(-xm))

    def h(self, x, y):
        gy = self.grad_log_lik(y)
        lhs = x - self.tod * np.matmul(self.A, y + (0.25 * self.delta) * gy)
        rhs = linalg.solve(self.tod * self.A + np.eye(self.T), gy)
        # TODO pre-computing the inverse here may speed things up
        return np.dot(lhs, rhs)

    def update_states(self, theta, x):
        if x is None:
            new_x, _ = ssm_cls(**smc_samplers.rec_to_dict(theta)).simulate(
                self.T)
        else:
            new_x = x[:]
        xa = np.array(new_x).flatten()
        m = np.matmul(self.A, self.tod * xa + self.grad_log_lik(xa))
        xp = stats.multivariate_normal.rvs(mean=m, cov=self.cov)
        mh_log_ratio = (self.log_lik(xp) - self.log_lik(xa) - self.h(xp, xa) +
                        self.h(xa, xp))
        if np.log(random.uniform()) <= mh_log_ratio:
            self.nacc += 1
            return list(xp)
        else:
            return new_x

    def update_theta(self, theta, x):
        return theta.copy()

# Particle Gibbs
################

class PGibbs_SV(SVmixin, mcmc.ParticleGibbs):
    pass

algos = OrderedDict()

algos['Gibbs'] = Gibbs_SV(ssm_cls=ssm_cls, data=data, prior=prior, theta0=theta0,
                          niter=10**6, store_x=True, verbose=10)

# uncomment this if you want to add PG to the comparison
# algos['particle Gibbs'] = PGibbs_SV(ssm_cls=ssm_cls, data=data, prior=prior,
#                         theta0=theta0, Nx=100, niter=10**3,
#                         backward_step=True, store_x=True,
#                         verbose=10)
#

M = (1. + rho0**2) * np.eye(T)
M[0, 0] = 1.
M[-1, -1] = 1.
for i in range(T - 1):
    M[i, i + 1] = -rho0
    M[i + 1, i] = -rho0
Cinv = M / (sigma0**2)

algos['marginal'] = Marginal(ssm_cls=ssm_cls, data=data, prior=prior, theta0=theta0,
                             niter=10**6, store_x=True, verbose=10, delta=1.,
                             Cinv=Cinv)

for alg_name, alg in algos.items():
    print('\nRunning ' + alg_name)
    alg.run()
    print('CPU time: %.2f min' % (alg.cpu_time / 60))

# re-center marginal sampler, print acceptance rate
algos['marginal'].chain.x += mu0
ar = algos['marginal'].nacc / algos['marginal'].niter
print('Acceptance rate of marginal sampler: %f' % ar)

# PLOTS
# =====
savefigs = True  # False if you don't want to save plots as pdfs
plt.style.use('ggplot')

# compare marginals of states
plt.figure()
ts = [int(k * (T - 1) / 11) for k in range(12)]
for i, t in enumerate(ts):
    plt.subplot(3, 4, i + 1)
    plt.hist(smth_traj[t], 40, density=True, alpha=0.5, label='FFBS',
             histtype='stepfilled')
    for alg_name, alg in algos.items():
        if isinstance(alg, mcmc.MCMC):
            burnin = int(alg.niter / 10)
            xs = alg.chain.x[burnin:, t]
            plt.hist(xs, 40, density=True, alpha=0.5, label=alg_name,
                     histtype='stepfilled')
            plt.xlabel(str(t))
            plt.legend()
if savefigs:
    plt.savefig('marginals_gibbs_marg_ffbs_stochvol.pdf')

# Figure 14.1: comparing ACFs
plt.figure()
nlags = 160
cols = {'Gibbs': 'gray', 'marginal': 'black'}
lss = {'Gibbs': '--', 'marginal': '-'}
for t in [0, 49, 99, 149, 199]:
    for alg_name, alg in algos.items():
        if isinstance(alg, mcmc.MCMC):
            burnin = int(alg.niter / 10)
            acf_x = acf(alg.chain.x[burnin:, t], nlags=nlags, fft=True)
            lbl = '_' if t > 0 else alg_name  # set label only once
            plt.plot(acf_x, label=lbl, color=cols[alg_name],
                     linestyle=lss[alg_name], linewidth=2)
plt.axis([0, nlags, -0.03, 1.])
plt.xlabel('lag')
plt.ylabel('ACF')
plt.legend()
if savefigs:
    plt.savefig('acf_gibbs_marginal_smoothing_stochvol.pdf')

# Figure 14.2: qq-plots to check that MCMC samplers target the same posterior
plt.figure()
qqplot_2samples(algos['Gibbs'].chain.x[:, 0], algos['marginal'].chain.x[:, 0])
if savefigs:
    plt.savefig('qqplots_gibbs_vs_marginal_stochvol.pdf')

plt.show()
