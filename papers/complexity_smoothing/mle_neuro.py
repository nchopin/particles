#!/usr/bin/env python

r"""
Numerical experiment of Chapter 14 (maximum likelihood estimation).

See Figure 14.8 and surrounding discussion.
The considered model and data are from Temereanca et al (2008):

    X_0 ~ N(0, sigma^2)
    X_t = rho X_{t-1} + \sigma U_t,     U_t ~ N(0, 1)
    Y_t ~ Bin(50, logit_inv(X_t))

    where logit_inv(x) = 1/(b 1 + exp(-x))

The parameter is theta = (rho, sigma^2), with 0 <= rho <= 1, and sigma^2 >= 0.

We plot the contour of the log-likelihood function, and compare the following
algorithms:

    * EM
    * Nelder-Mead (simplex)
    * gradient ascent (not plotted)

Note: MLE is (rho, sigma2) = (0.9981, 0.1089)

"""


import itertools
import time
import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.special import expit

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import state_space_models as ssms

# data
data = dts.Neuro().data
T = len(data)

# state space model
class NeuroXp(ssms.StateSpaceModel):
    default_params = {'M': 50, 'rho': .99, 'sig2': .0121}
    # values from Heng et al
    def PX0(self):
        return dists.Normal()
    def PX(self, t, xp):
        return dists.Normal(loc=self.rho * xp, scale=np.sqrt(self.sig2))
    def PY(self, t, xp, x):
        return dists.Binomial(n=self.M, p=expit(x))
    def upper_bound_log_pt(self, t):
        return - 0.5 * np.log(2. * np.pi * self.sig2)

def smoothing_trajectories(rho, sig2, N=100, method='mcmc', nsteps=1):
    fk = ssms.Bootstrap(ssm=NeuroXp(rho=rho, sig2=sig2), data=data)
    pf = particles.SMC(fk=fk, N=N, store_history=True)
    pf.run()
    if method == 'hybrid':
        paths = pf.hist.backward_sampling_reject(N)
    elif method == 'purereject':
        paths = pf.hist.backward_sampling_reject(N, max_trials=10**6 * N)
    elif method == 'mcmc':
        paths = pf.hist.backward_sampling_mcmc(N, nsteps=nsteps)
    else:
        raise ValueError('method not implemented')
    if method == 'purereject' or method == 'hybrid':
        ar = np.mean(pf.hist.acc_rate)
        print(f'acceptance rate for FFBS-{method}: {ar: .3f}')
    return (paths, pf.logLt)

# EM
####
def EM_step(rho, sig2, N=100, ffbs='mcmc', mcmc_steps=1):
    paths, loglik = smoothing_trajectories(rho, sig2, N=N, method=ffbs,
                                           nsteps=mcmc_steps)
    num = np.mean(sum(x * xp for x, xp in zip(paths[1:], paths[:-1])))
    den = np.mean(sum(x**2 for x in paths[:-1]))
    new_rho = num / den
    ssq = sum((x - new_rho * xp)**2
              for x, xp in zip(paths[1:], paths[:-1]))
    ssq += paths[0]**2
    new_sig2 = np.mean(ssq) / T
    return new_rho, new_sig2, loglik

def EM(rho0, sig20, N=100, maxiter=100, xatol=1e-2, ffbs='mcmc', mcmc_steps=1):
    rhos, sig2s, lls = [rho0], [sig20], []
    while len(rhos) < maxiter + 1:
        new_rho, new_sig2, ll = EM_step(rhos[-1], sig2s[-1], N=N, ffbs=ffbs,
                                        mcmc_steps=mcmc_steps)
        print(f'rho: {new_rho:.3f}, sigma^2: {new_sig2:.3f}')
        rhos.append(new_rho)
        sig2s.append(new_sig2)
        lls.append(ll)
        err = np.abs(rhos[-1] - rhos[-2]) + np.abs(sig2s[-1] - sig2s[-2])
        if err < xatol:
            break
    return {'rhos':rhos, 'sig2s': sig2s, 'lls': lls}

if __name__ == '__main__':
    for method in ['purereject', 'hybrid', 'mcmc']:
        print(f'EM algorithm, method={method}')
        rho0, sig20 = .1, .5
        tic = time.perf_counter()
        em_results = EM(rho0, sig20, N=100, xatol=1e-3, smooth_meth=method)
        cpu_time = time.perf_counter() - tic
        niter = len(em_results['lls'])
        print(f'elasped time: {cpu_time}, nr iterations: {niter}')

