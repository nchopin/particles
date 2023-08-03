#!/usr/bin/env python

"""
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

from __future__ import division, print_function

import itertools
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.special import expit
from scipy import optimize

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

class MinusLogLikEvaluator(object):
    def __init__(self, N=1000):
        self.N = N
        self.args = []
        self.lls = []
    def __call__(self, x):
        rho, sig2 = x[0], x[1]
        if rho > 1. or rho < 0. or sig2 <=0:
            return np.inf
        fk = ssms.Bootstrap(ssm=NeuroXp(rho=rho, sig2=sig2), data=data)
        pf = particles.SMC(fk=fk, N=self.N, qmc=True)
        pf.run()
        self.args.append(x)
        self.lls.append(pf.logLt)
        return - pf.logLt

def smoothing_trajectories(rho, sig2, N=100):
    fk = ssms.Bootstrap(ssm=NeuroXp(rho=rho, sig2=sig2), data=data)
    pf = particles.SMC(fk=fk, N=N, qmc=False, store_history=True)
    pf.run()
    (paths, ar) = pf.hist.backward_sampling(N, return_ar=True,
                                            linear_cost=True)
    print('Acceptance rate (FFBS-reject): %.3f' % ar)
    return (paths, pf.logLt)

# saving intermediate results
#############################
save_intermediate = False
results_file = 'neuro_results.pkl'
all_results = {}

def save_results(new_results):
    all_results.update(new_results)
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)

# Evaluate log-likelihood on a grid
###################################
ng = 50
rhos = np.linspace(0., 1., ng)
sig2s = np.linspace(1e-2, 25., ng)  # for sigma=0., returns Nan
ijs = list(itertools.product(range(ng), range(ng)))
fks = {ij: ssms.Bootstrap(ssm=NeuroXp(rho=rhos[ij[0]], sig2=sig2s[ij[1]]),
                          data=data)
       for ij in ijs}
outf = lambda pf: pf.logLt
nruns = 5
print('computing log-likelihood on a grid')
results = particles.multiSMC(fk=fks, N=100, qmc=True, nruns=nruns, nprocs=0,
                             out_func=outf)
save_results({'results': results})

# EM
####
def EM_step(rho, sig2, N=100):
    paths, loglik = smoothing_trajectories(rho, sig2, N=N)
    num = np.mean(sum(x * xp for x, xp in zip(paths[1:], paths[:-1])))
    den = np.mean(sum(x**2 for x in paths[:-1]))
    new_rho = num / den
    ssq = sum((x - new_rho * xp)**2
              for x, xp in zip(paths[1:], paths[:-1]))
    ssq += paths[0]**2
    new_sig2 = np.mean(ssq) / T
    return new_rho, new_sig2, loglik

def EM(rho0, sig20, N=100, maxiter=100, xatol=1e-2):
    rhos, sig2s, lls = [rho0], [sig20], []
    while len(rhos) < maxiter + 1:
        new_rho, new_sig2, ll = EM_step(rhos[-1], sig2s[-1], N=N)
        print('rho: %.3f, sigma^2: %.3f' % (new_rho, new_sig2))
        rhos.append(new_rho)
        sig2s.append(new_sig2)
        lls.append(ll)
        err = np.abs(rhos[-1] - rhos[-2]) + np.abs(sig2s[-1] - sig2s[-2])
        if err < xatol:
            break
    return {'rhos':rhos, 'sig2s': sig2s, 'lls': lls}

print('EM algorithm')
rho0, sig20 = .1, .5
em_results = EM(rho0, sig20, N=100, xatol=1e-3)
save_results(em_results)

# gradient-free optimisation
############################
print('gradient-free optim (Nelder-Mead)')
ll_eval = MinusLogLikEvaluator(N=5000)
res_opt = optimize.minimize(ll_eval, [rho0, sig20], method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-2})
save_results({'ll_eval': ll_eval})

# gradient ascent
#################
def grad(rho, sig2, N=100):
    paths, _ = smoothing_trajectories(rho, sig2, N=N)
    residuals = [(x - rho * xp) for x, xp in zip(paths[1:], paths[:-1])]
    sr2 = np.mean(sum(r**2 for r in residuals))
    grad_sig2 = 0.5 * (sr2 / sig2**2 - (T + 1) / sig2)
    srx = np.mean(sum(r * x for r, x in zip(residuals[1:], paths[:-1])))
    grad_rho = srx / sig2
    print('grads: %.3f, %3f' % (grad_rho, grad_sig2))
    return grad_rho, grad_sig2

def grad_ascent_step(rho, sig2, N=100, lambdat=1e-4):
    grad_rho, grad_sig2 = grad(rho, sig2, N=N)
    new_rho = rho + lambdat * grad_rho
    new_sig2 = sig2 + lambdat * grad_sig2
    return new_rho, new_sig2

print('gradient ascent')
rhos_ga, sig2s_ga = [rho0], [sig20]
for _ in range(100):
    new_rho, new_sig2 = grad_ascent_step(rhos_ga[-1], sig2s_ga[-1], lambdat=1e-6)
    rhos_ga.append(new_rho)
    sig2s_ga.append(new_sig2)
    print('rho=%.3f, sig2=%.3f' % (rhos_ga[-1], sig2s_ga[-1]))

save_results({'rhos_ga': rhos_ga, 'sig2s_ga': sig2s_ga})

# PLOTS
#######
plt.style.use('ggplot')
savefigs = True  # False if you don't want to save plots as pdfs

# contour plots
rho_mg, sig2_mg = np.meshgrid(rhos, sig2s)
ll_arr = np.zeros((ng, ng, nruns))
for r in results:
    i, j = r['fk']
    k = r['run']
    ll_arr[i, j, k] = r['output']
ll_var = np.var(ll_arr, axis=2)
ll_max = ll_arr.max(axis=2)
ll_mean = (np.log(np.mean(np.exp(ll_arr - ll_max[:, :, np.newaxis]), axis=2))
           + ll_max)
# mean is actually log-mean-exp

# variance contours
fig, ax = plt.subplots()
levels = (0.1, 1., 10., 100.)
CS = ax.contourf(rho_mg, sig2_mg, ll_var.T, levels=levels, cmap=plt.cm.bone)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\sigma^2$')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('variance log-lik')
if savefigs:
    plt.savefig('neuro_contour_var.pdf')

# log-lik contours
fig, ax = plt.subplots()
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
imax, jmax = np.unravel_index(np.argmax(ll_mean), (ng, ng))
mle_ll = ll_mean[imax, jmax]
ax.plot(rhos[imax], sig2s[jmax], markersize=3, marker="o", color='black')
levels = mle_ll + np.linspace(- 2 * 10**4, -0., 121)
CS = ax.contour(rho_mg, sig2_mg, ll_mean.T, levels=levels, colors='lightgray')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\sigma^2$')
ax.set_xlim(left=0., right=1.)
ax.set_ylim(bottom=0., top=sig2s[-1])

# add EM points
ax.plot(em_results['rhos'], em_results['sig2s'], 'o-k', zorder=20, label='EM')

# add simplex points
rhos_simplex = [x[0] for x in ll_eval.args]
sig2s_simplex = [x[1] for x in ll_eval.args]
ax.scatter(rhos_simplex, sig2s_simplex, marker='x', c='gray', zorder=10,
           label='simplex')
ax.legend(loc='upper left')
if savefigs:
    plt.savefig('neuro_contour_ll_with_EM_and_simplex.pdf')
