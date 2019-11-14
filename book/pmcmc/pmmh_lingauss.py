#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explore basic properties of PMMH on the following toy-example: 
    X_0 ~ N(0, \sigma_X^2)
    X_t = rho X_{t-1} + U_t,    U_t ~ N(0, sigma_X^2)
    Y_t = X_t + V_t,            V_t ~ N(0, sigma_Y^2) 
with theta=(rho, sigma_X^2, sigma_Y^2), and the following prior:
    + rho ~ U([-1,1])
    + sigma_X^2, sigma_Y^2 ~ Inv-Gamma(2, 2)

See end of Chapter 15 (Bayesian estimation) in the book, especially Figures 
15.4 to 15.6 and the surrounding discussion.  

Warning: takes more than 10 hrs to complete. 
"""

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as random 
import pandas
import seaborn as sb
from scipy import stats
from statsmodels.tsa.stattools import acf

import particles
from particles import distributions as dists
from particles import kalman
from particles import mcmc
from particles import smc_samplers
from particles import state_space_models as ssms 

# prior 
dict_prior = {'varX': dists.InvGamma(a=2., b=2.),
              'varY': dists.InvGamma(a=2., b=2.),
              'rho':dists.Uniform(a=-1., b=1.)
             }
prior = dists.StructDist(dict_prior)

# State-space model 
class ReparamLinGauss(kalman.LinearGauss):
    def __init__(self, varX=1., varY=1., rho=0.):
        sigmaX = np.sqrt(varX)
        sigmaY = np.sqrt(varY)
        sigma0 = sigmaX
        # Note: We take X_0 ~ N(0, sigmaX^2) so that Gibbs step is tractable
        kalman.LinearGauss.__init__(self, sigmaX=sigmaX, sigmaY=sigmaY, rho=rho,
                                    sigma0=sigma0)
 
# data was simulated as follows: 
# _, data = ReparamLinGauss(varX=1., varY=(0.2)**2, rho=.9).simulate(100)
data = np.loadtxt('./simulated_linGauss_T100_varX1_varY.04_rho.9.txt')


niter = 10 ** 5
burnin = int(niter/ 10)
algos = OrderedDict()
#rw_cov = np.diag(np.array([0.5, 0.5, 0.05]) ** 2)  # standard deviations
rw_cov = (0.15)**2 * np.eye(3)

# Basic Metropolis sampler 
class StaticLGModel(smc_samplers.StaticModel):
    def loglik(self, theta, t=None): 
        # Note: for simplicity we ignore argument t here,
        # and compute the full log-likelihood
        ll = np.zeros(theta.shape[0])
        for n, th in enumerate(theta): 
            mod = ReparamLinGauss(**smc_samplers.rec_to_dict(th))
            kf = kalman.Kalman(ssm=mod, data=data)
            kf.filter()
            ll[n] = np.sum(kf.logpyt)
        return ll 

sm = StaticLGModel(data=data, prior=prior)
algos['mh'] = mcmc.BasicRWHM(model=sm, niter=niter, adaptive=False, 
                             rw_cov=rw_cov, verbose=10)

# SMC^2 algorithm 
# Ntheta_smc2 = 5000
# fk_smc2 = smc_samplers.SMC2(ssm_cls=ReparamLinGauss, prior=prior, data=data,
#                             mh_options={'nsteps':0}, init_Nx=200, 
#                             ar_to_increase_Nx=0.1)
# algos['smc2'] = particles.SMC(fk=fk_smc2, N=Ntheta_smc2, compute_averages=True, 
#                            verbose=True)

# PMMH algorithms 
Nxs =  list(range(100, 1600, 100))  # needs list for Python 3 

for Nx in Nxs:
    key = 'pmmh' + '-%d' % Nx
    algos[key] = mcmc.PMMH(ssm_cls=ReparamLinGauss, prior=prior, data=data,
                           Nx=Nx, niter=niter, adaptive=False, rw_cov=rw_cov, 
                           verbose=10)

# Run the algorithms 
####################

for alg_name, alg in algos.items(): 
    print('\nRunning ' + alg_name)
    alg.run()
    print('CPU time: %.2f min' % (alg.cpu_time / 60))

# Compute variances 
###################
thin = int(niter / 100)  # compute average (of variances) over 100 points
thetas = algos['mh'].chain.theta[(burnin - 1)::thin]
fks = {k: ssms.Bootstrap(ssm=ReparamLinGauss(**smc_samplers.rec_to_dict(th)), data=data)
                        for k, th in enumerate(thetas)}
outf = lambda pf: pf.logLt
print('Computing variances of log-lik estimates as a function of N')
results = particles.multiSMC(fk=fks, N=Nxs, nruns=4, nprocs=0, out_func=outf)
df = pandas.DataFrame(results)
df_var = df.groupby(['fk', 'N']).var()  # variance as a function of fk and N
df_var = df_var.reset_index()
df_var_mean = df_var.groupby('N').mean()  # mean variance as function of N

# Plots
#######
savefigs = False
plt.style.use('ggplot')

def msjd(theta):
    """Mean square jump distance.
    """
    s = 0. 
    for p in theta.dtype.names:
        s += np.sum(np.diff(theta[p], axis=0) ** 2)
    return s

# pair plots from ideal sampler 
sb.pairplot(pandas.DataFrame(data=algos['mh'].chain.theta[burnin:]))

# acceptance rates vs Nx 
plt.figure()
plt.plot(Nxs, [algos['pmmh-%d' % Nx].acc_rate for Nx in Nxs], '-ok')
best_ar = algos['mh'].acc_rate
plt.axhline(y=best_ar, ls='--', color='black')
plt.ylim(0., 1.1 * best_ar)
plt.xlabel('N')
plt.ylabel('acceptance rate')
if savefigs:
    plt.savefig('pmmh_lingauss_ar_vs_N.pdf')

# ACFs (of MCMC algorithms)
nlags = 100
pretty_names = {'rho': r'$\rho$', 'varX': r'$\sigma_X^2$'}
plt.figure()
for i, param in enumerate(['varX', 'rho']):
    plt.subplot(1, 2, i + 1)
    for alg_name, col in zip(['pmmh-100', 'pmmh-300', 'pmmh-500', 'mh'],
                             reversed(np.linspace(0., 0.8, 4))):
        plt.plot(acf(algos[alg_name].chain.theta[param][burnin:], 
                     nlags=nlags, fft=True), label=alg_name, color=str(col),
                 lw=2)
        plt.title(pretty_names[param])
        plt.ylim(-0.05, 1.)
plt.legend()
if savefigs:
    plt.savefig('pmmh_lingauss_acfs.pdf')

# msjd vs variance of loglik estimates
plt.figure()
var_N, msjd_N = OrderedDict(), OrderedDict()
for N in Nxs:
    var_N[N] = df_var_mean['output'].loc[N] 
    msjd_N[N] = msjd(algos['pmmh-%d' % N].chain.theta[burnin:]) 
plt.plot(list(var_N.values()), list(msjd_N.values()), 'ok-')
for N in [100, 200, 300, 400, 600, 1500]:
    plt.text(var_N[N], msjd_N[N] + 20, str(N))
ideal_msjd = msjd(algos['mh'].chain.theta[burnin:])
plt.axhline(y=ideal_msjd, ls='--', color='black')
plt.xlabel('var log-lik')
plt.ylabel('mean squared jumping distance')
plt.xlim(xmin=0., xmax=max(var_N.values()) * 1.05)
plt.ylim(ymin=0., ymax=1.15 * ideal_msjd)
if savefigs:
    plt.savefig('pmmh_lingauss_msjd_vs_var_ll.pdf')

# CPU time vs N
plt.figure()
plt.plot(Nxs, [algos['pmmh-%d' % N].cpu_time for N in Nxs], 'ok-')
plt.xlabel('N')
plt.ylabel('CPU time')

plt.show()

