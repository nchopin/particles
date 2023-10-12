#!/usr/bin/env python

"""
Checks that IBIS samples from the correct (Gaussian) distribution when the
considered model is a linear regression.

Works in progress.

"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from scipy import stats

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssps
from particles.collectors import Moments


# for each dataset, we adapt:
# * N: number of particles
# * Ks = list of Ks (nr MCMC steps)
# * typK: value of M used for plots on "typical" run

# data
T, d = 30, 3
preds = np.random.randn(T, d)
preds[:, 0] = 1. # intercept
true_beta = np.array([0.3, 1., -0.2])
sig = 0.1  # fixed sigma for now
response = preds @ true_beta + sig * np.random.randn(T)
data = np.empty((T, d+1))
data[:, 0] = response
data[:, 1:] = preds

# prior, model and true values
scale_prior = 10.
cov_margy = sig**2 * np.eye(T) + scale_prior**2 * preds @ preds.T
true_evid = stats.multivariate_normal.logpdf(response, cov=cov_margy)
true_prec = ((1./sig**2) * preds.T @ preds 
             + (1. / scale_prior**2) * np.eye(d))
true_covp = np.linalg.inv(true_prec)
true_meanp = true_covp @ ((preds.T @ response) / sig**2)
prior = dists.StructDist({'beta':dists.MvNormal(scale=scale_prior,
                                                cov=np.eye(d))})

class LinearRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], self.data[t, 1:])
        return stats.norm.logpdf(self.data[t, 0], loc=lin, scale=sig)


# algorithms
N = 10_000
lc = 50
nruns = 1000

model = LinearRegression(data=data, prior=prior)
fk = ssps.IBIS(model=model, len_chain=lc)
results = particles.multiSMC(fk=fk, N=N, nruns=nruns, collect=[Moments], verbose=True, nprocs=0)

## PLOTS
plt.style.use('ggplot')

plt.figure()
plt.hist([r['output'].logLt for r in results], 30, density=True)
plt.axvline(x=true_evid, ymin=0., ymax=1., ls=':', c='black')
plt.xlabel('log evidence')

for r in results:
    alg = r['output']
    dmv = rs.wmean_and_var(alg.W, alg.X.theta['beta'])
    r.update({'mean_theta': dmv['mean'], 'cov_theta': dmv['var']})

plt.figure()
for i in range(d):
    plt.subplot(1, d, i + 1)
    plt.hist([r['mean_theta'][i] for r in results], 30)
    plt.axvline(x=true_meanp[i], ls=':', c='black')                     
plt.suptitle('posterior mean')

plt.figure()
for i in range(d):
    plt.subplot(1, d, i + 1)
    plt.hist([r['cov_theta'][i] for r in results], 30)
    plt.axvline(x=true_covp[i, i], ls=':', c='black')                     
plt.suptitle('posterior variance')


