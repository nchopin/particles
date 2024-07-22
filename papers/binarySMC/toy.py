"""
Toy example based on simulated data with a small number of predictors.

Compare the output of the SMC sampler with the exact probabilities (computed
through complete enumeration).

"""


import numpy as np
import sklearn.linear_model as lin

import particles
from particles import binary_smc as bin
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssps

n, npreds = 30, 5
preds = np.random.randn(n, npreds)
preds[:, 0] = 1. # intercept
noise = np.random.randn(n)
response = np.sum(preds[:, :3], axis=1) + 0.8 * noise
data = preds, response

prior = dists.IID(bin.Bernoulli(0.5), npreds)
model = bin.BayesianVS(data=data, prior=prior)

gam, l = model.complete_enum()
probs = rs.exp_and_normalise(l)
exact_marg_probs = np.average(gam, weights=probs, axis=0)


N = 10**4
P = 100
M = N // P
move = ssps.MCMCSequenceWF(mcmc=bin.BinaryMetropolis(), len_chain=P)
fk = ssps.AdaptiveTempering(model, len_chain=P, move=move)
results = particles.multiSMC(fk=fk, N=M, verbose=True, nruns=3, nprocs=0)

est_marg_probs = np.array([np.average(r['output'].X.theta, axis=0, 
                                      weights=r['output'].W)
                           for r in results])

abs_err = np.mean(np.abs(est_marg_probs - exact_marg_probs), axis=0)
print(f'absolute error: {abs_err}')
