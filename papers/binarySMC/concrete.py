import numpy as np
import sklearn.linear_model as lin

import particles
from particles import datasets
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssps
from particles import binary_smc as bin


dataset = datasets.Concrete()
names = dataset.predictor_names
raw = dataset.raw_data  # do NOT rescale predictors
n, p = raw.shape
response = raw[:, -1]

cols = {}
for i, k in enumerate(names):
    cols[k] = raw[:, i]
    # add log of certain variables
    if k in ['cement', 'water', 'coarse aggregate', 'age']:
        cols['log(%s)' % k] = np.log(cols[k])

# interactions
colkeys = list(cols.keys())
for i, k in enumerate(colkeys):
    for j in range(i):
        k2 = colkeys[j]
        cols[f'{k} x {k2}'] = cols[k] * cols[k2]

# add intercept last
cols['intercept'] = np.ones(n)

center = True  # Christian centered the columns for some reason
if center:
    for k, c in cols.items():
        if k != 'intercept':
            c -= c.mean(axis=0)

preds = np.stack(list(cols.values()), axis=1)
npreds = len(cols)
data = preds, response

# compare with full regression
reg = lin.LinearRegression(fit_intercept=False)
reg.fit(preds, response)

# n, p = 30, 5
# preds = np.random.randn(n, p)
# preds[:, 0] = 1. # intercept
# noise = np.random.randn(n)
# response = np.sum(preds[:, :3], axis=1) + 0.8 * noise
# data = preds, response

prior = dists.IID(bin.Bernoulli(0.5), npreds)
model = bin.BayesianVS(data=data, prior=prior)

# gam, l = model.complete_enum()
# probs = rs.exp_and_normalise(l)
# marg_probs = np.average(gam, weights=probs, axis=0)

# print(marg_probs)

N = 10**5
P = 1000
M = N // P
move = ssps.MCMCSequenceWF(mcmc=bin.BinaryMetropolis(), len_chain=P)
fk = ssps.AdaptiveTempering(model, len_chain=P, move=move)
results = particles.multiSMC(fk=fk, N=M, verbose=True, nruns=3, nprocs=0)
# pf = particles.SMC(fk=fk, N=1000, verbose=True)
# pf.run()
