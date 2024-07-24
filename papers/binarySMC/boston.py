"""
First example from Schäfer and Chopin (2013), Boston housing dataset.

To generate bar plots as in the paper, see bar_plots.py
"""


import numpy as np
import sklearn.linear_model as lin

import particles
from particles import binary_smc as bin
from particles import datasets
from particles import distributions as dists
from particles import smc_samplers as ssps

data_name = 'boston'
dataset = datasets.Boston()
names = dataset.predictor_names
raw = dataset.raw_data  # do NOT rescale predictors
n, p = raw.shape
response = np.log(raw[:, -1])

cols = {'intercept': np.ones(n)}
for i, k in enumerate(names):
    cols[k] = raw[:, i]
    # squares
    if k != 'CHAS':
        cols['%s^2' % k] = cols[k]**2
    # interactions
    for j in range(i):
        k2 = names[j]
        cols[f'{k} x {k2}'] = cols[k] * cols[k2]

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

prior = dists.IID(bin.Bernoulli(0.5), npreds)
model = bin.BayesianVS(data=data, prior=prior)

N = 10**5
P = 1_000
M = N // P
nruns = 3
move = ssps.MCMCSequenceWF(mcmc=bin.BinaryMetropolis(), len_chain=P)
fk = ssps.AdaptiveTempering(model, len_chain=P, move=move)
results = particles.multiSMC(fk=fk, N=M, verbose=True, nruns=nruns, nprocs=0)

# save results
mp = [np.average(r['output'].X.theta, axis=0, weights=r['output'].W)
      for r in results]
to_save = {'marg_probs': np.array(mp),
           'pred_names': list(cols.keys())}
with open(f'{data_name}.pkl', 'wb') as f:
    pickle.dump(to_save, f)
