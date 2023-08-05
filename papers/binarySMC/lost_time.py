"""Attempt at reproducing the numerical experiment of Griffin et al (2018). 


"""


import numpy as np
import sklearn.linear_model as lin

import particles
from particles import datasets
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssps
from particles import binary_smc as bin


dataset = datasets.Boston()
names = dataset.predictor_names
raw = dataset.raw_data  # do NOT rescale predictors
n, p = raw.shape
response = np.log(raw[:, -1])

# cols = {'intercept': np.ones(n)}
cols = {}
for i, k in enumerate(names):
    cols[k] = raw[:, i]
    # squares
    if k != 'CHAS':
        cols['%s^2' % k] = cols[k]**2
    # interactions
    for j in range(i):
        k2 = names[j]
        cols[f'{k} x {k2}'] = cols[k] * cols[k2]

rescale = True  # Jim says predictors were centered *and* scaled
def resc(x):
    return (x - x.mean()) / x.std()
if rescale:
    response = resc(response)
    for k in cols.keys():
        if k != 'intercept': # we removed the intercept anyway
            cols[k] = resc(cols[k])

preds = np.stack(list(cols.values()), axis=1)
npreds = len(cols)
data = preds, response

# compare with full regression
reg = lin.LinearRegression(fit_intercept=False)
reg.fit(preds, response)

prior = dists.IID(bin.Bernoulli(0.05), npreds)
model = bin.BayesianVS(data=data, prior=prior, nu=0., iv2=0.01, jitted=True)

# gam, l = model.complete_enum()
# probs = rs.exp_and_normalise(l)
# marg_probs = np.average(gam, weights=probs, axis=0)

# print(marg_probs)

waste = True
nruns = 30
if waste:
    ESSrmin = 0.5
    lc = 500
    N = 200
    move = ssps.MCMCSequenceWF(mcmc=bin.BinaryMetropolis(),
                                      len_chain=lc)
else:
    ESSrmin = 0.9
    lc = 10 + 1
    N = 4_000
    move = ssps.MCMCSequence(mcmc=bin.BinaryMetropolis(), len_chain=lc)

fk = ssps.AdaptiveTempering(model, ESSrmin=ESSrmin, len_chain=lc, 
                            wastefree=waste, move=move)
results = particles.multiSMC(fk=fk, N=N, verbose=True, nruns=nruns, nprocs=1)

ps = np.array([np.average(r['output'].X.theta, weights=r['output'].W, axis=0) 
               for r in results])
ph = ps.mean(axis=0)


# pf = particles.SMC(fk=fk, N=1000, verbose=True)
# pf.run()

# jitted=True, wall time = 1291 s (user 5444)
# jitted=mock true, wall time = 1150 s
# jitted=False, wall time = 1027 s (user 1487)
# IPython CPU timings (estimated):

 
