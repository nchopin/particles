#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numerical experiment of Chapter 17 (SMC samplers). 

Compare IBIS and SMC tempering for approximating: 
    * the normalising constant (marginal likelihood)
    * the posterior expectation of the p coefficients 
for a logistic regression model. 

See below for how to select the dataset. 
"""

from scipy import stats
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import seaborn as sb
# import statsmodels.api as sm

import particles
from particles import resampling as rs
from particles import distributions as dists
from particles import smc_samplers

# raw data
dataset = 'eeg'

if dataset == 'sonar':
    raw_data = np.loadtxt('../../datasets/sonar.all-data', delimiter=',', 
                          converters={60: lambda x: 1 if x ==b'R' else 0})
elif dataset == 'pima': 
    raw_data = np.loadtxt('../../datasets/pima-indians-diabetes.data',delimiter=',')
elif dataset == 'eeg': 
    raw_data = np.loadtxt('../../datasets/eeg_eye_state.data', delimiter=',', skiprows=19)

T, p = raw_data.shape
data = np.empty((T, p))
response = 2 * raw_data[:, -1] - 1  # 0/1 -> -1/1
preds = raw_data[:, :-1]

# normalise predictors (mean=0, sd=0.5)
data[:, 0] = 1. #intercept
data[:, 1:] = 0.5 * (preds - np.mean(preds, axis=0)) / np.std(preds, axis=0)

# flip signs according to response 
data *= response[:, np.newaxis]

# prior & model
prior = dists.StructDist({'beta':dists.MvNormal(scale=5., 
                                                  cov=np.eye(p))})

class LogisticRegression(smc_samplers.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta['beta'], data[t, :])
        return - np.log(1. + np.exp(-lin))

# algorithms 
N =  10 ** 3
Ms = [5, 10, 15]
ESSrmin = 0.5
nruns = 16
results = [] 

# runs
print('Dataset: %s' % dataset)
for M in Ms:
    for i in range(nruns):
        # need to shuffle the data for IBIS
        random.shuffle(data)
        model = LogisticRegression(data=data, prior=prior)
        for alg_type in ['tempering', 'ibis']: 
            if alg_type=='ibis': 
                fk = smc_samplers.IBIS(model, mh_options={'nsteps': M})
                pf = particles.SMC(N=N, fk=fk, ESSrmin=ESSrmin, 
                                moments=True, verbose=False)
            else:
                fk = smc_samplers.AdaptiveTempering(model, ESSrmin=ESSrmin, 
                                                    mh_options={'nsteps': M})
                pf = particles.SMC(N=N, fk=fk, ESSrmin=1., moments=True, 
                                verbose=True)
                # must resample at every time step when doing adaptive
                # tempering
            print('%s, M=%i, run %i' % (alg_type, M, i))
            pf.run()
            print('CPU time (min): %.2f' % (pf.cpu_time / 60))
            print('loglik: %f' % pf.logLt)
            res = {'M': M, 'type': alg_type, 'out': pf.summaries,
                   'cpu': pf.cpu_time}
            if alg_type=='ibis':
                n_eval = N * (T + M * sum([t for t in range(T) if
                                           pf.summaries.rs_flags[t]]))
            else:
                n_eval = N * T * (1. + M * (len(pf.summaries.ESSs) - 1))
                res['path_sampling'] = pf.X.path_sampling[-1]
            res['n_eval'] = n_eval
            results.append(res)

# save results
##############
# import pickle
# pickle.dump({'results':results, 'Ms': Ms, 'N': N, 'nruns': nruns, 'T': T, 'p':p, 
#              'ESSrmin' : ESSrmin, 'data': data}, 
#             open('%s_N%i.pickle' % (dataset, N), 'wb'))
# Then reload like this
# dp = pickle.load(file_name)
# locals().update(dp)

# plots
#######

savefigs = False  # do you want to save figures as pdfs

plt.style.use('ggplot')
pal = sb.dark_palette('white', n_colors=3)

# plt.figure()
# diff_est = [(r['out'].logLt - r['path_sampling'])
#             for r in results if r['type']=='tempering']
# plt.hist(diff_est)
# plt.xlabel('norm constant: ratio estimate minus path sampling estimate')


plt.figure()
sb.boxplot(x=[r['M'] for r in results],
           y=[r['out'].logLts[-1] for r in results], 
           hue=[r['type'] for r in results],
           palette=pal)
plt.xlabel('nb MCMC steps')
plt.ylabel('marginal likelihood')
if savefigs:
    plt.savefig(dataset + 'boxplots_marglik_vs_M.pdf')


plt.figure()
sb.boxplot(x=[r['M'] for r in results],
           y=[r['out'].moments[-1]['mean']['beta'][1] for r in results],
           hue=[r['type'] for r in results], 
           palette=pal)
plt.xlabel('nb MCMC steps')
plt.ylabel('posterior expectation first predictor')
if savefigs:
    plt.savefig(dataset + 'boxplots_postexp1_vs_M.pdf')


plt.figure()
cols = {'ibis': 'gray', 'tempering':'black'}
for i in range(p):
    for alg_type in ['ibis', 'tempering']:
        adj_var = []
        for M in Ms:
            mts = [r['out'].moments[-1] 
                   for r in results if r['M']==M and r['type']==alg_type]
            av = (M * np.var([m['mean']['beta'][i] for m in mts]) /
                             np.mean([m['var']['beta'][i] for m in mts]))
            adj_var.append(av)
        if i==0:
            plt.plot(Ms, adj_var, color=cols[alg_type], label=alg_type,
                     alpha=.8)
        else:
            plt.plot(Ms, adj_var, color=cols[alg_type], alpha=.8)
plt.legend()
plt.xlabel('nb MCMC steps')
plt.ylabel(r'variance times nb MCMC steps')
if savefigs:
    plt.savefig(dataset + 'postexp_var_vs_M.pdf')


