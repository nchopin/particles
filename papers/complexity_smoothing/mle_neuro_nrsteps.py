#!/usr/bin/env python

r"""
Impact of number of MCMC steps in FFBS-MCMC when used with the EM algorithm; 
based on the Numerical experiment of Chapter 14 (maximum likelihood estimation)
in the book. 

See mle_neuro.py for more details. Work in progress. 
"""


import time

from matplotlib import pyplot as plt
import seaborn as sb

import particles

import mle_neuro

def worker(rho0=0.1, sig20=0.5, N=10, ffbs='mcmc', maxiter=1, mcmc_steps=1):
    tic = time.perf_counter()
    results = mle_neuro.EM(rho0, sig20, N=N, maxiter=maxiter, xatol=0., 
                           ffbs=ffbs, mcmc_steps=mcmc_steps)
    results['cpu_time'] = time.perf_counter() - tic
    results['ffbs'] = ffbs
    results['mcmc_steps'] = mcmc_steps
    return results

nruns = 100
maxiter = 20
N = 1000
nsteps = [1, 3]

results = particles.utils.multiplexer(f=worker, N=N, maxiter=maxiter,
                                      mcmc_steps=nsteps, nruns=nruns, nprocs=1)
ref_method = 'hybrid'  # replace by 'pureject' to see improvement over book xp
ref_results = particles.utils.multiplexer(f=worker, N=N, maxiter=maxiter,
                                          ffbs=ref_method, nruns=nruns, nprocs=1)
results.extend(ref_results)

flat_res = []
for r in results:
    for i in range(maxiter):
        res_dict = {k: r[k + 's'][i] for k in ['rho', 'sig2', 'll']}
        method = r['ffbs']
        if method == 'mcmc':
            nsteps = r['mcmc_steps']
            method += f' {nsteps} steps'
        res_dict['method'] = method
        res_dict['iter'] = i 
        flat_res.append(res_dict)

skip_iter0 = True  # remove iteration 0 from plots?
flat_res_no_iter0 = [r for r in flat_res if r['iter'] > 0]
plt_res = flat_res_no_iter0 if skip_iter0 else flat_res

# PLOTS
#######
plt.style.use('ggplot')
savefigs = True

plt.figure()
sb.boxplot(y=[r['rho'] for r in plt_res], 
           x=[r['iter'] for r in plt_res],
           hue=[r['method'] for r in plt_res])
plt.ylabel(r'$\rho$')
plt.xlabel('EM iter')
if savefigs:
    plt.savefig('em_rho_iter_vs_nsteps.pdf')

plt.figure()
sb.boxplot(y=[r['sig2'] for r in plt_res], 
           x=[r['iter'] for r in plt_res],
           hue=[r['method'] for r in plt_res])
plt.ylabel(r'$\sigma^2$')
plt.xlabel('EM iter')
if savefigs:
    plt.savefig('em_sig2_iter_vs_nsteps.pdf')

plt.figure()
sb.boxplot(y=[r['ll'] for r in plt_res], 
           x=[r['iter'] for r in plt_res],
           hue=[r['method'] for r in plt_res]
          )
plt.xlabel('EM iter')
plt.ylabel('log-lik')
if savefigs:
    plt.savefig('em_ll_iter_vs_nsteps.pdf')
