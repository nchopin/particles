#!/usr/bin/env python

"""
Compare performance of SMC and SQMC on the popular toy example of Gordon et
al (1993).

For more details, see the numerical section of Chapter 13 (SQMC) of the book;
in particular Figure 13.4.

"""


from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import particles
from particles import state_space_models as ssms
from particles.collectors import Moments

# instantiate model
T = 100
model = ssms.Gordon_etal()
_, data = model.simulate(T)
fk = ssms.Bootstrap(ssm=model, data=data)

if __name__ == "__main__":
    # Actual computation
    Ns = [2**k for k in range(6, 21)]
    of = lambda pf: {'ll': pf.logLt,
                     'EXt': [m['mean'] for m in pf.summaries.moments]}
    results = particles.multiSMC(fk=fk, qmc={'smc': False, 'sqmc': True}, N=Ns,
                                 collect=[Moments], nruns=200, nprocs=0,
                                 out_func=of)
    drez = {'smc': [r for r in results if r['qmc'] == 'smc'],
            'sqmc': [r for r in results if r['qmc'] == 'sqmc']
           }

    # Plots
    # =====
    savefigs = True  # False if you don't want to save plots as pdfs
    plt.rc('text', usetex=True)  # to force tex rendering
    plt.style.use('ggplot')

    plt.figure()
    colors = {'smc': 'gray', 'sqmc': 'black'}
    lsts = {'smc': '--', 'sqmc': '-'}
    for m in ['smc', 'sqmc']:
        var = [np.var([r['ll'] for r in drez[m] if r['N']==N])
            for N in Ns]
        plt.plot(Ns, var, color=colors[m], linestyle=lsts[m], linewidth=2, label=m)
    plt.legend(loc=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N$')
    plt.ylabel('var log-likelihood')
    if savefigs:
        plt.savefig('sqmc_Gordon_var_vs_N_loglik.pdf')

    plt.figure()
    gains = {}
    N_pow = [10, 20]
    fmt = {10: 'k:', 20: 'k'}
    for k in N_pow:
        N = 2**k
        var_filt = {}
        for m, dr in drez.items():
            var_filt[m] = np.var([r['EXt'] for r in dr if r['N']==N], axis=0)
        gains[k] = var_filt['smc'] / var_filt['sqmc']
        plt.plot(list(range(T)), gains[k], fmt[k], label=r'$N=2^{%i}$' % k)
    plt.xlabel(r'$t$')
    plt.ylabel('gain filtering expectation')
    plt.yscale('log')
    plt.legend(loc=1)
    if savefigs:
        plt.savefig('sqmc_Gordon_gain_vs_t.pdf')

    plt.show()
