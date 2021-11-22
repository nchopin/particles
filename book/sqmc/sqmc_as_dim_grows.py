#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Illustrate how dimension affects performance of SQMC, by comparing four
algorithms:

    * bootstrap SMC
    * bootstrap SQMC
    * guided SMC
    * guided SQMC

applied to a certain class of Linear Gaussian models.

For more details, see either:

Chopin, N. and Gerber, M. (2018) Sequential quasi-Monte Carlo: Introduction for
Non-Experts, Dimension Reduction, Application to Partly Observed Diffusion
Processes. arXiv:1706.05305 (to be published in the MCQMC 2016 proceedings).

or the numerical section of Chapter 13 (SQMC) of the book, especially Figure 13.5.

"""

from __future__ import division, print_function

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy import stats

import particles
from particles import kalman
from particles import state_space_models as ssms
from particles.collectors import Moments

#parameter values
alpha0 = 0.3
T = 50
dims = range(5, 21, 5)

# instantiate models
models = OrderedDict()
true_loglik, true_filt_means = {}, {}
for d in dims:
    ssm = kalman.MVLinearGauss_Guarniero_etal(alpha=alpha0, dx=d)
    _, data = ssm.simulate(T)
    kf = kalman.Kalman(ssm=ssm, data=data)
    kf.filter()
    true_loglik[d] = np.cumsum(kf.logpyt)
    true_filt_means[d] = [f.mean for f in kf.filt]
    models['boot_%i' % d] = ssms.Bootstrap(ssm=ssm, data=data)
    models['guided_%i' % d] = ssms.GuidedPF(ssm=ssm, data=data)

# Get results
N = 10**4
results = particles.multiSMC(fk=models, qmc=[False, True], N=N,
                             collect=[Moments], nruns=100, nprocs=1)

# Format results
results_mse = []
for d in dims:
    for t in range(T):
        # this returns the estimate of E[X_t(1)|Y_{0:t}]
        estimate = lambda r: r['output'].summaries.moments[t]['mean'][0]
        for type_fk in ['guided', 'boot']:
            for qmc in [False, True]:
                est = np.array( [estimate(r) for r in results
                                 if r['qmc']==qmc and r['fk']==type_fk+'_%i'%d] )
                if type_fk=='guided' and qmc==False: #reference category
                    base_mean = np.mean(est)
                    base_mse = np.var(est)
                else:
                    mse = np.mean((est-base_mean)**2)
                    log10_gain = -np.log10(mse) + np.log10(base_mse)
                    results_mse.append( {'fk':type_fk, 'dim':d, 'qmc':qmc, 't':t,
                                         'log10_gain':log10_gain} )
# turn it into a pandas DataFrame
df = pd.DataFrame(results_mse)
df['fk_qmc'] = df['fk'] + df['qmc'].map({True:' SQMC', False:' SMC'})

# Plot
# ====
savefigs = True  # False if you don't want to save plots as pdfs
plt.rc('text', usetex=True) #to force tex rendering

plt.figure()
sb.set_style('darkgrid') #, {'axes.labelcolor': '.05'})
sb.set(font_scale=1.3)
plt.axhline(y=0., color='black', lw=2, zorder=0.8) # why 0.8, why?
ax = sb.violinplot(x="dim", y="log10_gain", hue="fk_qmc", data=df,
        hue_order=["boot SMC", "boot SQMC", "guided SQMC"],
        palette=sb.light_palette('black', 3, reverse=False))
plt.xlabel('dim')
plt.ylabel(r'gain for $E[X_t(1)|Y_{0:t}]$')
plt.legend(loc=1)
plt.ylim(-4, 4)
yt = list(range(-4, 5))
plt.yticks(yt, [r'$10^{%d}$' % i for i in yt])
if savefigs:
    plt.savefig('sqmc_as_dim_grows.pdf')

plt.show()
