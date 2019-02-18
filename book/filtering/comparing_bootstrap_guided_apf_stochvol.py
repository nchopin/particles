#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare bootstrap, guided and APF filters on a stochastic volatility model. 

The proposal of the guided and auxiliary filter, and the auxiliary function 
of the APF, are based on the Taylor expansion trick of Pitt and Shephard
(1999).

A SQMC algorithm run with with N=10^5 is used as a baseline. 

See Section 10.4.2 and Figure 10.3 in the book for a discussion. 

""" 

from __future__ import division, print_function

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

import particles
from particles import state_space_models as ssms

# Data and parameter values from Pitt & Shephard
raw_data = np.loadtxt('../../datasets/GBP_vs_USD_9798.txt',
                      skiprows=2, usecols=(3,), comments='(C)')
T = 201
data = 100. * np.diff(np.log(raw_data[:(T + 1)]))
my_ssm = ssms.StochVol(mu=2 * np.log(.5992), sigma=0.178, rho=0.9702)

# FK models
models = OrderedDict()
models['bootstrap'] = ssms.Bootstrap(ssm=my_ssm, data=data)
models['guided'] = ssms.GuidedPF(ssm=my_ssm, data=data)
models['apf'] = ssms.AuxiliaryPF(ssm=my_ssm, data=data)

# Get results
results = particles.multiSMC(fk=models, N=10**3, nruns=250, moments=True)

# Golden standard
bigN = 10**5
bigpf = particles.SMC(fk=models['bootstrap'], N=bigN, qmc=True,
        moments=True)
print('One SQMC run with N=%i' % bigN)
bigpf.run()

# PLOTS
# =====
plt.style.use('ggplot')
savefig = False  # True if you want to save figs as pdfs

# box-plots for log-likelihood evaluation
plt.figure()
sb.boxplot(x=[r['fk'] for r in results],
           y=[r['output'].logLt for r in results]
           )
plt.ylabel('log-likelihood estimate')

# Std of particle estimate of E(X_t|Y_{0:t}) vs time
plt.figure()
artists = []
for mod, col in zip(['bootstrap', 'guided'], ['0.75', '0.25']):
    errors = np.array([[mom['mean'] - mom0['mean'] for mom, mom0 in
                     zip(r['output'].summaries.moments, bigpf.summaries.moments)]
                    for r in results if r['fk'] == mod])
    artist = plt.fill_between(np.arange(T), np.percentile(errors, 75, axis=0),
                              np.percentile(errors, 25, axis=0), facecolor=col, alpha=0.9)
    artists.append(artist)
how_many_APF_traj = 25  # change this if you want more/less trajectories
for r in [r for r in results if r['fk'] == 'apf'][:how_many_APF_traj]:
    errors = np.array([avg['mean'] - true['mean'] for avg, true in
                       zip(r['output'].summaries.moments, bigpf.summaries.moments)])
    plt.plot(errors, 'k', alpha=0.8)
plt.xlabel(r'$t$')
plt.ylabel('estimate error for ' + r'$E(X_t|Y_{0:t})$')
import matplotlib.lines as mlines
black_line = mlines.Line2D([], [], color='black')
artists.append(black_line)
plt.legend(artists, models.keys(), loc=3)
if savefig:
    plt.savefig('stochvol_filtering_error_boot_vs_guided_vs_apf.pdf')

# ESS vs time (from a single run)
plt.figure()
for model in models.keys():
    pf = next(r['output'] for r in results if r['fk'] == model)
    plt.plot(pf.summaries.ESSs, label=model)
plt.legend(loc=3)
plt.xlabel(r'$t$')
plt.ylabel('ESS')

# and finally
plt.show()
