#!/usr/bin/env python

"""
Comparison of resampling schemes (Fig. 9.2 in resampling Chapter): 

    plots TV distance, as a function of tau, between the weighted
    sample and the resampled sample, where: 
    * particles are ~ N(0, 1) 
    * weights are exp(- (tau / 2) * (x - b)^2), with b=1. 
    * sample size is N=10^4 
    for four resampling schemes (average over 100 executions)
    
    Note: takes about 8 min on my laptop. 
"""


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from scipy import stats

from particles import resampling as rs

N = 10**4  # number of samples
ntrials = 100
taus = np.linspace(0., 10, 500)
rs_schemes = ['multinomial', 'residual', 'stratified', 'systematic']
bias = 1.


def tv_distance(x, y):
    """ TV distance between two discrete distributions. 

    x, y: the weights
    """
    return 0.5 * sum(abs(x - y))

results = {key: np.zeros((ntrials, len(taus))) for key in rs_schemes}

for i in range(ntrials):
    x = stats.norm.rvs(size=N)
    for j, tau in enumerate(taus):
        lw = -.5 * tau * (bias - x)**2
        W = rs.exp_and_normalise(lw)
        for scheme in rs_schemes:
            A = rs.resampling(scheme, W)
            counts = np.bincount(A, minlength=N)
            # counts start at 0
            results[scheme][i, j] = tv_distance(W, counts / N)

# PLOTS
# =====
savefigs = True
plt.style.use('ggplot')
sb.set_palette(sb.dark_palette("lightgray", n_colors=4, reverse=True))

# Actual figure
plt.figure()
for k, scheme in enumerate(rs_schemes):
    plt.plot(taus, np.mean(results[scheme], axis=0), label=scheme,
             linewidth=3)
plt.legend()
plt.xlabel('tau')
plt.ylabel('TV distance')
if savefigs:
    plt.savefig('resampling_comparison.pdf')

# 80% confidence intervals (not very interesting, as variance is very small
plt.figure()
col = {'multinomial': 'red', 'residual': 'green', 'stratified': 'yellow',
       'systematic': 'black'}
for k, scheme in enumerate(rs_schemes):
    plt.fill_between(taus, np.percentile(results[scheme], 0.90, axis=0),
                     np.percentile(results[scheme], 0.10, axis=0),
                     facecolor=col[scheme])

plt.show()
