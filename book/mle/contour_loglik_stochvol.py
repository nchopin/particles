#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot the contours of the log-likelihood of a stochastic volatility model;
see Figure 13.1 in the MLE Chapter for more details.

Notes: this script uses multi-processing to speed things up; this may not work
on certain OSes (see documentation). Replace 0 by 1 in line 41 below to discard
multiprocessing.
"""

from __future__ import division, print_function

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt

import particles
from particles import datasets as dts
from particles import state_space_models as ssms

# data
T = 200
data = dts.GBP_vs_USD_9798().data[:(T + 1)]


def fkmod(**kwargs):
    return ssms.Bootstrap(ssm=ssms.StochVol(**kwargs), data=data)


def logLt(pf):
    return pf.logLt


if __name__ == '__main__':
    sigma = 0.18
    ng = 100
    mu_grid = np.linspace(-2.5, 0., ng)
    rho_grid = np.linspace(-.999, .999, ng)
    mus, rhos = np.meshgrid(mu_grid, rho_grid)
    models = [fkmod(mu=mu, rho=rho, sigma=sigma)
              for mu, rho in zip(mus.flat, rhos.flat)]

    results = particles.multiSMC(fk=models, N=10 ** 4, qmc=True, nruns=1,
                                 nprocs=0,  #  multiprocessing!
                                 out_func=logLt)
    ll = np.array([r['output'] for r in results])
    ll.shape = (ng, ng)

    # PLOT
    # ====
    savefigs = True  # False if you don't want to save plots as pdfs

    plt.figure()
    levels = ll.max() + np.linspace(-20, 0, 21)
    ctour = plt.contourf(mus, rhos, ll, levels, cmap=cm.gray_r, alpha=.8)

    #  The two lines below are some black magic that ensures that the contour
    #  plot looks well when turned into a pdf (otherwise, the contour lines
    #  do not appear anymore; something to do with aliasing)
    for c in ctour.collections:
        c.set_edgecolor("face")

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\rho$')
    # show -1. and 1. limits even if they are a bit outside
    plt.yticks(np.linspace(-1., 1., 5))

    # add a black dot where the MLE is
    mu_max = mus.flatten()[ll.argmax()]
    rho_max = rhos.flatten()[ll.argmax()]
    plt.plot(mu_max, rho_max, 'ok')
    if savefigs:
        plt.savefig('contour_loglik_stochvol.pdf')

    plt.show()
