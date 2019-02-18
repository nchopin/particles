#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Plot the contours of the log-likelihood of a theta-logistic model; 
see Figure 13.2 in the MLE Chapter for more details. 

Notes: this script uses multi-processing to speed things up; this may not work
on certain OSes (see documentation). Set option nprocs to 1 to disable 
multiprocessing. 
"""
from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import particles
from particles import state_space_models as ssms

# Model 
def fkmod(**kwargs):
    return ssms.GuidedPF(ssm=ssms.ThetaLogistic(**kwargs), data=data)

# data 
T = 100
refresh_data = False 
if refresh_data:
    data, _ = ssms.ThetaLogistic().simulate(T)
else:
    data = np.loadtxt('simulated_data_ecological.txt')
plt.figure()
plt.plot(data)
plt.title('data') 

ng = 300
tau1_grid = np.linspace(0., 0.4, ng)
tau2_grid = np.linspace(0., 1.8, ng)
tau1s, tau2s = np.meshgrid(tau1_grid, tau2_grid)
models = [fkmod(tau1=tau1, tau2=tau2)
          for tau1, tau2 in zip(tau1s.flat, tau2s.flat)]

results = particles.multiSMC(fk=models, N=10**4, qmc=True, nruns=1,
                          nprocs=0, # multiprocessing
                          out_func= lambda pf: pf.logLt)
ll = np.array([r['output'] for r in results])
ll.shape = (ng, ng)

# PLOT
# ====
savefigs = False
plt.style.use('default')
plt.rc('text', usetex=True)

plt.figure()
levels = ll.max() + np.linspace(-20, 0, 21)
ctour = plt.contourf(tau1s, tau2s, ll, levels, cmap=cm.gray_r, alpha=.8)

# The two lines below are some black magic that ensures that the contour 
# plot looks well when turned into a pdf (otherwise, the contour lines 
# do not appear anymore; something to do with aliasing)
for c in ctour.collections:
    c.set_edgecolor("face")

plt.xlabel(r'$\tau_1$')
plt.ylabel(r'$\tau_2$')

# add a black dot to locate MLE
tau1_argmax = tau1s.flatten()[ll.argmax()]
tau2_argmax = tau2s.flatten()[ll.argmax()]
plt.plot(tau1_argmax, tau2_argmax, 'ok') 
if savefigs:
    plt.savefig('contour_loglik_ecological.pdf')

plt.show()
