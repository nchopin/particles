# -*- coding: utf-8 -*-

"""
Checks that our implementation of the Kalman filter returns 
the same results as pykalman, an external package available on PyPI. 
To install the latter:
    pip install pykalman

This does not generate any plot for the book. 

Recall the notations: 

X_0 ~ N(mu0, cov0)
X_t = F*X_{t-1} + U_t   U_t ~ N(0, covX)
Y_t = G*X_t + V_t       V_t ~ N(0, covY)  

"""

import numpy as np
from numpy import random

import pykalman  # pip install pykalman
from particles import kalman

# Multivariate model
T = 15
dx, dy = 3, 2  
F = np.eye(dx) + 0.01 * random.randn(dx, dx)
G = np.eye(dy, dx) + 0.02 * random.randn(dy, dx)
covX = np.eye(dx) + 0.2 * np.ones((dx, dx))
covY = 0.3 * np.eye(dy)
mu0 = np.ones(dx)
cov0 = 2. * covX 
mv_ssm = kalman.MVLinearGauss(F=F, G=G, covX=covX, covY=covY, cov0=cov0, mu0=mu0)

# Univariate model
rho, sigX, sigY, sig0 = 0.9, 1., 0.2, 3.
uni_ssm = kalman.LinearGauss(rho=rho, sigmaX=sigX, sigmaY=sigY, sigma0=sig0)

univariate = False # test univariate or multivariate model? 
if univariate:
    ssm = uni_ssm
    F, G, mu0 = rho, 1., 0.
    covX, covY, cov0 = sigX**2, sigY**2, sig0**2
else:
    ssm = mv_ssm 

# data
x, y = ssm.simulate(T)

# Our Kalman filter
mykf = kalman.Kalman(ssm=ssm, data=y)
mykf.smoother()  # this does both filtering and smoothing

# Their Kalman filter
theirkf = pykalman.KalmanFilter(transition_matrices = F, 
                                observation_matrices = G,
                                transition_covariance=covX, 
                                observation_covariance=covY, 
                                initial_state_mean=mu0,
                                initial_state_covariance=cov0) 
                
their_data = np.array(y).squeeze()
their_filt_means, their_filt_covs = theirkf.filter(their_data)
their_filt_means = their_filt_means.squeeze()

# Comparing filtering means and covs
my_fmeans = np.array([f.mean for f in mykf.filt]).squeeze()
my_fcovs = np.array([f.cov for f in mykf.filt]).squeeze()
err = abs(their_filt_means - my_fmeans).max()
print('max absolute error for filtering mean: %f'%err)
errc = abs(their_filt_covs - my_fcovs).max()
print('max absolute error for filtering cov: %f'%err)

# Comparing smoothing means and covs
their_smth_means, their_smth_covs = theirkf.smooth(their_data) 
their_smth_means = their_smth_means.squeeze()
my_smeans = np.array([s.mean for s in mykf.smth]).squeeze()
my_scovs = np.array([s.cov for s in mykf.smth]).squeeze()
err = abs(their_smth_means - my_smeans).max()
print('max absolute error for smoothing mean: %f'%err)
errc = abs(their_smth_covs - my_scovs).max()
print('max absolute error for smoothing cov: %f'%err)
