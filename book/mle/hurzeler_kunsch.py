#!/usr/bin/env python

"""
Plots Figure 13.5 which illustrates Hürzeler and Künsch's method for using CRN
(common random numbers) when evaluating the log-likelihood function.  See
Chapter 13 (MLE) for more details.

Note: takes less than one minute.
"""


import numpy as np
from matplotlib import pyplot as plt

import particles
from particles import datasets as dts
from particles import resampling as rs
from particles import state_space_models as ssms

# data
data = dts.GBP_vs_USD_9798().data

# ssm model
def ssmod(theta):
    mu = theta[0]
    rho = theta[1]
    sigma = theta[2]
    return ssms.StochVol(mu=mu, rho=rho, sigma=sigma)


def log_joint_density(theta, x):
    mod = ssmod(theta)
    l = mod.PX0().logpdf(x[0])
    for t, xt in enumerate(x):
        if t == 0:
            l = mod.PX0().logpdf(xt) + mod.PY(0, None, x[t]).logpdf(data[t])
        else:
            l += (mod.PX(t, x[t - 1]).logpdf(x[t])
                  + mod.PY(t, x[t - 1], x[t]).logpdf(data[t]))
    return l

# FK models
def fkmod(theta, T):
    return ssms.Bootstrap(ssm=ssmod(theta), data=data[:T])

# Choice of theta_0 and range of theta's
mu0 = -1.
rho0 = 0.9
sigma0 = 0.3
theta0 = [mu0, rho0, sigma0]
sigmas = sigma0 + np.linspace(-.199, .2, 401)
thetas = [[mu0, rho0, sig] for sig in sigmas]

# range of T's
Ts = [10, 100, 1000]
colors = {10: 'lightgray', 100: 'gray', 1000: 'black'}
plt.style.use('ggplot')
plt.figure()
for T in Ts:
    print('FFBS for T=%i' % T)
    alg = particles.SMC(fk=fkmod(theta0, T), N=100, store_history=True)
    alg.run()
    trajs = alg.hist.backward_sampling(M=100)
    ll0 = log_joint_density(theta0, trajs)
    ess_ls = []
    for theta in thetas:
        ll = log_joint_density(theta, trajs)
        ess = rs.essl(ll - ll0)
        ess_ls.append(ess)
    plt.plot(sigmas, ess_ls, label='T=%i' % T, color=colors[T])

plt.xlabel('sigma')
plt.ylabel('ESS')
plt.legend(loc=2)

savefigs = True  # False if you don't want to save plots as pdfs
if savefigs:
    plt.savefig('hurzeler_kunsch.pdf')

plt.show()
