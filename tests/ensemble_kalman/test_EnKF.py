#!/usr/bin/env python

""" Checks that the Ensemble Kalman filter works on a simple Lorenz 63 model """



from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from scipy import stats

import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles import ensemble_kalman as enk
from particles.collectors import Moments

# setup of filtering problem

class Lorenz_63(ssm.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.MvNormal(loc=self.mu, scale=self.sigma0)
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        x = xp[:,[0]]
        y = xp[:,[1]]
        z = xp[:,[2]]
        new_mean = xp + self.dt*np.hstack([self.sigma*(y-x), x*(self.rho - z)-y, x*y-self.beta*z])
        return dists.MvNormal(loc=new_mean, scale=self.noise)
    # def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
    #     return dists.MvNormal(loc=x, scale=self.obsnoise)
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=x[:,0], scale=self.obsnoise)



my_model = Lorenz_63(mu=np.zeros(3), sigma0=1., rho=28., sigma=10., beta=8./3, noise = 0*0.1, obsnoise=10.0, dt=0.01)  # actual model
true_states, data = my_model.simulate(200)  # we simulate from the model 100 data points

J = 500 # size of ensembles

plt.figure()
plt.subplot(211)
plt.plot(np.vstack(true_states))
plt.subplot(212)
plt.plot(np.vstack(data))

#%% ensemble Kalman

fk_EK = enk.EnsembleKalman(my_model, data)
ek = particles.SMC(fk=fk_EK, N=J, collect=[Moments()], store_history=True) 
ek.run()

ek_path = np.stack(ek.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means = np.stack([m['mean'] for m in ek.summaries.moments])
var = np.stack([m['var'] for m in ek.summaries.moments])

plt.figure()
for m in range(3):
  plt.plot(means[:,m], color=colors[m])
  plt.fill_between(range(len(data)), y1=means[:,m]-2*np.sqrt(var[:,m]), y2=means[:,m]+2*np.sqrt(var[:,m]), color=colors[m], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")



#%% Bootstrap particle filter
fk_model = ssm.Bootstrap(ssm=my_model, data=data)
pf = particles.SMC(fk=fk_model, N=J, collect=[Moments()], resampling='stratified', store_history=True) 
pf.run()
particle_path = np.stack(pf.hist.X)


means = np.stack([m['mean'] for m in pf.summaries.moments])
var = np.stack([m['var'] for m in pf.summaries.moments])

plt.figure()
for m in range(3):
  plt.plot(means[:,m], color=colors[m])
  plt.fill_between(range(len(data)), y1=means[:,m]-2*np.sqrt(var[:,m]), y2=means[:,m]+2*np.sqrt(var[:,m]), color=colors[m], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
