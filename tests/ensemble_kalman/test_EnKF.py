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

colors = ["tab:blue", "tab:green", "tab:orange"]
# class Lorenz_63_old(ssm.StateSpaceModel):
#     def PX0(self):  # Distribution of X_0
#         return dists.MvNormal(loc=self.mu, scale=self.sigma0)
#     def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
#         x = xp[:,[0]]
#         y = xp[:,[1]]
#         z = xp[:,[2]]
#         new_mean = xp + self.dt*np.hstack([self.sigma*(y-x), x*(self.rho - z)-y, x*y-self.beta*z])
#         return dists.MvNormal(loc=new_mean, scale=self.noise)
#     # def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
#     #     return dists.MvNormal(loc=x, scale=self.obsnoise)
#     def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
#         return dists.Normal(loc=x[:,0], scale=self.obsnoise)

class Lorenz_63(enk.MVNonlinearGauss):
  def F(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
      x = xp[:,[0]]
      y = xp[:,[1]]
      z = xp[:,[2]]
      return xp + self.dt*np.hstack([self.sigma*(y-x), x*(self.rho - z)-y, x*y-self.beta*z])
      # return dists.MvNormal(loc=new_mean, scale=self.noise)
  # def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
  #     return dists.MvNormal(loc=x, scale=self.obsnoise)
  def G(self, t, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
      return x[:,[0]]#return dists.Normal(loc=x[:,0], scale=self.obsnoise)
  
  

my_model = Lorenz_63(rho=28., sigma=10., beta=8./3, dt=0.01, covX = 1.0*np.eye(3), covY = 1.0*np.eye(1))  # actual model
# toymodel = enk.MVNonlinearGauss(F=lambda x: x, G=lambda x: np.exp(x), covX=1., covY=.1, mu0=None, cov0=None)
true_states, data = my_model.simulate(200)  # we simulate from the model 100 data points

J = 150 # size of ensembles

plt.figure()
plt.subplot(211)
plt.plot(np.vstack(true_states))
plt.subplot(212)
plt.plot(np.vstack(data))


#%% test nudged PF
fk_WEK = enk.WEnKF(my_model, data)
wek = particles.SMC(fk=fk_WEK, N=J, collect=[Moments()], store_history=True) 
wek.run()

wek_path = np.stack(wek.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means_wek = np.stack([m['mean'] for m in wek.summaries.moments])
var_wek = np.stack([m['var'] for m in wek.summaries.moments])

plt.figure(figsize=(6,6))
plt.subplot(411)
for m in range(3):
  plt.plot(means_wek[:,m], color=colors[m])
  plt.fill_between(range(len(data)), y1=means_wek[:,m]-2*np.sqrt(var_wek[:,m]), y2=means_wek[:,m]+2*np.sqrt(var_wek[:,m]), color=colors[m], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("WEnKF")

fk_nudged = enk.NudgedPF(my_model, data)
npf = particles.SMC(fk=fk_nudged, N=J, collect=[Moments()], store_history=True) 
npf.run()

npf_path = np.stack(npf.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means_npf = np.stack([m['mean'] for m in npf.summaries.moments])
var_npf = np.stack([m['var'] for m in npf.summaries.moments])

# plt.figure(figsize=(6,6))
plt.subplot(412)
for m in range(3):
  plt.plot(means_npf[:,m], color=colors[m])
  plt.fill_between(range(len(data)), y1=means_npf[:,m]-2*np.sqrt(var_npf[:,m]), y2=means_npf[:,m]+2*np.sqrt(var_npf[:,m]), color=colors[m], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("nudged particle filter")


fk_EK = enk.EnsembleKalman(my_model, data)
ek = particles.SMC(fk=fk_EK, N=J, collect=[Moments()], store_history=True) 
# cProfile.run('ek.run()')
ek.run()

ek_path = np.stack(ek.hist.X)



means = np.stack([m['mean'] for m in ek.summaries.moments])
var = np.stack([m['var'] for m in ek.summaries.moments])

plt.subplot(413)
for m in range(3):
  plt.plot(means[:,m], color=colors[m])
  plt.fill_between(range(len(data)), y1=means[:,m]-2*np.sqrt(var[:,m]), y2=means[:,m]+2*np.sqrt(var[:,m]), color=colors[m], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("Ensemble Kalman filter")


# Bootstrap particle filter
fk_model = ssm.Bootstrap(ssm=my_model, data=data)
pf = particles.SMC(fk=fk_model, N=J, collect=[Moments()], resampling='stratified', store_history=True) 
pf.run()
particle_path = np.stack(pf.hist.X)


means_BP = np.stack([m['mean'] for m in pf.summaries.moments])
var_BP = np.stack([m['var'] for m in pf.summaries.moments])

plt.subplot(414)
for m in range(3):
  plt.plot(means_BP[:,m], color=colors[m])
  plt.fill_between(range(len(data)), y1=means_BP[:,m]-2*np.sqrt(var_BP[:,m]), y2=means_BP[:,m]+2*np.sqrt(var_BP[:,m]), color=colors[m], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("Bootstrap Particle filter (for comparison)")
plt.tight_layout()

