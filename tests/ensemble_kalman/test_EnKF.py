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
def atleast_2d_last(x):
    """Ensure x has at least 2 dims, adding a new axis at the end if needed."""
    x = np.array(x, copy=False)
    if x.ndim < 2:
        return x[..., np.newaxis]
    return x
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
  
  

# my_model = Lorenz_63(rho=28., sigma=10., beta=8./3, dt=0.01, covX = 1.0*np.eye(3), covY = 1.0*np.eye(1))  # actual model
my_model = ssm.StochVol()
# toymodel = enk.MVNonlinearGauss(F=lambda x: x, G=lambda x: np.exp(x), covX=1., covY=.1, mu0=None, cov0=None)
true_states, data = my_model.simulate(200)  # we simulate from the model 100 data points

J = 150 # size of ensembles

plt.figure()
plt.subplot(211)
plt.plot(np.vstack(true_states))
plt.subplot(212)
plt.plot(np.vstack(data))


# #%% test nudged PF
# fk_WEK = enk.WEnKF(my_model, data)
# wek = particles.SMC(fk=fk_WEK, N=J, collect=[Moments()], store_history=True) 
# wek.run()

# wek_path = np.stack(wek.hist.X)

# colors = ["tab:blue", "tab:green", "tab:orange"]


# means_wek = np.stack([m['mean'] for m in wek.summaries.moments])
# var_wek = np.stack([m['var'] for m in wek.summaries.moments])

# plt.figure(figsize=(6,6))
# plt.subplot(411)
# for m in range(3):
#   plt.plot(means_wek[:,m], color=colors[m])
#   plt.fill_between(range(len(data)), y1=means_wek[:,m]-2*np.sqrt(var_wek[:,m]), y2=means_wek[:,m]+2*np.sqrt(var_wek[:,m]), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("WEnKF")

# fk_nudged = enk.NudgedPF(my_model, data)
# npf = particles.SMC(fk=fk_nudged, N=J, collect=[Moments()], store_history=True) 
# npf.run()

# npf_path = np.stack(npf.hist.X)

# colors = ["tab:blue", "tab:green", "tab:orange"]


# means_npf = np.stack([m['mean'] for m in npf.summaries.moments])
# var_npf = np.stack([m['var'] for m in npf.summaries.moments])

# # plt.figure(figsize=(6,6))
# plt.subplot(412)
# for m in range(3):
#   plt.plot(means_npf[:,m], color=colors[m])
#   plt.fill_between(range(len(data)), y1=means_npf[:,m]-2*np.sqrt(var_npf[:,m]), y2=means_npf[:,m]+2*np.sqrt(var_npf[:,m]), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("nudged particle filter")

# fk_EK = enk.EnsembleKalman(my_model, data)
# ek = particles.SMC(fk=fk_EK, N=J, collect=[Moments()], store_history=True) 
# # cProfile.run('ek.run()')
# ek.run()

# ek_path = np.stack(ek.hist.X)



# means = np.stack([m['mean'] for m in ek.summaries.moments])
# var = np.stack([m['var'] for m in ek.summaries.moments])

# plt.subplot(413)
# for m in range(3):
#   plt.plot(means[:,m], color=colors[m])
#   plt.fill_between(range(len(data)), y1=means[:,m]-2*np.sqrt(var[:,m]), y2=means[:,m]+2*np.sqrt(var[:,m]), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("Ensemble Kalman filter")


# # Bootstrap particle filter
# fk_model = ssm.Bootstrap(ssm=my_model, data=data)
# pf = particles.SMC(fk=fk_model, N=J, collect=[Moments()], resampling='stratified', store_history=True) 
# pf.run()
# particle_path = np.stack(pf.hist.X)


# means_BP = np.stack([m['mean'] for m in pf.summaries.moments])
# var_BP = np.stack([m['var'] for m in pf.summaries.moments])

# plt.subplot(414)
# for m in range(3):
#   plt.plot(means_BP[:,m], color=colors[m])
#   plt.fill_between(range(len(data)), y1=means_BP[:,m]-2*np.sqrt(var_BP[:,m]), y2=means_BP[:,m]+2*np.sqrt(var_BP[:,m]), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("Bootstrap Particle filter (for comparison)")
# plt.tight_layout()


#%% more systematically
# algorithms = [enk.WEnKF, enk.NudgedPF, enk.EnsembleKalman, ssm.Bootstrap]
# alg_titles = ["WEnKF", "NudgedPF", "EnKF", "Bootstrap"]
algorithms = [enk.EnsembleKalman, ssm.Bootstrap]
alg_titles = ["EnKF", "Bootstrap"]
plt.figure(figsize=(6,6))
N_MC = 25
MSEs_mean = [[None  for m in range(N_MC)] for n in range(len(algorithms))]
coverage = [[None  for m in range(N_MC)] for n in range(len(algorithms))]
for n_alg, alg in enumerate(algorithms):
  for nMC in range(N_MC):
    fk = alg(my_model, data)
    filt = particles.SMC(fk=fk, N=J, collect=[Moments()], store_history=True) 
    filt.run()
  
    filt_path = np.stack(filt.hist.X)
  
    colors = ["tab:blue", "tab:green", "tab:orange"]
  
  
    means =  atleast_2d_last(np.stack([m['mean'] for m in filt.summaries.moments]))
    var = atleast_2d_last(np.stack([m['var'] for m in filt.summaries.moments]))
    if nMC == 0:
      plt.subplot(len(algorithms),1,n_alg+1)
      for m in range(means.shape[1]):
        plt.plot(means[:,m], color=colors[m])
        plt.fill_between(range(len(data)), y1=means[:,m]-2*np.sqrt(var[:,m]), y2=means[:,m]+2*np.sqrt(var[:,m]), color=colors[m], alpha=0.3)
      plt.plot(np.vstack(true_states), "k--")
      plt.title(alg_titles[n_alg])
    
    MSEs_mean[n_alg][nMC] = np.sqrt(np.sum((np.vstack(true_states) - means)**2))
    
    
    coverage[n_alg][nMC] = [np.mean((np.vstack(true_states) >= means-nn*np.sqrt(var)) & (np.vstack(true_states) <= means+nn*np.sqrt(var))) for nn in [1,2,3]]
  
MSEs_mean = np.array(MSEs_mean)
coverage = np.array(coverage)
plt.tight_layout()
#%%
plt.figure()
color_quantile = ["tab:green", "tab:orange", "tab:red"]

for n_quantile in range(3):
  parts = plt.violinplot(coverage[:,:,n_quantile].T, positions=range(len(algorithms)))
  for pc in parts['bodies']:
    pc.set_facecolor(color_quantile[n_quantile])
    pc.set_edgecolor(color_quantile[n_quantile])
    
  parts['cmaxes'].set_colors("black")
  parts['cmaxes'].set_alpha(0.2)
  parts['cmins'].set_colors("black")
  parts['cmins'].set_alpha(0.2)
  parts['cbars'].set_colors("black")
  parts['cbars'].set_alpha(0.2)
  plt.plot(coverage[:,:,n_quantile], '.k')
plt.axhline(y = 0.68, label="$\\pm 1 \\sigma$", color = "tab:green")
plt.axhline(y = 0.95, label="$\\pm 2 \\sigma$", color = "tab:orange")
plt.axhline(y = 0.997, label="$\\pm 3 \\sigma$", color = "tab:red")
plt.xticks(range(len(algorithms)), alg_titles)
plt.legend()


plt.figure()