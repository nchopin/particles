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
np.random.seed(1)
# setup of filtering problem
def atleast_2d_last(x):
    """Ensure x has at least 2 dims, adding a new axis at the end if needed."""
    x = np.array(x, copy=False)
    if x.ndim < 2:
        return x[..., np.newaxis]
    return x
colors = ["tab:blue", "tab:green", "tab:orange"]

class Lorenz_63(enk.MVNonlinearGauss):
  def F(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
      x = xp[:,[0]]
      y = xp[:,[1]]
      z = xp[:,[2]]
      return xp + self.dt*np.hstack([self.sigma*(y-x), x*(self.rho - z)-y, x*y-self.beta*z])
  def G(self, t, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
      return x[:,[0]]#return dists.Normal(loc=x[:,0], scale=self.obsnoise)
  
  

# my_model = Lorenz_63(rho=28., sigma=10., beta=8./3, dt=0.01, covX = np.eye(3), covY = 1.*np.eye(1))  # actual model

my_model = enk.MVNonlinearGauss(F=lambda t, x: 0.9*x, G=lambda t, x: x+1*np.exp(x), covX=1., covY=1.5, mu0=None, cov0=None)
# my_model = ssm.StochVol()
# toymodel = enk.MVNonlinearGauss(F=lambda x: x, G=lambda x: np.exp(x), covX=1., covY=.1, mu0=None, cov0=None)
true_states, data = my_model.simulate(2) #200  # we simulate from the model 100 data points

J = 500000 # size of ensembles

plt.figure()
plt.subplot(211)
plt.plot(np.vstack(true_states))
plt.subplot(212)
plt.plot(np.vstack(data))


#%% 

if len(data) == 1:
  x0s = np.linspace(-4,4,1000).reshape((-1,1))
  px0s = np.exp(my_model.PX0().logpdf(x0s))
  plt.figure()
  plt.plot(x0s, px0s)
  likes = np.exp(my_model.PY(0,None,x0s).logpdf(data[0]))
  plt.plot(x0s, likes)
  post = px0s*likes
  post /= np.trapz(post, x=x0s.flatten())
  plt.plot(x0s, post)

if len(data) == 2:
  x0s = np.linspace(-4,4,1000).reshape((-1,1))
  px0s = np.exp(my_model.PX0().logpdf(x0s)).squeeze()
  likes0 = np.exp(my_model.PY(0,None,x0s).logpdf(data[0])).squeeze()
  post0 = px0s*likes0
  post0 /= np.trapz(post0, x=x0s.flatten())
  kernel = np.exp(np.array([my_model.PX(1, x0).logpdf(x0s) for x0 in x0s])).squeeze()
  # plt.matshow(kernel)
  prior1 = kernel@post0
  prior1 /= np.trapz(prior1, x=x0s.flatten())
  likes1 = np.exp(my_model.PY(1,None,x0s).logpdf(data[1])).squeeze()
  post1 = prior1*likes1
  post1 /= np.trapz(post1, x=x0s.flatten())
  
  plt.plot(x0s, post1)

else:
  x0s = np.linspace(-4,4,1000).reshape((-1,1))
  prior = np.exp(my_model.PX0().logpdf(x0s)).flatten()
  for t in range(len(data)):
    likes = np.exp(my_model.PY(t,None,x0s).logpdf(data[t])).squeeze()
    post = prior*likes
    post /= np.trapz(post, x=x0s.flatten())
    if t < len(data) - 1:
      kernel = np.exp(np.array([my_model.PX(t, x0).logpdf(x0s) for x0 in x0s])).squeeze()
      prior = kernel@post
      prior /= np.trapz(prior, x=x0s.flatten())
    
  

#%% 
algorithms = [enk.WEnKF, enk.NudgedPF, enk.EnsembleKalman, ssm.Bootstrap]
alg_titles = ["WEnKF", "NudgedPF", "EnKF", "Bootstrap"]
# algorithms = [enk.EnsembleKalman, ssm.Bootstrap]
# alg_titles = ["EnKF", "Bootstrap"]
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
        if means.shape[0] > 1:
          plt.plot(means[:,m], color=colors[m])
          plt.fill_between(range(len(data)), y1=means[:,m]-2*np.sqrt(var[:,m]), y2=means[:,m]+2*np.sqrt(var[:,m]), color=colors[m], alpha=0.3)
        else:
          plt.errorbar(y=0, x=true_states[0], xerr=[means[:,m]-2*np.sqrt(var[:,m]),means[:,m]+2*np.sqrt(var[:,m])],
            capsize=5,
            ecolor="lightgrey",
            markerfacecolor="black",
            markeredgecolor="black",
            marker="o",
            linestyle="none",)
          plt.ylim([-0.5,0.5])
      plt.plot(np.vstack(true_states), "k--")
      plt.title(alg_titles[n_alg])
    
    MSEs_mean[n_alg][nMC] = np.sqrt(np.sum((np.vstack(true_states) - means)**2))
    
    
    coverage[n_alg][nMC] = [np.mean((np.vstack(true_states) >= means-nn*np.sqrt(var)) & (np.vstack(true_states) <= means+nn*np.sqrt(var))) for nn in [1,2,3]]
  
MSEs_mean = np.array(MSEs_mean)
coverage = np.array(coverage)
plt.tight_layout()

#%%
if len(data) == 1:
  plt.figure(figsize=(6,6))
  for n_alg, alg in enumerate(algorithms):
      fk = alg(my_model, data)
      filt = particles.SMC(fk=fk, N=J, collect=[Moments()], store_history=True) 
      filt.run()
    
      plt.subplot(len(algorithms),1,n_alg+1)
      plt.hist(filt.X, 15, weights=filt.W, density=True)
      print(f"range of samples in {alg}: {np.min(filt.X)} -- {np.max(filt.X)}")
      plt.plot(x0s, post, 'k--')
      plt.xlim([0,2.5])
      plt.title(alg_titles[n_alg])
    
      colors = ["tab:blue", "tab:green", "tab:orange"]

  plt.tight_layout()
  
if len(data) == 2:
  plt.figure(figsize=(6,6))
  for n_alg, alg in enumerate(algorithms):
      fk = alg(my_model, data)
      filt = particles.SMC(fk=fk, N=J, collect=[Moments()], store_history=True) 
      filt.run()
    
      plt.subplot(len(algorithms),1,n_alg+1)
      # plt.hist(filt.X, 150, density=True)
      plt.hist(filt.X, 150, weights=filt.W, density=True)
      print(f"range of samples in {alg}: {np.min(filt.X)} -- {np.max(filt.X)}")
      plt.plot(x0s, post1, 'k--')
      plt.xlim([-2,2.5])
      plt.title(alg_titles[n_alg])
    
      colors = ["tab:blue", "tab:green", "tab:orange"]

  plt.tight_layout()

else:
  plt.figure(figsize=(6,6))
  for n_alg, alg in enumerate(algorithms):
      fk = alg(my_model, data)
      filt = particles.SMC(fk=fk, N=J, collect=[Moments()], store_history=True) 
      filt.run()
    
      plt.subplot(len(algorithms),1,n_alg+1)
      plt.hist(filt.X, 150, weights=filt.W, density=True)
      plt.plot(x0s, post, 'k--')
      plt.xlim([-2,2.5])
      plt.title(alg_titles[n_alg])
    
      colors = ["tab:blue", "tab:green", "tab:orange"]
  
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
plt.title("MSE")
plt.boxplot(MSEs_mean.T, positions=range(len(algorithms)))
plt.xticks(range(len(algorithms)), alg_titles)