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
np.random.seed(2)
# setup of filtering problem

colors = ["tab:blue", "tab:green", "tab:orange"]
# class toymodel(ssm.StateSpaceModel):
#     def PX0(self):  # Distribution of X_0
#         return dists.Normal()
#     def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
#         return dists.Normal(loc=xp)
#     # def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
#     #     return dists.MvNormal(loc=x, scale=self.obsnoise)
#     def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
#         return dists.Normal(loc=np.exp(x), scale=self.obsnoise)



# my_model = toymodel(obsnoise=1.0)  # actual model
# true_states, data = my_model.simulate(1)  # we simulate from the model 100 data points

# J = 1000 # size of ensembles

# plt.figure()
# plt.subplot(211)
# plt.plot(0, np.vstack(true_states), 'x')

# plt.subplot(212)
# plt.plot(0, np.vstack(data), 'x')


# true_post_dens = lambda x: np.exp(my_model.PX0().logpdf(x) + my_model.PY(0, None, x).logpdf(data[0]))
# xplot = np.linspace(-2,2,200)
# dens = true_post_dens(xplot)
# dens /= np.trapz(dens, x=xplot)



toymodel = enk.MVNonlinearGauss(F=lambda t, x: x, G=lambda t, x: np.exp(x), covX=1., covY=.1, mu0=None, cov0=None)
  
# my_model = toymodel(obsnoise=1.0)  # actual model
true_states, data = toymodel.simulate(1)  # we simulate from the model 100 data points

J = 100000 # size of ensembles

# plt.figure()
# plt.subplot(211)
# plt.plot(0, np.vstack(true_states), 'x')

# plt.subplot(212)
# plt.plot(0, np.vstack(data), 'x')

xplot = np.linspace(-3,3,200).reshape((-1,1))
true_post_dens = lambda x: np.exp(toymodel.PX0().logpdf(x) + toymodel.PY(0, None, x).logpdf(data[0]))
prior_grid = toymodel.PX0().logpdf(xplot)
dens = true_post_dens(xplot)
dens /= np.trapz(dens, x=xplot.squeeze())
#%% test nudged PF

# def main():
#   fk_nudged = enk.WEnKF(my_model, data)
#   npf = particles.SMC(fk=fk_nudged, N=J, collect=[Moments()], store_history=True) 
#   #
#   npf.run()
fk_nudged = enk.WEnKF(toymodel, data)
npf = particles.SMC(fk=fk_nudged, N=J, collect=[Moments()], store_history=True) 
npf.run()
# cProfile.run('main()', sort="tottime")
npf_path = np.stack(npf.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means_npf = np.stack([m['mean'] for m in npf.summaries.moments])
var_npf = np.stack([m['var'] for m in npf.summaries.moments])

plt.figure()
plt.subplot(331)
# for m in range(1):
#   plt.plot(means_npf, color=colors[m])
#   plt.fill_between(range(len(data)), y1=means_npf-2*np.sqrt(var_npf), y2=means_npf+2*np.sqrt(var_npf), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("nudged particle filter")

plt.hist(npf_path.squeeze(), 100, density=True)
plt.plot(xplot, dens, '--')
# plt.xlim(-3,3)
# plt.ylim([0,2])
plt.subplot(332)
plt.hist(npf_path.squeeze(), 100, weights=npf.wgts.W, density=True)
plt.plot(xplot, dens, '--')
# plt.xlim(-3,3)
# plt.ylim([0,2])
# ensemble Kalman

plt.subplot(333)
plt.semilogy(np.sort(npf.wgts.W))

print(f"ESS Weighted Ensemble Kalman filter = {1/np.sum(npf.wgts.W**2)}")
fk_EK = enk.EnsembleKalman(toymodel, data)
ek = particles.SMC(fk=fk_EK, N=J, collect=[Moments()], store_history=True) 
ek.run()

ek_path = np.stack(ek.hist.X)



means = np.stack([m['mean'] for m in ek.summaries.moments])
var = np.stack([m['var'] for m in ek.summaries.moments])

plt.subplot(334)
# for m in range(1):
#   plt.plot(means, color=colors[m])
#   plt.fill_between(range(len(data)), y1=means-2*np.sqrt(var), y2=means+2*np.sqrt(var), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("Ensemble Kalman filter")

plt.hist(ek_path.squeeze(), 100, density=True)
plt.plot(xplot, dens, '--')
# plt.xlim(-3,3)
# plt.ylim([0,2])

# Bootstrap particle filter
fk_model = ssm.Bootstrap(ssm=toymodel, data=data)
pf = particles.SMC(fk=fk_model, N=J, collect=[Moments()], resampling='stratified', store_history=True) 
pf.run()
particle_path = np.stack(pf.hist.X)


means_BP = np.stack([m['mean'] for m in pf.summaries.moments])
var_BP = np.stack([m['var'] for m in pf.summaries.moments])

plt.subplot(337)
# for m in range(1):
#   plt.plot(means_BP, color=colors[m])
#   plt.fill_between(range(len(data)), y1=means_BP-2*np.sqrt(var_BP), y2=means_BP+2*np.sqrt(var_BP), color=colors[m], alpha=0.3)
# plt.plot(np.vstack(true_states), "k--")
# plt.title("Bootstrap Particle filter (for comparison)")
# plt.tight_layout()
plt.hist(particle_path.squeeze(), 100, density=True)
plt.plot(xplot, dens, '--')
# plt.xlim(-3,3)
# plt.ylim([0,2])
plt.subplot(338)
plt.hist(particle_path.squeeze(), 100, weights=pf.wgts.W, density=True)
plt.plot(xplot, dens, '--')

# plt.xlim(-3,3)
# plt.ylim([0,2])



plt.subplot(339)
plt.semilogy(np.sort(pf.wgts.W))
plt.tight_layout()

print(f"ESS particle filter = {1/np.sum(pf.wgts.W**2)}")
