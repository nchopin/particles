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



my_model = ssm.StochVol()
true_states, data = my_model.simulate(1000)  # we simulate from the model 100 data points

J = 500 # size of ensembles

plt.figure()
plt.subplot(211)
plt.plot(np.vstack(true_states))
plt.subplot(212)
plt.plot(np.vstack(data))




#%% test nudged PF
fk_nudged = enk.NudgedPF(my_model, data)
npf = particles.SMC(fk=fk_nudged, N=J, collect=[Moments()], store_history=True) 
npf.run()

npf_path = np.stack(npf.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means_npf = np.stack([m['mean'] for m in npf.summaries.moments])
var_npf = np.stack([m['var'] for m in npf.summaries.moments])


plt.figure()
plt.subplot(211)
plt.plot(means_npf, color=colors[0])
plt.fill_between(range(len(data)), y1=means_npf-2*np.sqrt(var_npf), y2=means_npf+2*np.sqrt(var_npf), color=colors[0], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("nudged particle filter")

#%% ensemble Kalman

fk_EK = enk.EnsembleKalman(my_model, data)
ek = particles.SMC(fk=fk_EK, N=J, collect=[Moments()], store_history=True) 
ek.run()

ek_path = np.stack(ek.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means = np.stack([m['mean'] for m in ek.summaries.moments])
var = np.stack([m['var'] for m in ek.summaries.moments])

plt.figure()
plt.subplot(211)
plt.plot(means, color=colors[0])
plt.fill_between(range(len(data)), y1=means-2*np.sqrt(var), y2=means+2*np.sqrt(var), color=colors[0], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("Ensemble Kalman filter (this does not work, as is expected!)")


# Bootstrap particle filter
fk_model = ssm.Bootstrap(ssm=my_model, data=data)
pf = particles.SMC(fk=fk_model, N=J, collect=[Moments()], resampling='stratified', store_history=True) 
pf.run()
particle_path = np.stack(pf.hist.X)


means_BP = np.stack([m['mean'] for m in pf.summaries.moments])
var_BP = np.stack([m['var'] for m in pf.summaries.moments])

plt.subplot(212)
plt.plot(means_BP, color=colors[0])
plt.fill_between(range(len(data)), y1=means_BP-2*np.sqrt(var_BP), y2=means_BP+2*np.sqrt(var_BP), color=colors[0], alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
plt.title("Bootstrap Particle filter (for comparison)")
plt.tight_layout()