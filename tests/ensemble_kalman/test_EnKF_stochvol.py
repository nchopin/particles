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

#%% ensemble Kalman

fk_EK = enk.EnsembleKalman(my_model, data)
ek = particles.SMC(fk=fk_EK, N=J, collect=[Moments()], store_history=True) 
ek.run()

ek_path = np.stack(ek.hist.X)

colors = ["tab:blue", "tab:green", "tab:orange"]


means = np.stack([m['mean'] for m in ek.summaries.moments])
var = np.stack([m['var'] for m in ek.summaries.moments])

plt.figure()
plt.plot(means, color=colors[m])
plt.fill_between(range(len(data)), y1=means-2*np.sqrt(var), y2=means+2*np.sqrt(var), alpha=0.3)
plt.plot(np.vstack(true_states), "k--")



#%% Bootstrap particle filter
fk_model = ssm.Bootstrap(ssm=my_model, data=data)
pf = particles.SMC(fk=fk_model, N=J, collect=[Moments()], resampling='stratified', store_history=True) 
pf.run()
particle_path = np.stack(pf.hist.X)


means = np.stack([m['mean'] for m in pf.summaries.moments])
var = np.stack([m['var'] for m in pf.summaries.moments])

plt.figure()
plt.plot(means, color=colors[m])
plt.fill_between(range(len(data)), y1=means-2*np.sqrt(var), y2=means+2*np.sqrt(var), alpha=0.3)
plt.plot(np.vstack(true_states), "k--")
