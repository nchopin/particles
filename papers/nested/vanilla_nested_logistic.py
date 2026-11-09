#!/usr/bin/env python

"""
Check the variability of vanilla nested sampling on a logistic regression
example (same example as for NS-SMC, see the other script in this folder).

For the record, if the number of MCMC steps is taken too low (e.g. 10 for
pima), then the estimates have lower empirical variance but high bias.

"""

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from particles import datasets as dts
from particles import distributions as dists
from particles import nested
from particles import smc_samplers as ssps
from particles.utils import multiplexer

datasets = {"pima": dts.Pima, "eeg": dts.Eeg, "sonar": dts.Sonar}
dataset_name = "pima"  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape

# Standard SMC: N is number of particles, K is number of MCMC steps
# Waste-free SMC: M is number of resampled particles, P is length of MCMC
# chains (same notations as in the paper)
# All of the runs are such that N*K or M*P equal N0


# prior & model
scales = 5.0 * np.ones(p)
scales[0] = 20.0  # intercept has a larger scale
prior = dists.StructDist({"beta": dists.MvNormal(scale=scales, cov=np.eye(p))})


class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta["beta"], data[t, :])
        return -np.logaddexp(0.0, -lin)


model = LogisticRegression(data=data, prior=prior)


def worker(N=1_000, nsteps=50):
    alg = nested.Nested_RWmoves(model=model, N=N, nsteps=nsteps, eps=1e-5)
    alg.run()
    return {"est": alg.lZhats[-1], "T": len(alg.lZhats), "cpu": alg.cpu_time}


nruns = 10
nsteps = [10, 20, 30, 40, 50]
results = multiplexer(f=worker, nsteps=nsteps, nruns=nruns, nprocs=0)

true_val = -392.8684  # average over many runs SMC sampler runs
for r in results:
    r["bias"] = np.abs(r["est"] - true_val)

# PLOTS
#######
plt.style.use("ggplot")

plt.figure()
sb.boxplot(y=[r["bias"] for r in results], x=[r["nsteps"] for r in results])
plt.xlabel("nr steps")
plt.ylabel("error log-evidence")
plt.show()
plt.savefig(f"{dataset_name}_bias_vanilla.pdf")
