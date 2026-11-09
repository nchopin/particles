#!/usr/bin/env python

"""
Compare the performance of tempering SMC and nested sampling SMC (introduced by
Salomone et al, 2019) on a simple logistic example (essentially adapted from
the chapter on SMC samplers in the book).

Reference
---------

"""

import numpy as np
import pandas
import seaborn as sb
from matplotlib import pyplot as plt

import particles
from particles import datasets as dts
from particles import distributions as dists
from particles import nested
from particles import smc_samplers as ssps

datasets = {"pima": dts.Pima, "eeg": dts.Eeg, "sonar": dts.Sonar}
dataset_name = "pima"  # choose one of the three
data = datasets[dataset_name]().data
T, p = data.shape

# prior & model
scales = 5.0 * np.ones(p)
scales[0] = 20.0  # intercept has a larger scale
prior = dists.StructDist({"beta": dists.MvNormal(scale=scales, cov=np.eye(p))})


class LogisticRegression(ssps.StaticModel):
    def logpyt(self, theta, t):
        # log-likelihood factor t, for given theta
        lin = np.matmul(theta["beta"], data[t, :])
        return -np.logaddexp(0.0, -lin)


nruns = 100
results = []
model = LogisticRegression(data=data, prior=prior)

lc = 100
N = 1_000
alphas = np.linspace(0.05, 0.95, 19)


def out_func(pf):
    try:
        est = pf.X.shared["log_evid"][-1]
    except (KeyError, IndexError):  # not an NS SMC algorithm
        est = pf.logLt
    return {"nevals": N * ((lc - 1) * pf.t + 1), "est": est}


results = []
algs = ["nested", "tempering"]
for a in alphas:
    fks = {
        "nested": nested.NestedSamplingSMC(model=model, len_chain=lc, ESSrmin=a),
        "tempering": ssps.AdaptiveTempering(model=model, len_chain=lc, ESSrmin=a),
    }
    res = particles.multiSMC(
        fk=fks, N=N, verbose=False, nruns=nruns, out_func=out_func, nprocs=0
    )
    for r in res:
        r["alpha"] = a
    results += res

grand_mean = np.mean([r["est"] for r in results])
for r in results:
    r["mse"] = (r["est"] - grand_mean) ** 2

df = pandas.DataFrame(results)
dfm = df.groupby(["fk", "alpha"]).mean()  # variance as a function of fk and N
dfm = dfm.reset_index()

# plots
#######
savefigs = True  # do you want to save figures as pdfs
plt.style.use("ggplot")
pal = sb.dark_palette("white", n_colors=2)
colors = {"nested": "red", "tempering": "green"}

plt.figure()
for alg in algs:
    dfma = dfm[dfm["fk"] == alg]
    plt.plot(dfma["nevals"], dfma["mse"], color=colors[alg], label=alg)
    plt.xlabel(r"nr evals")
    plt.xscale("log")
    plt.ylabel("MSE")
    plt.yscale("log")
plt.legend()
if savefigs:
    plt.savefig(f"{dataset_name}_nested_vs_tempering_mse_vs_evals.pdf")

plt.figure()
for alg in algs:
    dfma = dfm[dfm["fk"] == alg]
    plt.plot(dfma["alpha"], dfma["mse"], color=colors[alg], label=alg)
    plt.xlim(0.0, 1.0)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("MSE")
    plt.yscale("log")
plt.legend()
if savefigs:
    plt.savefig(f"{dataset_name}_nested_vs_tempering_mse_vs_alpha.pdf")

plt.figure()
for alg in algs:
    dfma = dfm[dfm["fk"] == alg]
    plt.plot(dfma["alpha"], dfma["nevals"] * dfma["mse"], color=colors[alg], label=alg)
    plt.xlim(0.0, 1.0)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("MSE x nr eval")
    plt.yscale("log")
plt.legend()
if savefigs:
    plt.savefig(f"{dataset_name}_nested_vs_tempering_work_normalised.pdf")
