# -*- coding: utf-8 -*-

"""Nested sampling 

    EXPERIMENTAL, NOT TESTED
"""

from __future__ import print_function, division

import copy as cp
import numpy as np
from numpy import random
from scipy import linalg, stats

from particles import utils
from particles.resampling import log_sum_exp_ab as log_sum_exp
from particles import smc_samplers as smc


def unif_minus_one(N, m):
    """ sample uniformly from 0, ..., N-1, minus m """
    return random.randint(m + 1, m + N) % N


def xxT(x):
    v = x[:, np.newaxis]
    return np.dot(v, v.T)


class MeanCovTracker(object):
    """ tracks the mean and cov of a set of points

    Note: points must be given as a (N,d) np.array
    """

    def __init__(self, x):
        self.N = x.shape[0]
        self.sx = x.sum(axis=0)
        self.sxxT = np.dot(x.T, x)
        self.update_mean_cov()

    def update_mean_cov(self):
        self.mean = self.sx / self.N
        self.cov = (self.sxxT / self.N) - xxT(self.mean)
        self.L = linalg.cholesky(self.cov, lower=True)

    def remove_point(self, x):
        self.N -= 1
        self.sx -= x
        self.sxxT -= xxT(x)
        self.update_mean_cov()

    def add_point(self, x):
        self.N += 1
        self.sx += x
        self.sxxT += xxT(x)
        self.update_mean_cov()


class NestedParticles(smc.ThetaParticles):
    containers = ['theta', 'lprior', 'llik']
    shared = []

    def __init__(self, theta=None, lprior=None, llik=None):
        smc.ThetaParticles.__init__(self, theta=theta,
                                    lprior=lprior, llik=llik)


class NestedSampling(object):
    """Base class for nested sampling algorithms
    
    Parameters
    ----------
    * model: a static model
    * N: int
        number of simultaneous points 
    * eps: positive number
        the algorithm stops when relative error is < eps 

    Returns
    -------
    Upon completion (method run), the NestedSampling object has the
    following attributes: 
        * log_weights: list
            log of the weight at each iteration:
            equal to exp(-i/N) - exp(-(i+1)/N) at iteration i
        * points: list
             list of points selected at each iteration
        * lZhats: list
            log of estimate of normalising constant at each iteration i 
            (typically we use the last one as the practical estimate)


    Note: this is the base class; an actual algorithm requires to implement
    the mutate method, which mutates the selected point through MCMC step
    """

    def __init__(self, model=None, N=100, eps=1e-8):
        self.model = model
        self.N = N
        self.eps = eps

    def init_particles(self, th):
        return NestedParticles(theta=th,
                               lprior=self.model.prior.logpdf(th),
                               llik=self.model.loglik(th))

    def setup(self):
        th = self.model.prior.rvs(size=self.N)
        self.x = self.init_particles(th)

    def mutate(self, n, m):
        """ n: index of deleted point
            m: index of starting point for MCMC
        """
        raise NotImplementedError

    def step(self):
        n_lowest = np.argmin(self.x.llik)
        self.points.append(cp.deepcopy(self.x[n_lowest]))
        # a copy is needed here (x[n] is a view)
        n_start = unif_minus_one(self.N, n_lowest)
        self.mutate(n_lowest, n_start)

    def stopping_time(self):
        return np.abs(self.lZhats[-1] - self.lZhats[-2]) < self.eps 

    @utils.timer
    def run(self):
        self.setup()
        self.points = []
        self.log_weights = [np.log(1. - np.exp(-1. / self.N))]
        self.step()
        self.lZhats = [self.log_weights[0] + self.points[0].llik]
        while True:
            self.step()
            b = self.log_weights[-1] + self.points[-1].llik
            self.lZhats.append(log_sum_exp(self.lZhats[-1], b))
            if self.stopping_time():
                break
            next_lw = self.log_weights[-1] - 1. / self.N
            self.log_weights.append(next_lw)
            if len(self.log_weights) % self.N == 0:
                print('iteration %i: log(Z_hat) = %f' % (len(self.log_weights),
                                                         self.lZhats[-1]))


class Nested_RWmoves(NestedSampling):
    """Nested sampling with (adaptive) random walk Metropolis moves"""
    def __init__(self, nsteps=1, scale=None, **kwargs):
        NestedSampling.__init__(self, **kwargs)

    def setup(self):
        NestedSampling.setup(self)
        self.tracker = MeanCovTracker(self.x.arr)
        if self.scale is None:  # We know dim only after x is set
            self.scale = 2.38 / np.sqrt(self.x.dim)
        self.xp = NestedParticles(theta=np.empty(1, dtype=self.x.theta.dtype),
                                  llik=np.zeros(1), lprior=np.zeros(1))
        self.nacc = 0

    def update_xp_fields(self):
        self.xp.lprior = self.model.prior.logpdf(self.xp.theta)
        self.xp.llik = self.model.loglik(self.xp.theta)

    def mutate(self, n, m):
        self.tracker.remove_point(self.x.arr[n])
        lmin = self.x.llik[n]
        self.x.arr[n] = self.x.arr[m]
        for _ in range(self.nsteps):
            z = self.scale * np.dot(self.tracker.L,
                                    stats.norm.rvs(size=self.x.dim))
            self.xp.arr[0] = self.x.arr[n] + z
            self.update_xp_fields()
            if self.xp.llik[0] > lmin:
                if np.log(random.rand()) < self.xp.lprior - self.x.lprior[n]:
                    self.x.copyto_at(n, self.xp, 0)
                    self.nacc += 1
        self.tracker.add_point(self.x.arr[n])
