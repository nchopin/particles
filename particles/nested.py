# -*- coding: utf-8 -*-

"""Nested sampling (vanilla and SMC).

.. warning:: This module is less tested than the rest of the package.
Moreover, this documentation does not explain precisely how nested sampling
works (and this topic is not covered in our book). Thus, refer to e.g. the
original papers of Skilling or Chopin and Robert (2010, Biometrika). For nested
sampling SMC, see the paper of Salomone et al (2018).

Vanilla nested sampling
=======================

This module contains classes that implement nested sampling:

    * `NestedSampling`: base class;
    * `Nested_RWmoves` : nested sampling algorithm based on random walk 
      Metropolis steps. 

To use the latter, you need to define first a static model, in the same way as
in the `smc_samplers` module. For instance::

    import particles
    from particles import smc_samplers as ssp
    from particles import distributions as dists

    class ToyModel(ssp.smc_samplers):
        def logpyt(self, theta, t):  # log-likelihood of data-point at time t
            return stats.norm.logpdf(self.data[t], loc=theta)

    my_prior = dists.Normal()  # theta ~ N(0, 1)
    y = random.normal(size=1)  # y | theta ~ N(theta, 1)
    toy_model = ToyModel(data=y, prior=my_prior)

Then, the algorithm may be set and run as follows::

    from particles import nested

    algo = nested.Nested_RWmoves(model=toy_model, N=1000, nsteps=3, eps=1e-8)
    algo.run()
    print('estimate of log-evidence: %f' % algo.lZhats[-1])

This will run a nested sampling algorithm which propagates N=1000 particles;
each time a point is deleted, it is replaced by another point obtained as
follows: another point is selected at random, and then moved through ``nsteps``
steps of a Gaussian random walk Metropolis kernel (which leaves invariant the
prior constrained to the current likelihood contour). To make these steps
reasonably efficient, the covariance matrix of the random walk proposal is
dynamically adapted to the current sample of points. The algorithm is stopped
when the different between the two most recent estimates of the log-evidence 
is below ``eps``. 

To implement your own algorithm, you must sub-class `NestedSampling` like
this::

    from particles import nested

    class MyNestedSampler(nested.NestedSampling):
        def mutate(self, n, m):
            # implement a MCMC step that replace point X[n] with the point
            # obtained by starting at X[m] and doing a certain number of steps 
            return value 

Nested sampling Sequential Monte Carlo
======================================

Salomone et al (2018) proposed a SMC sampler inspired by nested sampling. The
target distribution at time t is the prior constrained to the likelihood being
larger than constant l_t. These constants may be chosen adaptively: in this
implementation, the next l_t is set to the ``ESSrmin`` upper-quantile of the
likelihood of the current points (where ``ESSrmin`` is specified by the user).
In other words, l_t is chosen so that the ESS equals this value. 
(``ESSrmin`` corresponds to 1 - rho in Salomone et al's notations.)

This module implements this SMC sampler as `NestedSamplingSMC`, a sub-class of
`smc_samplers.FKSMCsampler`, which may be used the same way as other SMC
samplers defined in module `smc_samplers`::

    fk = nested.NestedSamplingSMC(model=toy_model, wastefree=True, ESSrmin=0.3)
    alg = particles.SMC(fk=fk, N=1_000)
    alg.run()

Upon completion, the dictionary `alg.X.shared` will contain the successive
estimates of the log-evidence (log of marginal likelihood, in practice the
final one is the one you want to use), and the successive values of l_t.

Note that a waste-free version of NS-SMC is run by default, but the original
paper of Salomone et al (which predates NS-SMC) only considers a standard
version. (For more details on waste-free SMC vs standard SMC, see module
`smc_samplers` and the corresponding jupyter notebook.)

Reference
---------
Salomone, South L., Drovandi C.  and Kroese D. (2018). Unbiased and Consistent 
Nested Sampling via Sequential Monte Carlo, arxiv 1805.03924.

"""

from __future__ import print_function, division

import numpy as np
from numpy import random
from scipy import linalg, stats, special

from particles import resampling as rs
from particles import smc_samplers as ssps
from particles import utils


def unif_minus_one(N, m):
    """Sample uniformly from 0, ..., N-1, minus m.
    """
    return random.randint(m + 1, m + N) % N


def xxT(x):
    v = x[:, np.newaxis]
    return np.dot(v, v.T)


class MeanCovTracker(object):
    """Tracks mean and cov of a set of points. 

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



class NestedSampling(object):
    """Base class for nested sampling algorithms.

    Parameters
    ----------
    * model: SMCsamplers.StaticModel object 
        a static model
    * N: int
        number of simultaneous points
    * eps: positive number
        the algorithm stops when relative error is smaller than eps 

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


    .. note:: this is the base class; an actual algorithm requires to implement
    the mutate method, which mutates the selected point through MCMC steps.
    """

    def __init__(self, model=None, N=100, eps=1e-8):
        self.model = model
        self.N = N
        self.eps = eps

    def setup(self):
        th = self.model.prior.rvs(size=self.N)
        lp = self.model.prior.logpdf(th)
        ll = self.model.loglik(th)
        self.x = ssps.ThetaParticles(theta=th, lprior=lp, llik=ll)

    def mutate(self, n, m):
        """ n: index of deleted point
            m: index of starting point for MCMC
        """
        raise NotImplementedError

    def step(self):
        n_lowest = int(np.argmin(self.x.llik))
        self.points.append(self.x[n_lowest])
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
        self.lZhats = [self.log_weights[0] + self.points[0]["llik"]] # TODO
        while True:
            self.step()
            b = self.log_weights[-1] + self.points[-1]["llik"]
            self.lZhats.append(rs.log_sum_exp_ab(self.lZhats[-1], b))
            if self.stopping_time():
                break
            next_lw = self.log_weights[-1] - 1. / self.N
            self.log_weights.append(next_lw)
            if len(self.log_weights) % self.N == 0:
                print('iteration %i: log(Z_hat) = %f' % (len(self.log_weights),
                                                         self.lZhats[-1]))


class Nested_RWmoves(NestedSampling):
    """Nested sampling with (adaptive) random walk Metropolis moves.
    """
    def __init__(self, model=None, N=100, eps=1e-8, nsteps=1, scale=None):
        super().__init__(model=model, N=N, eps=eps)
        self.nsteps = nsteps
        self.scale = scale

    def setup(self):
        super().setup()
        arr = ssps.view_2d_array(self.x.theta)
        N, d = arr.shape
        self.tracker = MeanCovTracker(arr)
        if self.scale is None:  # We know dim only after x is set
            self.scale = 2.38 / np.sqrt(d)
        self.xp = ssps.ThetaParticles(theta=np.empty(1, dtype=self.x.theta.dtype),
                                      llik=np.zeros(1), lprior=np.zeros(1))
        self.nacc = 0

    def update_xp_fields(self):
        self.xp.lprior = self.model.prior.logpdf(self.xp.theta)
        self.xp.llik = self.model.loglik(self.xp.theta)

    def mutate(self, n, m):
        arr = ssps.view_2d_array(self.x.theta)
        N, d = arr.shape
        self.tracker.remove_point(arr[n])
        lmin = self.x.llik[n]
        arr[n] = arr[m]
        xarr = ssps.view_2d_array(self.xp.theta)
        for _ in range(self.nsteps):
            z = self.scale * np.dot(self.tracker.L,
                                    stats.norm.rvs(size=d))
            xarr[0] = arr[n] + z
            self.update_xp_fields()
            if self.xp.llik[0] > lmin:
                if np.log(random.rand()) < self.xp.lprior - self.x.lprior[n]:
                    self.x.copyto_at(n, self.xp, 0)
                    self.nacc += 1
        self.tracker.add_point(arr[n])

#############################
## NS-SMC

class NestedSamplingSMC(ssps.FKSMCsampler):
    """Feynman-Kac class for the nested sampling SMC algorithm.

    Based on Salomone et al. (2018). Target a time t is prior constrained to
    likelihood being above constant lt. 

    Parameters
    ----------
    ESSrmin: float
        next lt is chosen so that probability that L(x) > lt is ESSrmin.
    eps: float
         algorithm stops when delta between the two latest log-evidence is
         below eps.

    See base class for other parameters.

    The successive estimates of the log-evidence is stored in the list
    `self.X.shared['log_evid']`. 

    Reference
    ---------
    Salomone, South L., Drovandi C.  and Kroese D. (2018). Unbiased and Consistent Nested 
    Sampling via Sequential Monte Carlo, arxiv 1805.03924.
    """
    def __init__(self, model=None, wastefree=True, len_chain=10, move=None,
                 ESSrmin=0.1, eps=0.01):
        super().__init__(model=model, wastefree=wastefree,
                         len_chain=len_chain, move=move)
        self.ESSrmin = ESSrmin
        self.eps = eps

    def time_to_resample(self, smc):
        self.move.calibrate(smc.W, smc.X)
        return True  # We *always* resample

    def done(self, smc):
        try:
            lt = smc.X.shared['lts'][-1]
        except: # attribute does not exist yet, or list is empty
            lt = 0.
        return lt == np.inf

    def summary_format(self, smc):
        msg = super().summary_format(smc)
        return '%s, loglik=%f' % (msg, smc.X.shared['lts'][-1])

    def logG(self, t, xp, x):
        curr_evid = x.shared['log_evid'][-1]
        lt = np.percentile(x.llik, 100. * (1. - self.ESSrmin))
        # estimate for non-terminal iteration
        lZt = (t * np.log(self.ESSrmin) - np.log(x.N) 
              + special.logsumexp(x.llik[x.llik <= lt]))
        new_evid = rs.log_sum_exp_ab(curr_evid, lZt)
        # estimate at final time, taking lt=infinity
        lZt_final = (t * np.log(self.ESSrmin) - np.log(x.N) 
               + special.logsumexp(x.llik))
        new_evid_final = rs.log_sum_exp_ab(curr_evid, lZt_final)
        if np.abs(new_evid - new_evid_final) < self.eps: # stopping criterion
            lt = np.inf
            lw = np.zeros_like(x.llik)
            new_evid = new_evid_final
        else:
            lw = np.where(x.llik > lt, 0., -np.inf)
        x.shared['lts'].append(lt)
        x.shared['log_evid'].append(new_evid)
        return lw

    def current_target(self, lt):
        def func(x):
            x.lprior = self.model.prior.logpdf(x.theta)
            x.llik = self.model.loglik(x.theta)
            if lt == -np.inf:
                x.lpost = x.lprior.copy()
            else:
                x.lpost = np.where(x.llik >= lt, x.lprior, -np.inf)
                #TODO better name for target density
        return func

    def _M0(self, N):
        x0 = ssps.ThetaParticles(theta=self.model.prior.rvs(size=N))
        x0.shared['lts'] = [-np.inf]
        x0.shared['log_evid'] = [-np.inf]
        self.current_target(-np.inf)(x0)
        return x0

    def M(self, t, xp):
        return self.move(xp, self.current_target(xp.shared['lts'][-1]))
