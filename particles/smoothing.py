# -*- coding: utf-8 -*-

"""
This module implements:

1. particle history classes,  which store the full or partial history
   of a SMC algorithm.

2. off-line smoothing algorithms as methods of these classes.

For on-line smoothing, see instead the `collectors` module.

History classes
===============

A `SMC` object has a ``hist`` attribute, which is used to record at *certain*
times t:

* the N current particles :math:`X_t^n`;
* their weights;
* (optionally, see below), the ancestor variables :math:`A_t^n`.

The frequency at which history is recorded depends on option ``store_history``
of class `SMC`. Possible options are:

* ``True``: records full history (at every time t);
* ``False``: no history (attribute `hist` set to ``None``);
* callable ``f``: history is recorded at time t if ``f(t)`` returns True
* int k: records a rolling window history of length k (may be used
  to perform fixed-lag smoothing)

This module implements different classes that correspond to the different cases:

* `ParticleHistory`: full history (based on lists)
* `PartialParticleHistory`: partial history (based on dictionaries)
* `RollingParticleHistory`: rolling window history (based on `deques`_)

.. _deques: https://docs.python.org/3/library/collections.html#collections.deque

All these classes provide a similar interface. If ``smc`` is a `SMC` object,
then:

* ``smc.hist.X[t]`` returns the N particles at time t
* ``smc.hist.wgts[t]`` returns the N weights at time t (see `resampling.weights`)
* ``smc.hist.A[t]`` returns the N ancestor variables at time t

Partial History
===============

Here are some examples on one may record history only at certain times::

    # store every other 10 iterations
    smc = SMC(fk=fk, N=100, store_history=lambda t: (t % 10) == 0)

    # store at certain times given by a list
    times = [10, 30, 84]
    smc = SMC(fk=fk, N=100, store_history=lambda t: t in times)

Once the algorithm is run, ``smc.hist.X`` and ``smc.hist.wgts`` are
dictionaries, the keys of which are the times where history was recorded. The
ancestor variables are not recorded in that case::

    smc.run()
    smc.hist.X[10]  # the N particles at time 10
    smc.hist.A[10]  # raises an error


Full history, off-line smoothing algorithms
===========================================

For a given state-space model, off-line smoothing amounts to approximate the
distribution of the complete trajectory :math:`X_{0:T}`, given data
:math:`y_{0:T}`, at some fixed time horizon T. The corresponding algorithms
take as an input the complete history of a particle filter, run until time T
(forward pass). Say::

    # forward pass
    fk = ssm.Bootstrap(ssm=my_ssm, data=y)
    pf = particles.SMC(fk=fk, N=100, store_history=True)
    pf.run()

Then, ``pf.hist`` is an instance of class `ParticleHistory`. It implements two
types of approaches:

    * two-filter smoothing: uses two particle filters (one forward, one
      backward) to estimate marginal expectations; see `two_filter_smoothing`.

    * FFBS (forward filtering backward sampling): uses one particle filter,
      then generates trajectories from its history, using different methods
      (exact, rejection, MCMC, QMC). See `backward_sampling_mcmc`,
      `backward_sampling_ON2`, `backward_sampling_reject`, and
      `backward_sampling_qmc`. Recommended method is `backward_sampling_mcmc`,
      see discussion in Dang & Chopin (2022).

For more details, see the documentation of `ParticleHistory`, the ipython
notebook on smoothing, Chapter 12 of the book, and Dang & Chopin (2022).

.. warning:: the complete history of a particle filter may take a lot of
  memory.

Rolling history, Fixed-lag smoothing
====================================

To obtain a rolling window (fixed-length) history::

    smc = SMC(fk=fk, N=100, store_history=10)
    smc.run()

In that case, fields ``smc.hist.X``, ``smc.hist.wgts`` and ``smc.hist.A`` are
`deques`_  of max length 10.  Using negative indices::

    smc.hist.X[-1]  # the particles at final time T
    smc.hist.X[-2]  # the particles at time T - 1
    # ...
    smc.hist.X[-10] # the N particles at time T - 9
    smc.hist.X[-11] # raises an error

Note that this type of history makes it possible to perform fixed-lag smoothing
as follows::

    B = smc.hist.compute_trajectories()
    # B[t, n] is index of ancestor of X_T^n at time t
    phi = lambda x: x  # any test function
    est = np.average(phi(smc.hist.X[-10][B[-10, :]]), weights=smc.W)
    # est is an estimate of E[ phi(X_{T-9}) | Y_{0:T}]

.. note:: recall that it is possible to run `SMC` algorithms step by step,
   since they are iterators. Hence it is possible to do fixed-lag smoothing
   step-by-step as well.


"""

from __future__ import absolute_import, division, print_function

from collections import deque
from itertools import islice
import numpy as np
from numpy import random
from scipy import stats  # worker
import time

import particles # worker
from particles import hilbert
from particles import resampling as rs
from particles import rqmc

def generate_hist_obj(option, smc):
    if option is True:
        return ParticleHistory(smc.fk, smc.qmc)
    elif option is False:
        return None
    elif callable(option):
        return PartialParticleHistory(option)
    elif isinstance(option, int) and option >= 0:
        return RollingParticleHistory(option)
    else:
        raise ValueError('store_history: invalid option')

class PartialParticleHistory(object):
    """Partial history.

    History that records the particle system only at certain times.
    See `smoothing` module doc for more details.
    """
    def __init__(self, func):
        self.is_save_time = func
        self.X, self.wgts = {}, {}

    def save(self, smc):
        t = smc.t
        if self.is_save_time(t):
            self.X[t] = smc.X
            self.wgts[t] = smc.wgts

class RollingParticleHistory(object):
    """Rolling window history.

    History that keeps only the k most recent particle systems. Based on
    deques. See `smoothing` module doc for more details.

    """
    def __init__(self, length):
        self.X = deque([], length)
        self.A = deque([], length)
        self.wgts = deque([], length)

    @property
    def N(self):
        """Number of particles at each time step.
        """
        return self.X[0].shape[0]

    @property
    def T(self):
        """Current length of history.
        """
        return len(self.X)

    def save(self, smc):
        self.X.append(smc.X)
        self.A.append(smc.A)
        self.wgts.append(smc.wgts)

    def compute_trajectories(self):
        """Compute the N trajectories that constitute the current genealogy.

        Returns a (T, N) int array, such that B[t, n] is the index of ancestor
        at time t of particle X_T^n, where T is the current length of history.
        """
        Bs = [np.arange(self.N)]
        for A in list(self.A)[-1:0:-1]:  # list in case self.A is a deque
            Bs.append(A[Bs[-1]])
        Bs.reverse()
        return np.array(Bs)

class ParticleHistory(RollingParticleHistory):
    """Particle history.

    A class to store the full history of a particle algorithm, i.e.
    at each time t=0,...T, the N particles, their weights, and their ancestors.
    Off-line smoothing algorithms are methods of this class.

    `SMC` creates an object of this class when invoked with
    ``store_history=True``, and then save at every time t the set of particles,
    their weights (and their logarithm), and the ancestor variables.

    Attributes
    ----------
    X: list
        X[t] is the object that represents the N particles at iteration t
    wgts: list
        wgts[t] is a `Weights` object (see module `resampling`) that represents
        the N weights at time t
    A: list
        A[t] is the vector of ancestor indices at time t

    """

    def __init__(self, fk, qmc):
        self.X, self.A, self.wgts = [], [], []
        if qmc:
            self.h_orders = []
        self.fk = fk

    def save(self, smc):
        RollingParticleHistory.save(self, smc)
        if hasattr(smc, 'h_order'):
            self.h_orders.append(smc.h_order)

    def extract_one_trajectory(self):
        """Extract a single trajectory from the particle history.

        The final state is chosen randomly, then the corresponding trajectory
        is constructed backwards, until time t=0.
        """
        traj = []
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                n = rs.multinomial_once(self.wgts[-1].W)
            else:
                n = self.A[t + 1][n]
            traj.append(self.X[t][n])
        return traj[::-1]

    def _check_h_orders(self):
        if not hasattr(self, 'h_orders'):
            raise ValueError('QMC FFBS requires particles have been Hilbert\
                             ordered during the forward pass')

    def _init_backward_sampling(self, M):
        idx = np.empty((self.T, M), dtype=int)
        idx[-1, :] = rs.multinomial(self.wgts[-1].W, M=M)
        return idx

    def _output_backward_sampling(self, idx):
        # When M=1, we want a list of states, not a list of arrays containing
        # one state
        if idx.shape[1] == 1:
            idx = idx.squeeze(axis=1)
        paths = [self.X[t][idx[t]] for t in range(self.T)]
        return paths

    def backward_sampling_ON2(self, M):
        """Default O(N^2) version of FFBS.

        Parameter
        ---------
        M: int
            number of trajectories to generate

        Returns
        -------
        paths: a list of ndarrays
            paths[t][n] is component t of trajectory m.
        """
        idx = self._init_backward_sampling(M)
        for m in range(M):
            for t in reversed(range(self.T - 1)):
                lwm = (self.wgts[t].lw + self.fk.logpt(t + 1, self.X[t],
                                                self.X[t + 1][idx[t + 1, m]]))
                idx[t, m] = rs.multinomial_once(rs.exp_and_normalise(lwm))
        return self._output_backward_sampling(idx)

    def backward_sampling_mcmc(self, M, nsteps=1):
        """MCMC-based backward sampling.

        Uses one step of an independent Metropolis kernel, where the proposal
        is the multinomial distribution based on the weights. This is now the
        method recommended by default, since it has O(N) (deterministic)
        complexity and it seems to work well, as explained in Dau & Chopin
        (2022).

        Parameters
        ----------
        M: int
            number of trajectories to generate
        nsteps: int (default: 1)
            number of independent Metropolis steps

        Returns
        -------
        paths: a list of ndarrays
            paths[t][n] is component t of trajectory m.

        References
        ----------
        Dau, H.D. and Chopin, N. (2022).On the complexity of backward smoothing
        algorithms, arXiv:2207.00976
        """
        idx = self._init_backward_sampling(M)
        for t in reversed(range(self.T - 1)):
            idx[t, :] = self.A[t + 1][idx[t + 1, :]]
            prop = rs.multinomial(self.wgts[t].W, M=M)
            xn = self.X[t + 1][idx[t + 1, :]]
            lpr_acc = (self.fk.logpt(t + 1, self.X[t][prop], xn)
                       - self.fk.logpt(t + 1, self.X[t][idx[t, :]], xn))
            lu = np.log(np.random.rand(M))
            idx[t, :] = np.where(lu < lpr_acc, prop, idx[t, :])
        return self._output_backward_sampling(idx)

    def backward_sampling_reject(self, M, max_trials=None):
        """Rejection-based backward sampling.

        Because of the issues with the pure rejection method discussed in Dau
        and Chopin (2022), i.e. execution time is random and may have infinite
        expectation, we implement the hybrid version introduced in Dau & Chopin
        (2022), where, for a given particle, we make at most `max_trials`
        attempts, before switching back to the exact (expensive) method. To
        recover the "pure rejection" scheme, simply set this parameter to
        infinity (or a very large value).

        Parameters
        ----------
        M: int
            number of trajectories to generate
        max_trials: int (default: M)
            max number of rejection steps before we switch to the expensive
            method.

        Returns
        -------
        paths: a list of ndarrays
            paths[t][n] is component t of trajectory m.

        Note
        ----
        1. To use rejection, you need to define (in the model) a method
           `upper_bound_trans`, which provides the log of a constant C_t such
           that :math:`p_t(x_t|x_{t-1}) \leq C_t`.

        2.  The average acceptance rate is saved in `self.acc_rate`.

        3. Dau & Chopin (2022) recommend to set max_trials to M, or a multiple
           of M.

        References
        ----------
        Dau, H.D. and Chopin, N. (2022).On the complexity of backward smoothing
        algorithms, arXiv:2207.00976
        """
        idx = self._init_backward_sampling(M)
        if max_trials is None:
            max_trials = M
        tot_nattempts = 0
        for t in reversed(range(self.T - 1)):
            where_rejected = np.arange(M)
            who_rejected = self.X[t + 1][idx[t + 1, :]]
            nrejected = M
            nattempts = 0
            gen = rs.MultinomialQueue(self.wgts[t].W, M=M)
            while nrejected > 0 and nattempts < max_trials:
                nattempts += nrejected
                nprop = gen.dequeue(nrejected)
                lpr_acc = (self.fk.logpt(t + 1, self.X[t][nprop],
                                            who_rejected)
                           - self.fk.upper_bound_trans(t + 1))
                newly_accepted = np.log(random.rand(nrejected)) < lpr_acc
                still_rejected = np.logical_not(newly_accepted)
                idx[t, where_rejected[newly_accepted]] = nprop[newly_accepted]
                where_rejected = where_rejected[still_rejected]
                who_rejected = who_rejected[still_rejected]
                nrejected -= sum(newly_accepted)
            if nrejected > 0:
                for m in where_rejected:
                    lwm = (self.wgts[t].lw + self.fk.logpt(t + 1, self.X[t],
                                                         self.X[t + 1][idx[t + 1, m]]))
                    idx[t, m] = rs.multinomial_once(rs.exp_and_normalise(lwm))
            tot_nattempts += nattempts
        self.acc_rate = (M * (self.T - 1)) / tot_nattempts
        return self._output_backward_sampling(idx)

    def backward_sampling_qmc(self, M):
        """QMC version of backward sampling.

        Parameters
        ----------
        M : int
            number of trajectories to generate

        Note
        ----
        This is the version to use if your particle filter relies on QMC.
        Otherwise use one of the other methods.
        """
        self._check_h_orders()
        u = rqmc.sobol(M, self.T)
        # the final particles have not been sorted
        hT = hilbert.hilbert_sort(self.X[-1])
        # searchsorted to avoid having to sort in place u according to u[:,T-1]
        idx = np.searchsorted(np.cumsum(self.wgts[-1].W[hT]), u[:, -1])
        paths = [self.X[-1][hT][idx], ]
        for t in reversed(range(self.T - 1)):
            idx = np.empty(M, dtype=np.int64)
            for m, xn in enumerate(paths[-1]):
                lwm = self.wgts[t].lw + self.fk.logpt(t + 1, self.X[t], xn)
                # use ordered version here
                cw = np.cumsum(rs.exp_and_normalise(lwm[self.h_orders[t]]))
                idx[m] = np.searchsorted(cw, u[m, t])
            paths.append(self.X[t][self.h_orders[t]][idx])
        paths.reverse()
        return paths


#     def backward_sampling_lincost_pedagogical(self, M):
#         """ O(N) FFBS
#
#             Don't use this! This is the *pedagogical* version of O(N) FFBS:
#             code is simpler to understand, but quite slow, because it has loops
#         """
#         if not hasattr(self.fk, 'upper_bound_trans'):
#             raise ValueError('O(N) version of backward smoothing'
#                              +'requires to specify constant upper_bound_trans(t)'
#                              +' s.t. log p_t(x_t|x_{t-1})<upper_bound_trans(t)')
#         idx = np.empty((self.T, M), dtype=int)
#         idx[-1, :] = rs.multinomial(M, self.wgts[-1].W)
#         nattempts = 0
#         for t in xrange(self.T - 2, -1, -1):
#             gen = rs.MulinomialQueue(M, self.wgts[t].W)
#             for m in xrange(M):
#                 while True:
#                     nattempts += 1
#                     nprop = gen.dequeue(1)
#                     lpr_acc = (self.fk.logpt(t+1, self.X[t][nprop],
#                                                 self.X[t+1][idx[t+1, m]])
#                                -self.fk.upper_bound_trans(t+1))
#                     if np.log(random.rand()) < lpr_acc:
#                         break
#                 idx[t, m] = nprop
#         print('O(N) FFBS: acceptance rate is %1.2f' %
#               (M * (self.T - 1) / nattempts))
#         return [self.X[t][idx[t, :]] for t in range(self.T)]

    def two_filter_smoothing(self, t, info, phi, loggamma, linear_cost=False,
                             return_ess=False, modif_forward=None,
                             modif_info=None):
        """Two-filter smoothing.

        Parameters
        ----------
        t: time, in range 0 <= t < T-1
        info: SMC object
            the information filter
        phi: function
            test function, a function of (X_t,X_{t+1})
        loggamma: function
            a function of (X_{t+1})
        linear_cost: bool
            if True, use the O(N) variant (basic version is O(N^2))

        Returns
        -------
        Two-filter estimate of the smoothing expectation of phi(X_t,x_{t+1})
        """
        ti = self.T - 2 - t  # t+1 in reverse
        if t < 0 or t >= self.T - 1:
            raise ValueError(
                'two-filter smoothing: t must be in range 0,...,T-2')
        lwinfo = info.hist.wgts[ti].lw - loggamma(info.hist.X[ti])
        if linear_cost:
            return self._two_filter_smoothing_ON(t, ti, info, phi, lwinfo,
                                               return_ess,
                                               modif_forward, modif_info)
        else:
            return self._two_filter_smoothing_ON2(t, ti, info, phi, lwinfo)

    def _two_filter_smoothing_ON2(self, t, ti, info, phi, lwinfo):
        """O(N^2) version of two-filter smoothing.

        This method should not be called directly, see two_filter_smoothing.
        """
        sp, sw = 0., 0.
        upb = lwinfo.max() + self.wgts[t].lw.max()
        if hasattr(self.fk, 'upper_bound_trans'):
            upb += self.fk.upper_bound_trans(t + 1)
        # Loop over n, to avoid having in memory a NxN matrix
        for n in range(self.N):
            omegan = np.exp(lwinfo + self.wgts[t].lw[n] - upb
                            + self.fk.logpt(t + 1, self.X[t][n],
                                               info.hist.X[ti]))
            sp += np.sum(omegan * phi(self.X[t][n], info.hist.X[ti]))
            sw += np.sum(omegan)
        return sp / sw

    def _two_filter_smoothing_ON(self, t, ti, info, phi, lwinfo, return_ess,
                               modif_forward, modif_info):
        """O(N) version of two-filter smoothing.

        This method should not be called directly, see two_filter_smoothing.
        """
        if modif_info is not None:
            lwinfo += modif_info
        Winfo = rs.exp_and_normalise(lwinfo)
        I = rs.multinomial(Winfo)
        if modif_forward is not None:
            lw = self.wgts[t].lw + modif_forward
            W = rs.exp_and_normalise(lw)
        else:
            W = self.wgts[t].W
        J = rs.multinomial(W)
        log_omega = self.fk.logpt(t + 1, self.X[t][J], info.hist.X[ti][I])
        if modif_forward is not None:
            log_omega -= modif_forward[J]
        if modif_info is not None:
            log_omega -= modif_info[I]
        Om = rs.exp_and_normalise(log_omega)
        est = np.average(phi(self.X[t][J], info.hist.X[ti][I]), axis=0,
                         weights=Om)
        if return_ess:
            return (est, 1. / np.sum(Om**2))
        else:
            return est


def smoothing_worker(method=None, N=100, fk=None, fk_info=None,
                     add_func=None, log_gamma=None):
    """Generic worker for off-line smoothing algorithms.

    This worker may be used in conjunction with utils.multiplexer in order to
    run in parallel off-line smoothing algorithms.

    Parameters
    ----------
    method: string
         ['FFBS_purereject', 'FFBS_hybrid', FFBS_MCMC', 'FFBS_ON2', 'FFBS_QMC',
           'two-filter_ON', 'two-filter_ON_prop', 'two-filter_ON2']
    N: int
        number of particles
    fk: Feynman-Kac object
        The Feynman-Kac model for the forward filter
    fk_info: Feynman-Kac object (default=None)
        the Feynman-Kac model for the information filter; if None,
        set to the same Feynman-Kac model as fk, with data in reverse
    add_func: function, with signature (t, x, xf)
        additive function, at time t, for particles x=x_t and xf=x_{t+1}
    log_gamma: function
        log of function gamma (see book)

    Returns
    -------
    a dict with fields:
        est: a ndarray of length T
        cpu_time

    Notes
    -----
    'FFBS_hybrid' is the hybrid method that makes at most N attempts to
    generate an ancestor using rejection, and then switches back to the
    standard (expensive method). On the other hand, 'FFBS_purereject' is the
    original rejection-based FFBS method, where only rejection is used. See Dau
    & Chopin (2022) for a discussion.
    """
    T = fk.T
    if fk_info is None:
        fk_info = fk.__class__(ssm=fk.ssm, data=fk.data[::-1])
    est = np.zeros(T - 1)
    if method=='FFBS_QMC':
        pf = particles.SQMC(fk=fk, N=N, store_history=True)
    else:
        pf = particles.SMC(fk=fk, N=N, store_history=True)
    tic = time.perf_counter()
    pf.run()
    if method.startswith('FFBS'):
        submethod = method.split('_')[-1]
        if submethod == 'QMC':
            z = pf.hist.backward_sampling_qmc(N)
        elif submethod == 'ON2':
            z = pf.hist.backward_sampling_ON2(N)
        elif submethod == 'MCMC':
            z = pf.hist.backward_sampling_mcmc(N)
        elif submethod == 'hybrid':
            z = pf.hist.backward_sampling_reject(N)
        elif submethod == 'purereject':
            z = pf.hist.backward_sampling_reject(N, max_trials=N * 10**9)
        for t in range(T - 1):
            est[t] = np.mean(add_func(t, z[t], z[t + 1]))
    elif method in ['two-filter_ON2', 'two-filter_ON', 'two-filter_ON_prop']:
        infopf = particles.SMC(fk=fk_info, N=N, store_history=True)
        infopf.run()
        for t in range(T - 1):
            psi = lambda x, xf: add_func(t, x, xf)
            if method == 'two-filter_ON2':
                est[t] = pf.hist.two_filter_smoothing(t, infopf, psi, log_gamma)
            else:
                ti = T - 2 - t  # t+1 for info filter
                if method == 'two-filter_ON_prop':
                    modif_fwd = stats.norm.logpdf(pf.hist.X[t],
                                          loc=np.mean(infopf.hist.X[ti + 1]),
                                          scale=np.std(infopf.hist.X[ti + 1]))
                    modif_info = stats.norm.logpdf(infopf.hist.X[ti],
                                           loc=np.mean(pf.hist.X[t + 1]),
                                           scale=np.std(pf.hist.X[t + 1]))
                else:
                    modif_fwd, modif_info = None, None
                est[t] = pf.hist.two_filter_smoothing(t, infopf, psi, log_gamma,
                                                     linear_cost=True,
                                                     modif_forward=modif_fwd,
                                                     modif_info=modif_info)
    else:
        print('smoothing_worker: no such method?')
    cpu_time = time.perf_counter() - tic
    print(method + ' took %.2f s for N=%i' % (cpu_time, N))
    return {'est': est, 'cpu': cpu_time}
