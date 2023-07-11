#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baum-Welch filter/smoother for hidden Markov models.

Overview
========

A hidden Markov model (HMM) is a state-space model with a finite state-space,
{1, ..., K}. The Baum-Welch algorithm allows to compute (exactly) the filtering and
smoothing distributions of such a model; i.e. the probabilities that X_t=k
given data Y_{0:t} (filtering) or the complete data Y_{0:T} (smoothing).
In addition, one may sample trajectories from the complete smoothing
distribution (the distribution of all the states, X_{0:T}, given all the data).

Hidden Markov models and the Baum-Welch algorithm are covered in Chapter 6 of
the book.

Defining a hidden Markov model
==============================

Hidden Markov models are represented as `HMM` objects; `HMM` is a subclass
of `StateSpaceModel` (from module `state_space_models`), which assigns:

    * A categorical distribution to X_0 (parameter `init_dist`)
    * A categorical distribution to X_t given X_{t-1} (parameter `trans_mat`)

The distributions of Y_t|X_t must be defined by sub-classing `HMM`. For instance,
this module defines a `GaussianHMM` class as follows::

    class GaussianHMM(HMM):
        default_params = {'mus': None, 'sigmas': None}
        default_params.update(HMM.default_params)

        def PY(self, t, xp, x):
            return dists.Normal(loc=self.mus[x], scale=self.sigmas[x])

One may now define a particular model in the usual way::

    tm = np.array([[0.9 0.1], [0.2, 0.8]])
    my_hmm = hmm.GaussianHMM(mus=np.array([0., 1.], sigmas=np.ones(2),
                         trans_mat=tm)

and e.g. sample data from this model::

    true_states, y = my_hmm.simulate(100)

(This works because, again, `HMM` is a subclass of ``StateSpaceModels``).

.. warning::
   Since `HMM` is a subclass of `StateSpaceModel`, method `PY` has the same
   signature as in its parent class, but argument `xp` is not used. In other
   words, you cannot specify a HMM model where $Y_t$ would depend on both
   $X_t$ and $X_{t-1}$ (unlike in the general case).

Running the Baum-Welch algorithm
================================

Class `BaumWelch` is instantiated as follows::

    bw = BaumWelch(hmm=my_hmm, data=y)

To actually run the algorithm, one must invoke the appropriate methods, e.g.::

    bw.forward()  # computes the filtering probs
    bw.backward()  # computes the marginal smoothing probs
    bw.sample(N=30)  # generate 30 smoothing trajectories

If you invoke either ``backward`` or ``sample`` directly, the forward pass will be
run first. The output of the forward and backward passes are attributes of
object `bw`, which are lists of K-length numpy arrays. For instance,
`self.filt` is a list of arrays containing the filtering probabilities; see the
documentation of ``BaumWelch`` for more details.


Running the forward pass step by step
=====================================

A ``BaumWelch`` object is an iterator; each iteration performs a single step of
the forward pass. It is thus possible for the user to run the forward pass step
by step::

    next(bw)  # performs one step of the forward pass

This may be useful in a variety of scenarios, such as when data are acquired on
the fly (in that case, modify attribute `self.data` directly), or when one
wants to perform the smoothing pass at different times; in particular::

    bw = BaumWelch(hmm=mh_hmm, data=y)
    for t, _ in enumerate(y):
        bw.step()
        bw.backward()
        ## save the results in bw.smth somewhere

would compute all the intermediate smoothing distributions (for data $Y_0$,
then $Y_{0:1}$, and so on). This is expensive, of course (cost is O(T^2)).
"""

from __future__ import division, print_function

import numpy as np

from particles import resampling as rs
from particles import distributions as dists
from particles import state_space_models as ssms


class HMM(ssms.StateSpaceModel):
    """Base class for hidden Markov models.

    To define a HMM, subclass HMM and define method PY.
    See module hmm for more information (and Chapter 6 of the book).
    """

    default_params = {"init_dist": None, "trans_mat": None}

    def __init__(self, **kwargs):
        ssms.StateSpaceModel.__init__(self, **kwargs)
        if self.trans_mat is None:
            raise ValueError("Transition Matrix is missing")
        self.dim = self.trans_mat.shape[0]
        if self.init_dist is None:
            self.init_dist = np.full(self.dim, 1.0 / self.dim)
        err_msg = "Wrong shape for trans_mat or init_dist"
        assert self.trans_mat.shape == (self.dim, self.dim), err_msg
        assert self.init_dist.shape == (self.dim,), err_msg

    def PX0(self):
        return dists.Categorical(p=self.init_dist)

    def PX(self, t, xp):
        return dists.Categorical(p=self.trans_mat[xp, :])


class GaussianHMM(HMM):
    """Gaussian HMM: :math:`Y_t|X_t=k \sim N(\mu_k, \sigma_k^2)`"""

    default_params = {"mus": None, "sigmas": None}
    default_params.update(HMM.default_params)

    def PY(self, t, xp, x):
        return dists.Normal(loc=self.mus[x], scale=self.sigmas[x])


class BaumWelch(object):
    """Baum-Welch filtering/smoothing algorithm.

    Parameters
    ----------
    hmm:   HMM object
        the hidden Markov model of interest
    data:  list-like
        the data

    Attributes
    ----------
    filt: list of numpy arrays
        filtering probabilities (computed during forward pass)
    pred: list of numpy arrays
        predictive probabilities (computed during forward pass)
    logpyt: list of scalars
        log density of Y_t given Y_{0:t-1}
    smth: list of numpy arrays
        smoothing probabilities (computed during the backward pass)

    Note
    ----
    To define a given hidden Markov model, one must subclass HMM; see
    documentation of ``HMM`` for more details. The considered HMM must be such that:
    1. Y_t depends on X_t only (not on X_{t-1}).
    2. Markov chain {X_t} is homogeneous (i.e. X_t|X_{t-1} does not depend on
    t).
    """

    def __init__(self, hmm=None, data=None):
        self.hmm = hmm
        self.data = data
        self.pred, self.filt, self.logpyt, self.logft = [], [], [], []

    @property
    def t(self):
        return len(self.filt)

    def pred_step(self):
        if self.filt:
            p = np.matmul(self.filt[-1], self.hmm.trans_mat)
        else:
            p = self.hmm.init_dist
        self.pred.append(p)

    def filt_step(self, t, yt):
        emis = self.hmm.PY(t, None, np.arange(self.hmm.dim)).logpdf(yt)
        lp = np.log(self.pred[-1]) + emis
        logpyt = rs.log_sum_exp(lp)
        f = np.exp(lp - logpyt)
        self.logft.append(emis)
        self.logpyt.append(logpyt)
        self.filt.append(f)

    def __next__(self):
        try:
            yt = self.data[self.t]
        except IndexError:
            raise StopIteration
        self.pred_step()
        self.filt_step(self.t, yt)

    def next(self):
        return self.__next__()  # Python 2 compatibility

    def __iter__(self):
        return self

    def forward(self):
        """Forward recursion.

        Upon completion, the following lists of length T are available:
        * filt: filtering probabilities
        * pred: predicting probabilities
        * logpyt: log-likelihood factor, i.e. log of p(y_t|y_{0:t-1})
        """
        for _ in self:
            pass

    def backward(self):
        """Backward recursion.

        Upon completion, the following list of length T is available:
        * smth: marginal smoothing probabilities

        Note
        ----
        Performs the forward step in case it has not been performed before.
        """
        if not self.filt:
            self.forward()
        self.smth = [self.filt[-1]]
        log_trans = np.log(self.hmm.trans_mat)
        ctg = np.zeros(self.hmm.dim)  # cost to go (log-lik of y_{t+1:T} given x_t=k)
        for filt, next_ft in reversed(list(zip(self.filt[:-1], self.logft[1:]))):
            new_ctg = np.empty(self.hmm.dim)
            for k in range(self.hmm.dim):
                new_ctg[k] = rs.log_sum_exp(log_trans[k, :] + next_ft + ctg)
            ctg = new_ctg
            smth = rs.exp_and_normalise(np.log(filt) + ctg)
            self.smth.append(smth)
        self.smth.reverse()

    def run(self):
        self.forward()
        self.backward()

    def sample(self, N=1):
        """Sample N trajectories from the posterior.

        Note
        ----
        Performs the forward step in case it has not been performed.

        """
        if not self.filt:
            self.forward()
        paths = np.empty((len(self.filt), N), np.int64)
        paths[-1, :] = rs.multinomial(self.filt[-1], M=N)
        log_trans = np.log(self.hmm.trans_mat)
        for t, f in reversed(list(enumerate(self.filt[:-1]))):
            for n in range(N):
                probs = rs.exp_and_normalise(log_trans[:, paths[t + 1, n]] + np.log(f))
                paths[t, n] = rs.multinomial_once(probs)
        return paths
