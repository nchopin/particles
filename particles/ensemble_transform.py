# -*- coding: utf-8 -*-

"""
Ensemble Transform, as introduced by S. Reich in https://arxiv.org/abs/1210.0375

Overview
========

This module implements the optimal transport based ensemble transform introduced by S. Reich in 2013.
It depends on the particles weights as well as their location. This module depends on the POT library https://github.com/PythonOT/POT.

    from particles import ensemble_transform as et

Ensemble transform schemes
==================

All the ensemble transform schemes are implemented as classes with the following
signature::

    class EnsembleTransform:
        def __init__(self, *args, **kwargs):
            # Hyperparameters of the method if needed
            pass

        def resample(self, W, X):
            return X-like array

where:

  * ``W`` is a vector of N normalised weights (i.e. positive and summing to
    one).

  * ``X`` (ndarray) are the locations of the particles


Here the list of currently implemented resampling schemes:

* `EnsembleTransform`
* `EnsembleTransformFilter`

"""

import numpy as np
import ot
import scipy.linalg as lin

from .core import SMC


class EnsembleTransformBase(object):
    def resample(self, W, X):
        """docstring to do"""
        raise NotImplementedError


class EnsembleTransform(EnsembleTransformBase):
    """Method from S. Reich, A non-parametric ensemble transform method for Bayesian inference (2013)"""

    def __init__(self, metric="sqeuclidean", p=2):
        self.metric = metric
        self.p = p

    def resample(self, W, X):
        N = W.shape[0]
        uniform_weights = np.full_like(W, 1 / N)
        M = ot.utils.cdist(X.reshape(N, -1), X.reshape(N, -1), self.metric, p=self.p)
        return N * ot.lp.emd(uniform_weights, W, M) @ X

class EntropyRegularizedEnsembleTransform(EnsembleTransformBase):
    """Method from S. Reich, A non-parametric ensemble transform method for Bayesian inference (2013)"""

    def __init__(self, epsilon=0.1, metric="sqeuclidean", p=2):
        self.metric = metric
        self.p = p
        self.epsilon = epsilon

    def resample(self, W, X):
        N = W.shape[0]
        uniform_weights = np.full_like(W, 1 / N)
        M = ot.utils.cdist(X.reshape(N, -1), X.reshape(N, -1), self.metric, p=self.p)
        new_X = N * ot.bregman.sinkhorn(uniform_weights, W, M, self.epsilon) @ X
        return new_X

class CorrectedEntropyRegularizedEnsembleTransform(EnsembleTransformBase):
    """Method from S. Reich, A non-parametric ensemble transform method for Bayesian inference (2013)"""

    def __init__(self, epsilon=0.1, metric="sqeuclidean", p=2):
        self.metric = metric
        self.p = p
        self.epsilon = epsilon

    def resample(self, W, X):
        N = W.shape[0]
        uniform_weights = np.full_like(W, 1 / N)
        M = ot.utils.cdist(X.reshape(N, -1), X.reshape(N, -1), self.metric, p=self.p)
        transport = N * ot.bregman.sinkhorn(uniform_weights, W, M, self.epsilon)
        diag_W = np.diag(W)
        A = N * diag_W - transport.T.dot(transport)
        delta = lin.solve_continuous_are(-transport, np.eye(N), A, np.eye(N))
        new_X = (transport + delta).dot(X)
        return new_X


class EnsembleTransformFilter(SMC):
    """Subclass of SMC that implements ensemble transform techniques (that depend on the particles locations as well as
     their weights).
     Takes an EnsembleTransformBase instance and the SMC arguments (resampling is ignored).

    """

    def __init__(self,
                 ensemble_transform=EnsembleTransform(),
                 fk=None,
                 N=100,
                 qmc=False,
                 ESSrmin=0.5,
                 store_history=False,
                 verbose=False,
                 summaries=True,
                 **sum_options):
        super(EnsembleTransformFilter, self).__init__(fk, N, qmc, None, ESSrmin, store_history, verbose, summaries,
                                                      **sum_options)

        self.ensemble_transform = ensemble_transform

    def resample_move(self):
        self.rs_flag = self.aux.ESS < self.N * self.ESSrmin
        if self.rs_flag:  # if resampling
            self.Xp = self.ensemble_transform.resample(self.aux.W, self.X)
            self.reset_weights()
            self.X = self.fk.M(self.t, self.Xp)


    def resample_move_qmc(self):
        self.rs_flag = True  # we *always* resample in SQMC
        self.resample_move()
