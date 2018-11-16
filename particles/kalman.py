# -*- coding: utf-8 -*-

r"""
A basic implementation of the Kalman filter (and smoother). 

Overview
=========

The Kalman filter/smoother is a well-known algorithm to compute recursively 
the expectation and variance of the filtering/smoothing distributions of 
a linear Gaussian model, i.e. a model of the form:

.. math::
    X_0 & \sim N(\mu_0,C_0) \\
    X_t & = F X_{t-1} + U_t, \quad   U_t \sim N(0, C_X) \\
    Y_t & = G X_t + V_t,     \quad   V_t \sim N(0, C_Y)

`Kalman` class
==============

The Kalman filter is implemented as a class, `Kalman`, with methods
`filter` and `smoother`. When instantiating the class, one passes 
as arguments the value of the parameters of the model above. Here is a simple 
example::

    import numpy as np

    # ... get some data
    F = np.array([[1., 0.3], [0., 1.]])
    kf = Kalman(F=F, covX=np.eye(2), covY=.01)
    fwd = kf.filter(data)

In this example, the states X_t are of dimension 2, the observations Y_t are
of dimension 1. These dimensions are inferred from arguments `covX` and `covY`. 
The other arguments are optional; if not present, they are set to default
values with the proper dimension. For instance, G was set to the identity
matrix of dimension 1 (that is, the scalar 1.). 

Object `fwd` is an instance of `KalmanOutputs`; it has the
following attributes: 

    * `fwd.pred`: prediction distributions: `fwd.pred.mean` and 
      `fwd.pred.cov` are (T,d) and (T, d, d) arrays containing the T mean
      vectors and the T covariance matrices of the prediction distributions.
    * `fwd.filt`: filtering distributions: same structure as for `fwd.pred`.
    * `filt.logpyts`: array containing the incremental log-likelihoods (i.e. 
      the log of density of Y_t given Y_{0:t-1}). 

Smoothing works along the same lines. One may perform smoothing directly as
follows::

    bwd = kf.smoother(data=data)

or, alternatively, since the Kalman smoother works by doing a backward pass
that involves quantities computed during the forward (filtering pass), 
you may "recycle" the results of the filter as follows:: 

    bwd = kf.smoother(out_fwd=fwd)

Both commands will return the same results, but the latter is faster because it
does not repeat the computations done during the forward pass. Object `bwd` is
again a `KalmanOutputs` object, with the same attributes as above, plus an
extra attribute, `smth`, which contains the smoothing means and covariance 
matrices. 


Functions to perform a single step
==================================

The module also defines functions that perform a single step of the forward 
or backward step. Some of these function makes it possible to perform such 
steps *in parallel* (e.g. for N predictive means). This may be useful in a
particle filter.  

"""

from __future__ import division, print_function

import numpy as np
from numpy.linalg import inv

from particles import distributions as dists

error_msg = "arguments of KalmanFilter.__init__ have inconsistent shapes"

#######################

# Idea: means and dta points are 1D, covs, F and G are 2D
# however, works too if means are (N,d)
# this is why computations are "transposed" for the means


def dotdot(a, b, c):
    return np.dot(np.dot(a, b), c)


def predict_step(F, covX, filt_mean, filt_cov):
    """Predictive step.
    
    Note
    ----
    filt_mean may either be of shape (dx,) or (N, dx); in the latter case 
    N predictive steps are performed in parallel. 
    """
    pred_mean = np.matmul(filt_mean, F.T)
    pred_cov = dotdot(F, filt_cov, F.T) + covX
    return (pred_mean, pred_cov)


def filter_step(G, covY, pred_mean, pred_cov, yt):
    """Filtering step.

    Note
    ----
    filt_mean may either be of shape (dx,) or (N, dx); in the latter case 
    N predictive steps are performed in parallel. 
    """
    # data prediction
    data_pred_mean = np.matmul(pred_mean, G.T)
    data_pred_cov = dotdot(G, pred_cov, G.T) + covY
    if covY.shape[0] == 1:
        logpyt = dists.Normal(loc=data_pred_mean,
                              scale=np.sqrt(data_pred_cov)).logpdf(yt)
    else:
        logpyt = dists.MvNormal(
            loc=data_pred_mean,
            cov=data_pred_cov).logpdf(yt)
    # filter
    residual = yt - data_pred_mean
    gain = dotdot(pred_cov, G.T, inv(data_pred_cov))
    filt_mean = pred_mean + np.matmul(residual, gain.T)
    filt_cov = pred_cov - dotdot(gain, G, pred_cov)
    return filt_mean, filt_cov, logpyt


def filter_step_asarray(G, covY, pred_mean, pred_cov, yt):
    # deals with the case where shape of input is (N,) ==> (N,1)
    pm = pred_mean[:, np.newaxis] if pred_mean.ndim == 1 else pred_mean
    filt_mean, filt_cov, logpyt = filter_step(G, covY, pm, pred_cov, yt)
    if pred_mean.ndim == 1:
        filt_mean.squeeze()
    return filt_mean, filt_cov, logpyt


def smoother_step(F, filt_mean, filt_cov, next_pred_mean, next_pred_cov,
                  next_smth_mean, next_smth_cov):
    """Smoothing step.
    """
    J = dotdot(filt_cov, F.T, inv(next_pred_cov))
    smth_cov = filt_cov + dotdot(J, next_smth_cov - next_pred_cov, J.T)
    smth_mean = filt_mean + np.matmul(next_smth_mean - next_pred_mean, J.T)
    return smth_mean, smth_cov

#############################


class MeansAndCovs(object):
    """Stores a collection of means and covariance matrices.
    """

    def __init__(self, d, T):
        self.means = np.empty((T, d))
        self.covs = np.empty((T, d, d))

    def __getitem__(self, t):
        return self.means[t, :], self.covs[t, :, :]

    def __setitem__(self, t, item):
        self.means[t, :] = item[0]
        self.covs[t, :, :] = item[1]


class KalmanOutputs(object):
    """Stores the output of the filter and smoother steps of class `Kalman`.

    Attributes
    ----------
    d: int
        dimension of the states
    T: int 
        number of time steps 
    logpyts: (T,) numpy array 
        the log-density of Y_t given Y_{0:t-1} for t=0, ..., T-1
    pred: MeansAndCovs objects: 
        + pred.means is a (T, d) array containing the T predictive means
        + pred.cov is a (T, d, d) array containing the T predictive covariance
          matrices 
    filt: `MeansAndCovs` object (as above) 
    smth: `MeansAndCovs` object (as above)
    """

    def __init__(self, d, T):
        self.d, self.T = d, T
        self.pred = MeansAndCovs(d, T)
        self.filt = MeansAndCovs(d, T)
        self.logpyts = np.empty(T)

    def set(self, t, pred_mean, pred_cov, filt_mean, filt_cov, logpyt):
        self.pred[t] = pred_mean, pred_cov
        self.filt[t] = filt_mean, filt_cov
        self.logpyts[t] = logpyt

    def add_smooth_means_covs(self):
        self.smth = MeansAndCovs(self.d, self.T)


class Kalman(object):
    """ Kalman filter/smoother for the linear Gaussian model above.
    """

    def __init__(self, F=None, G=None, covX=None,
                 covY=None, mu0=None, cov0=None):
        """
        Parameters
        ----------
        F: numpy array
            transition matrix, i.e. X_t = F X_{t-1} + noise (if None, set to 
            identity matrix) 
        G: numpy array 
            observation matrix, i.e. Y_t = G X_t + noise (if None set to 
            matrix of shape (dx, dy) with 1 on the diagonal)
        covX: numpy array 
            covariance matrix of state transition 
        covY: numpy array
            covariance matrix of observation equation 
        mu0: numpy array
            initial mean of the states (if None, set to a vector of zeros)
        cov0: numpy array
            initial cov of the states (if None, set to covX) 
        """
        self.covX, self.covY = np.atleast_2d(covX), np.atleast_2d(covY)
        self.dx, self.dy = self.covX.shape[0], self.covY.shape[0]
        self.mu0 = np.zeros(self.dx) if mu0 is None else mu0
        self.cov0 = self.covX if cov0 is None else np.atleast_2d(cov0)
        self.F = np.eye(self.dx) if F is None else np.atleast_2d(F)
        if G is None:
            self.G = np.zeros((self.dy, self.xy))
            for i in range(min(self.dx, self.dy)):
                self.G[i, i] = 1.
        else:
            self.G = np.atleast_2d(G)
        self.check_shapes()

    def check_shapes(self):
        """
        Check all dimensions are correct.
        """
        assert self.covX.shape == (self.dx, self.dx), error_msg
        assert self.covY.shape == (self.dy, self.dy), error_msg
        assert self.F.shape == (self.dx, self.dx), error_msg
        assert self.G.shape == (self.dy, self.dx), error_msg
        assert self.mu0.shape == (self.dx,), error_msg
        assert self.cov0.shape == (self.dx, self.dx), error_msg

    def filter(self, data):
        """ Forward recursion: compute mean/variance of filter and prediction.

        Parameters
        ----------
        data: list-like 
            the data

        Returns
        -------
        `KalmanOutputs` object
        """
        out = KalmanOutputs(self.dx, len(data))
        for t, yt in enumerate(data):
            if t == 0:
                pred_mean, pred_cov = self.mu0, self.cov0
            else:
                pred_mean, pred_cov = predict_step(self.F, self.covX,
                                                   filt_mean, filt_cov)
            filt_mean, filt_cov, logpyt = filter_step(self.G, self.covY,
                                                      pred_mean, pred_cov, yt)
            out.set(t, pred_mean, pred_cov, filt_mean, filt_cov, logpyt)
        return out

    def smoother(self, data=None, out_fwd=None):
        """Backward recursion: compute mean/variance of marginal smoother.

        Parameters
        ----------
        data: list-like
            data (ignored if `out_fwd` is present)
        out_fwd: KalmanOutputs object
            output of self.filter()

        Returns
        -------
        `KalmanOutputs` object, with extra attribute `smth`. 
        """
        out = self.filter(data) if out_fwd is None else out_fwd
        out.add_smooth_means_covs()
        out.smth[-1] = out.filt[-1]
        next_smth_mean, next_smth_cov = out.filt[-1]
        for t in range(out.T - 2, -1, -1):
            filt_mean, filt_cov = out.filt[t]
            next_pred_mean, next_pred_cov = out.pred[t + 1]
            smth_mean, smth_cov = smoother_step(self.F, filt_mean, filt_cov,
                                                next_pred_mean, next_pred_cov,
                                                next_smth_mean, next_smth_cov)
            next_smth_mean, next_smth_cov = smth_mean, smth_cov
            out.smth[t] = smth_mean, smth_cov
        return out

    def posterior_t(self, x, yt):
        """Computes moments of Gaussian dist p(X_t|X_{t-1}=x,Y_t=yt)
        for each element in an array x (of shape (N,dx))

        same output as filter_step_asarray:
            * filt_mean: an array of the same shape as x
            * filt_cov:  a dx*dx array
            * logpyt: an array of size N=x.xhape[0]

        """
        return filter_step_asarray(self.G, self.covY, np.matmul(x, self.F.T),
                                   self.covX, yt)

    def posterior_0(self, y0):
        """Computes moments of Gaussian dist p(X_0|Y_0=y_0)"""

        return filter_step(self.G, self.covY, self.mu0, self.cov0, y0)
