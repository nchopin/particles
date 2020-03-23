# -*- coding: utf-8 -*-

r"""
Basic implementation of the Kalman filter (and smoother).

Overview
=========

The Kalman filter/smoother is a well-known algorithm for computing recursively
the filtering/smoothing distributions of a linear Gaussian model, i.e. a model
of the form:

.. math::
    X_0 & \sim N(\mu_0,C_0) \\
    X_t & = F X_{t-1} + U_t, \quad   U_t \sim N(0, C_X) \\
    Y_t & = G X_t + V_t,     \quad   V_t \sim N(0, C_Y)

Linear Gaussian models and the Kalman filter are covered in Chapter 7 of the
book.

MVLinearGauss class and subclasses
==================================

To define a specific linear Gaussian model, we instantiate class
`MVLinearGauss` (or one its subclass) as follows::

    import numpy as np
    from particles import kalman

    ssm = kalman.MVLinearGauss(F=np.ones((1, 2)), G=np.eye(2), covX=np.eye(2),
                               covY=.3)

where the parameters have the same meaning as above. It is also possible to
specify `mu0`and  `cov0` (the mean and covariance of the initial state $X_0$).
(See the documentation of the class for more details.)

Class `MVLinearGauss` is a sub-class of `StateSpaceModel` in module
`state_space_models`, so it inherits methods from its parent such as::

    true_states, data = ssm.simulate(30)

Class `MVLinearGauss` implements methods `proposal`, `proposal0` and `logeta`,
which correspond respectively to the optimal proposal distributions and
auxiliary function for a guided or auxiliary particle filter; see Chapter 11
and module `state_space_models` for more details. (That the optimal quantities
are tractable is, of course, due to the fact that the model is linear and
Gaussian.)

To define a univariate linear Gaussian model, you may want to use instead the
more conveniently parametrised class `LinearGauss` (which is a sub-class of
``MVLinearGauss`)::

    ssm = LinearGauss(rho=0.3, sigX=1., sigY=.2, sig0=1.)

which corresponds to model:

.. math::
    X_0                 & \sim N(0, \sigma_0^2) \\
    X_t|X_{t-1}=x_{t-1} & \sim N(\rho * X_{t-1},\sigma_X^2) \\
    Y_t |X_t=x_t        & \sim N(x_t, \sigma_Y^2)

Another sub-class of `MVLinearGauss` defined in this module is
`MVLinearGauss_Guarniero_etal`, which implements a particular class of linear
Gaussian models often used as a benchmark (after Guarniero et al, 2016).


`Kalman` class
==============

The Kalman filter is implemented as a class, `Kalman`, with methods
`filter` and `smoother`. When instantiating the class, one passes
as arguments the data, and an object that represents the considered model (i.e.
an instance of MvLinearGauss, see above)::

    kf = kalman.Kalman(ssm=ssm, data=data)
    kf.filter()

The second line implements the forward pass of a Kalman filter. The results are
stored as lists of `MeanAndCov` objects, that is, named tuples with attributes
'mean' and 'cov' that represent a Gaussian distribution. For instance::

    kf.filt[3].mean  # mean of the filtering distribution at time 3
    kf.pred[7].cov  # cov matrix of the predictive distribution at time 7

The forward pass also computes the log-likelihood of the data:

    kf.logpyt[5]  # log-density of Y_t | Y_{0:t-1} at time t=5

Smoothing works along the same lines::

    kf.smoother()

then object kf contains a list called smooth, which represents the successive
(marginal) smoothing distributions::

    kf.smth[8].mean  # mean of the smoothing dist at time 8

It is possible to call method `smoother` directly (without calling `filter`
first). In that case, the filtering step is automatically performed as a
preliminary step.

Kalman objects as iterators
===========================

It is possible to perform the forward pass step by step; in fact a `Kalman`
object is an iterator::

    kf = kalman.Kalman(ssm=ssm, data=data)
    next(kf)  # one step
    next(kf)  # one step

If you run the smoother after k such steps, you will obtain the smoothing
distribution based on the k first data-points. It is therefore possible to
compute recursively the successive smoothing distributions, but (a) at a high
CPU cost; and (b) at each time, you must save the results somewhere, as
attribute `kf.smth` gets written over and over.

Functions to perform a single step
==================================

The module also defines low-level functions that perform a single step of the
forward or backward step. Some of these function makes it possible to perform
such steps *in parallel* (e.g. for N predictive means).  The table below lists
these functions. Some of the required inputs are `MeanAndCov` objects, which
may be defined as follows::

    my_predictive_dist = kalman.MeanAndCov(mean=np.ones(2), cov=np.eye(2))

+----------------------------------------------+
| Function (with signature)                    |
+==============================================+
| predict_step(F, covX, filt)                  |
+----------------------------------------------+
| filter_step(G, covY, pred, yt)               |
+----------------------------------------------+
| filter_step_asarray(G, covY, pred, yt)       |
+----------------------------------------------+
| smoother_step(F, filt, next_pred, next_smth) |
+----------------------------------------------+

"""

from __future__ import division, print_function

import collections
import numpy as np
from numpy.linalg import inv

from particles import distributions as dists
from particles import state_space_models as ssms

error_msg = "arguments of KalmanFilter.__init__ have inconsistent shapes"

########################
# Low-level functions
########################

def dotdot(a, b, c):
    return np.dot(np.dot(a, b), c)

MeanAndCov = collections.namedtuple('MeanAndCov', 'mean cov')

def predict_step(F, covX, filt):
    """Predictive step of Kalman filter.

    Parameters
    ----------
    F:  (dx, dx) numpy array
        Mean of X_t | X_{t-1} is F * X_{t-1}
    covX: (dx, dx) numpy array
        covariance of X_t | X_{t-1}
    filt: MeanAndCov object
        filtering distribution at time t-1

    Returns
    -------
    pred: MeanAndCov object
        predictive distribution at time t

    Note
    ----
    filt.mean may either be of shape (dx,) or (N, dx); in the latter case
    N predictive steps are performed in parallel.
    """
    pred_mean = np.matmul(filt.mean, F.T)
    pred_cov = dotdot(F, filt.cov, F.T) + covX
    return MeanAndCov(mean=pred_mean, cov=pred_cov)


def filter_step(G, covY, pred, yt):
    """Filtering step of Kalman filter.

    Parameters
    ----------
    G:  (dy, dx) numpy array
        mean of Y_t | X_t is G * X_t
    covX: (dx, dx) numpy array
        covariance of Y_t | X_t
    pred: MeanAndCov object
        predictive distribution at time t

    Returns
    -------
    pred: MeanAndCov object
        filtering distribution at time t
    logpyt: float
        log density of Y_t | Y_{0:t-1}
    """
    # data prediction
    data_pred_mean = np.matmul(pred.mean, G.T)
    data_pred_cov = dotdot(G, pred.cov, G.T) + covY
    if covY.shape[0] == 1:
        logpyt = dists.Normal(loc=data_pred_mean,
                              scale=np.sqrt(data_pred_cov)).logpdf(yt)
    else:
        logpyt = dists.MvNormal(loc=data_pred_mean,
                                cov=data_pred_cov).logpdf(yt)
    # filter
    residual = yt - data_pred_mean
    gain = dotdot(pred.cov, G.T, inv(data_pred_cov))
    filt_mean = pred.mean + np.matmul(residual, gain.T)
    filt_cov = pred.cov - dotdot(gain, G, pred.cov)
    return MeanAndCov(mean=filt_mean, cov=filt_cov), logpyt


def filter_step_asarray(G, covY, pred, yt):
    """Filtering step of Kalman filter: array version.

    Parameters
    ----------
    G:  (dy, dx) numpy array
        mean of Y_t | X_t is G * X_t
    covX: (dx, dx) numpy array
        covariance of Y_t | X_t
    pred: MeanAndCov object
        predictive distribution at time t

    Returns
    -------
    pred: MeanAndCov object
        filtering distribution at time t
    logpyt: float
        log density of Y_t | Y_{0:t-1}

    Note
    ----
    This performs the filtering step for N distinctive predictive means:
    filt.mean should be a (N, dx) or (N) array; pred.mean in the output
    will have the same shape.

    """
    pm = pred.mean[:, np.newaxis] if pred.mean.ndim == 1 else pred.mean
    new_pred = MeanAndCov(mean=pm, cov=pred.cov)
    filt, logpyt = filter_step(G, covY, new_pred, yt)
    if pred.mean.ndim == 1:
        filt.mean.squeeze()
    return filt, logpyt


def smoother_step(F, filt, next_pred, next_smth):
    """Smoothing step of Kalman filter/smoother.

    Parameters
    ----------
    F:  (dx, dx) numpy array
        Mean of X_t | X_{t-1} is F * X_{t-1}
    filt: MeanAndCov object
        filtering distribution at time t
    next_pred: MeanAndCov object
        predictive distribution at time t+1
    next_smth: MeanAndCov object
        smoothing distribution at time t+1

    Returns
    -------
    smth: MeanAndCov object
        smoothing distribution at time t
    """
    J = dotdot(filt.cov, F.T, inv(next_pred.cov))
    smth_cov = filt.cov + dotdot(J, next_smth.cov - next_pred.cov, J.T)
    smth_mean = filt.mean + np.matmul(next_smth.mean - next_pred.mean, J.T)
    return MeanAndCov(mean=smth_mean, cov=smth_cov)


###############################
# State-space model classes
###############################

class MVLinearGauss(ssms.StateSpaceModel):
    """Multivariate linear Gaussian model.

    .. math::
        X_0 & \sim N(\mu_0, cov_0) \\
        X_t & = F * X_{t-1} + U_t, \quad   U_t\sim N(0, cov_X) \\
        Y_t & = G * X_t + V_t,     \quad   V_t \sim N(0, cov_Y)

    The only mandatory parameters are `covX` and `covY` (from which the
    dimensions dx and dy of, respectively, X_t, and Y_t, are deduced). The
    default values for the other parameters are:
        * `mu0`:: an array of zeros (of size dx)
        * `cov0`: cov_X
        * `F`: Identity matrix of shape (dx, dx)
        * `G`: (dx, dy) matrix such that G[i, j] = 1[i=j]

    Note
    ----
    The Kalman filter takes as an input an instance of this class (or one of
    its subclasses).
    """

    def __init__(self, F=None, G=None, covX=None, covY=None, mu0=None,
                 cov0=None):
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

    def PX0(self):
        return dists.MvNormal(loc=self.mu0, cov=self.cov0)

    def PX(self, t, xp):
        return dists.MvNormal(loc=np.dot(xp, self.F.T), cov=self.covX)

    def PY(self, t, xp, x):
        return dists.MvNormal(loc=np.dot(x, self.G.T), cov=self.covY)

    def proposal(self, t, xp, data):
        pred = MeanAndCov(mean=np.matmul(xp, self.F.T), cov=self.covX)
        f, _ = filter_step_asarray(self.G, self.covY, pred, data[t])
        return dists.MvNormal(loc=f.mean, cov=f.cov)

    def proposal0(self, data):
        pred0 = MeanAndCov(mean=self.mu0, cov=self.cov0)
        f, _ = filter_step(self.G, self.covY, pred0, data[0])
        return dists.MvNormal(loc=f.mean, cov=f.cov)

    def logeta(self, t, x, data):
        pred = MeanAndCov(mean=np.matmul(x, self.F.T), cov=self.covX)
        _, logpyt = filter_step_asarray(self.G, self.covY, pred, data[t+1])
        return logpyt

class MVLinearGauss_Guarniero_etal(MVLinearGauss):
    """Special case of a MV Linear Gaussian ssm from Guarnierio et al. (2016).

    .. math::
        G = cov_X = cov_Y = cov_0 = I_{d_x}

        F_{i, j} = \alpha^ { 1 + |i-j|}

    See `MVLinearGauss` for the definition of these quantities.

    Parameters
    ----------
    alpha:  float (default: 0.4)
        value of alpha
    dx: int (must be >1; default: 2)
        dimension of state-space

    Reference
    ---------
    Guarnierio et al (2016). The Iterated Auxiliary Particle Filter,
        arxiv:1511.06286, JASA.
    """
    def __init__(self, alpha=0.4, dx=2):
        F = np.empty((dx, dx))
        for i in range(dx):
            for j in range(dx):
                F[i, j] = alpha**(1 + abs(i - j))
        MVLinearGauss.__init__(self, F=F, G=np.eye(dx), covX=np.eye(dx),
                               covY=np.eye(dx))

class LinearGauss(MVLinearGauss):
    r"""A basic (univariate) linear Gaussian model.

        .. math::
            X_0                 & \sim N(0, \sigma_0^2) \\
            X_t|X_{t-1}=x_{t-1} & \sim N(\rho * X_{t-1},\sigma_X^2) \\
            Y_t |X_t=x_t        & \sim N(x_t, \sigma_Y^2)

        Note
        ----
        If parameter sigma0 is set to None, it is replaced by the quantity that
        makes the state process invariant:
        :math:`\sigma_X^2 / (1 - \rho^2)`
    """
    default_params = {'sigmaY': .2, 'rho': 0.9, 'sigmaX': 1.,
                      'sigma0': None}

    def __init__(self, **kwargs):
        ssms.StateSpaceModel.__init__(self, **kwargs)
        if self.sigma0 is None:
            self.sigma0 = self.sigmaX / np.sqrt(1. - self.rho**2)
        # arguments for Kalman
        MVLinearGauss.__init__(self, F=self.rho, G=1., covX=self.sigmaX**2,
                               covY= self.sigmaY**2, cov0=self.sigma0**2)

    def PX0(self):
        return dists.Normal(scale=self.sigma0)

    def PX(self, t, xp):
        return dists.Normal(loc=self.rho * xp, scale=self.sigmaX)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=self.sigmaY)

    def proposal0(self, data):
        sig2post = 1. / (1. / self.sigma0**2 + 1. / self.sigmaY**2)
        mupost = sig2post * (data[0] / self.sigmaY**2)
        return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))

    def proposal(self, t, xp, data):
        sig2post = 1. / (1. / self.sigmaX**2 + 1. / self.sigmaY**2)
        mupost = sig2post * (self.rho * xp / self.sigmaX**2
                             + data[t] / self.sigmaY**2)
        return dists.Normal(loc=mupost, scale=np.sqrt(sig2post))

    def logeta(self, t, x, data):
        law = dists.Normal(loc=self.rho * x,
                           scale=np.sqrt(self.sigmaX**2 + self.sigmaY**2))
        return law.logpdf(data[t + 1])

#################################
# Kalman filter/smoother class
#################################

class Kalman(object):
    """ Kalman filter/smoother.


    See the documentation of the module for more details.
    """

    def __init__(self, ssm=None, data=None):
        """
        Parameters
        ----------
        ssm: MVLinearGaussian object
            the linear Gaussian model of interest
        data: list-like
            the data
        """
        self.ssm = ssm
        self.data = data
        self.pred, self.filt, self.logpyt = [], [], []

    @property
    def t(self):
        return len(self.filt)

    def __next__(self):
        try:
            yt = self.data[self.t]
        except IndexError:
            raise StopIteration
        if not self.pred:
            self.pred += [MeanAndCov(mean=self.ssm.mu0, cov=self.ssm.cov0)]
        else:
            self.pred += [predict_step(self.ssm.F, self.ssm.covX, self.filt[-1])]
        new_filt, new_logpyt = filter_step(self.ssm.G, self.ssm.covY,
                                           self.pred[-1], yt)
        self.filt.append(new_filt)
        self.logpyt.append(new_logpyt)

    def next(self):
        return self.__next__()  # Python 2 compatibility

    def __iter__(self):
        return self

    def filter(self):
        """ Forward recursion: compute mean/variance of filter and prediction.
        """
        for _ in self:
            pass

    def smoother(self):
        """Backward recursion: compute mean/variance of marginal smoother.

        Performs the filter step in a preliminary step if needed.
        """
        if not self.filt:
            self.filter()
        self.smth = [self.filt[-1]]
        for t, f in reversed(list(enumerate(self.filt[:-1]))):
            self.smth += [smoother_step(self.ssm.F, f, self.pred[t + 1],
                                        self.smth[t + 1])]
        self.smth.reverse()
