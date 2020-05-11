# -*- coding: utf-8 -*-

"""
Probability distributions as Python objects.

Overview
========

This module lets users define probability distributions as Python objects.

The probability distributions defined in this module may be used:

  * to define state-space models (see module `state_space_models`);
  * to define a prior distribution, in order to perform parameter estimation
    (see modules `smc_samplers` and `mcmc`).

Univariate distributions
========================

The module defines the following classes of univariate continuous distributions:

=======================================  =====================
  class (with signature)                       comments
=======================================  =====================
Normal(loc=0., scale=1.)                 N(loc,scale^2) distribution
Logistic(loc=0., scale=1.)
Laplace(loc=0., scale=1.)
Beta(a=1., b=1.)
Gamma(a=1., b=1.)                        scale = 1/b
InvGamma(a=1., b=1.)                     Distribution of 1/X for X~Gamma(a,b)
Uniform(a=0., b=1.)                      uniform over interval [a,b]
Student(loc=0., scale=1., df=3)
TruncNormal(mu=0, sigma=1., a=0., b=1.)  N(mu, sigma^2) truncated to interval [a,b]
Dirac(loc=0.)                            Dirac mass at point *loc*
=======================================  =====================

and the following classes of univariate discrete distributions:

=======================================  =====================
  class (with signature)                       comments
=======================================  =====================
Poisson(rate=1.)                         Poisson distribution, with expectation ``rate``
Binomial(n=1, p=0.5)
Geometric(p=0.5)
=======================================  =====================

Note that all the parameters of these distributions have default values, e.g.::

    some_norm = Normal(loc=2.4)  # N(2.4, 1)
    some_gam = Gamma()  # Gamma(1, 1)

Transformed distributions
=========================

To further enrich the list of available univariate distributions, the module
lets you define **transformed distributions**, that is, the distribution of
Y=f(X), for a certain function f, and a certain base distribution for X.

+--------------------------------+--------------------------+
| class name (and signature)     | description              |
+================================+==========================+
+--------------------------------+--------------------------+
| LinearD(base_dist, a=1., b=0.) | Y = a * X + b            |
+--------------------------------+--------------------------+
| LogD(base_dist)                | Y = log(X)               |
+--------------------------------+--------------------------+
| LogitD(base_dist, a=0., b=1.)  | Y = logit( (X-a)/(b-a) ) |
+--------------------------------+--------------------------+

A quick example::

    from particles import distributions as dists
    d = dists.LogD(dists.Gamma(a=2., b=2.))  # law of Y=log(X), X~Gamma(2, 2)

.. note:: These transforms are often used to obtain random variables
   defined over the full real line. This is convenient in particular
   when implementing random walk Metropolis steps.

Multivariate distributions
==========================

Some of the univariate distributions actually have a `dim` keyword that
allows to define multivariate distributions::

    d = dists.Normal(loc=np.array([3., 2.]), dim=2)
    d.rvs(size=30)

The distribution above has two independent components, with respective means 3
and 2 (and variance 1 for both components).

The module also implements multivariate Normal distributions; see `MvNormal`.

Furthermore, the module provides two ways to construct multivariate
distributions from a collection of univariate distributions:

* `IndepProd`: product of independent distributions. May be used to
  define state-space models.

* `StructDist`: distributions for named variables; may be used to specify
  prior distributions; see modules `smc_samplers` and `mcmc` (and the
  corresponding tutorials).

Under the hood
==============

Probability distributions are represented as objects of classes that inherit
from base class `ProbDist`, and  implement the following methods:

* ``logpdf(self, x)``: computes the log-pdf (probability density function) at
  point ``x``;
* ``rvs(self, size=None)``: simulates ``size`` random variates; (if set to
  None, number of samples is either one if all parameters are scalar, or
  the same number as the common size of the parameters, see below);
* ``ppf(self, u)``: computes the quantile function (or Rosenblatt transform
  for a multivariate distribution) at point ``u``.

A quick example::

    some_dist = dists.Normal(loc=2., scale=3.)
    x = some_dist.rvs(size=30)  # a (30,) ndarray containing IID N(2, 3^2) variates
    z = some_dist.logpdf(x)  # a (30,) ndarray containing the log-pdf at x

By default, the inputs and outputs of these methods are either scalars or Numpy
arrays (with appropriate type and shape). In particular, passing a Numpy
array to a distribution parameter makes it possible to define "array
distributions". For instance::

    some_dist = dists.Normal(loc=np.arange(1., 11.))
    x = some_dist.rvs(size=10)

generates 10 Gaussian-distributed variates, with respective means 1., ..., 10.
This is how we manage to define "Markov kernels" in state-space models; e.g.
when defining the distribution of X_t given X_{t-1} in a state-space model::

    class StochVol(ssm.StateSpaceModel):
        def PX(self, t, xp, x):
            return stats.norm(loc=xp)
        ### ... see module state_space_models for more details

Then, in practice, in e.g. the bootstrap filter, when we generate particles
X_t^n, we call method ``PX`` and pass as an argument a numpy array of shape
(N,) containing the N ancestors.

..  note::
    ProbDist objects are roughly similar to the frozen distributions of package
    :package:`scipy.stats`. However, they are not equivalent. Using such a
    frozen distribution when e.g. defining a state-space model will return an
    error.

Posterior distributions
=======================

A few classes also implement a ``posterior`` method, which returns the posterior
distribution that corresponds to a prior set to ``self``, a model which is
conjugate for the considered class, and some data. Here is a quick example::

    from particles import distributions as dists
    prior = dists.InvGamma(a=.3, b=.3)
    data = random.randn(20)  # 20 points generated from N(0,1)
    post = prior.posterior(data)
    # prior is conjugate wrt model X_1, ..., X_n ~ N(0, theta)
    print("posterior is Gamma(%f, %f)" % (post.a, post.b))

Here is a list of distributions implementing posteriors:

============    =================== ==================
Distribution    Corresponding model comments
============    =================== ==================
Normal          N(theta, sigma^2),   sigma fixed (passed as extra argument)
TruncNormal     same
Gamma           N(0, 1/theta)
InvGamma        N(0, theta)
MvNormal        N(theta, Sigma)     Sigma fixed (passed as extra argument)
============    =================== ==================


Implementing your own distributions
===================================

If you would like to create your own univariate probability distribution, the
easiest way to do so is to sub-class `ProbDist`, for a continuous distribution,
or `DiscreteDist`, for a discrete distribution. This will properly set class
attributes ``dim`` (the dimension, set to one, for a univariate distribution),
and ``dtype``, so that they play nicely with `StructDist` and so on. You will
also have to properly define methods ``rvs``, ``logpdf`` and ``ppf``. You may
omit ``ppf`` if you do not plan to use SQMC (Sequential quasi Monte Carlo).


"""

from __future__ import division, print_function

from collections import OrderedDict  # see prior
import numpy as np
import numpy.random as random
import scipy.stats as stats
from scipy.linalg import cholesky, solve_triangular, inv

HALFLOG2PI = 0.5 * np.log(2. * np.pi)


class ProbDist(object):
    """Base class for probability distributions.

    To define a probability distribution class, subclass ProbDist, and define
    methods:

    * ``logpdf(self, x)``: the log-density at point x
    * ``rvs(self, size=None)``: generates *size* variates from distribution
    * ``ppf(self, u)``: the inverse CDF function at point u

    and attributes:

        * ``dim``: dimension of variates (default is 1)
        * ``dtype``: the dtype of inputs/outputs arrays (default is 'float64')

    """
    dim = 1  # distributions are univariate by default
    dtype = 'float64'  # distributions are continuous by default

    def shape(self, size):
        if size is None:
            return None
        else:
            return (size,) if self.dim == 1 else (size, self.dim)

    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=None):
        raise NotImplementedError

    def ppf(self, u):
        raise NotImplementedError

##############################
# location-scale distributions
##############################


class LocScaleDist(ProbDist):
    """Base class for location-scale distributions.
    """
    def __init__(self, loc=0., scale=1., dim=1):
        self.loc = loc
        self.scale = scale
        self.dim = dim


class Normal(LocScaleDist):
    """N(loc,scale^2) distribution.
    """
    def rvs(self, size=None):
        return random.normal(loc=self.loc, scale=self.scale,
                             size=self.shape(size))

    def logpdf(self, x):
        l = stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
        return l if self.dim == 1 else np.sum(l, axis=1)

    def ppf(self, u):
        return stats.norm.ppf(u, loc=self.loc, scale=self.scale)

    def posterior(self, x, sigma=1.):
        """Model is X_1,...,X_n ~ N(theta, sigma^2), theta~self, sigma fixed"""
        pr0 = 1. / self.scale**2  # prior precision
        prd = x.size / sigma**2  # data precision
        varp = 1. / (pr0 + prd)  # posterior variance
        mu = varp * (pr0 * self.loc + prd * x.mean())
        return Normal(loc=mu, scale=np.sqrt(varp))


class Logistic(LocScaleDist):
    """Logistic(loc,scale) distribution.
    """
    def rvs(self, size=None):
        return random.logistic(loc=self.loc, scale=self.scale,
                               size=self.shape(size))

    def logpdf(self, x):
        l = stats.logistic.logpdf(x, loc=self.loc, scale=self.scale)
        return l if self.dim == 1 else np.sum(l, axis=1)

    def ppf(self, u):
        return stats.logistic.ppf(u, loc=self.loc, scale=self.scale)


class Laplace(LocScaleDist):
    """Laplace(loc,scale) distribution.
    """

    def rvs(self, size=None):
        return random.laplace(loc=self.loc, scale=self.scale,
                              size=self.shape(size))

    def logpdf(self, x):
        l = stats.laplace.logpdf(x, loc=self.loc, scale=self.scale)
        return l if self.dim == 1 else np.sum(l, axis=1)

    def ppf(self, u):
        return stats.laplace.ppf(u, loc=self.loc, scale=self.scale)


################################
# Other continuous distributions
################################

class Beta(ProbDist):
    """Beta(a,b) distribution.
    """
    def __init__(self, a=1., b=1.):
        self.a = a
        self.b = b

    def rvs(self, size=None):
        return random.beta(self.a, self.b, size=size)

    def logpdf(self, x):
        return stats.beta.logpdf(x, self.a, self.b)

    def ppf(self, x):
        return stats.beta.ppf(x, self.a, self.b)


class Gamma(ProbDist):
    """Gamma(a,b) distribution, scale=1/b.
    """
    def __init__(self, a=1., b=1.):
        self.a = a
        self.b = b
        self.scale = 1. / b

    def rvs(self, size=None):
        return random.gamma(self.a, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.gamma.logpdf(x, self.a, scale=self.scale)

    def ppf(self, u):
        return stats.gamma.ppf(u, self.a, scale=self.scale)

    def posterior(self, x):
        """Model is X_1,...,X_n ~ N(0, 1/theta), theta ~ Gamma(a, b)"""
        return Gamma(a=self.a + 0.5 * x.size,
                     b=self.b + 0.5 * np.sum(x**2))


class InvGamma(ProbDist):
    """Inverse Gamma(a,b) distribution.
    """
    def __init__(self, a=1., b=1.):
        self.a = a
        self.b = b

    def rvs(self, size=None):
        return stats.invgamma.rvs(self.a, scale=self.b, size=size)

    def logpdf(self, x):
        return stats.invgamma.logpdf(x, self.a, scale=self.b)

    def ppf(self, u):
        return stats.invgamma.ppf(u, self.a, scale=self.b)

    def posterior(self, x):
        " Model is X_1,...,X_n ~ N(0, theta), theta ~ InvGamma(a, b) "
        return InvGamma(a=self.a + 0.5 * x.size,
                        b=self.b + 0.5 * np.sum(x**2))


class Uniform(ProbDist):
    """Uniform([a,b]) distribution.
    """
    def __init__(self, a=0, b=1.):
        self.a = a
        self.b = b
        self.scale = b - a

    def rvs(self, size=None):
        return random.uniform(low=self.a, high=self.b, size=size)

    def logpdf(self, x):
        return stats.uniform.logpdf(x, loc=self.a, scale=self.scale)

    def ppf(self, u):
        return stats.uniform.ppf(u, loc=self.a, scale=self.scale)


class Student(ProbDist):
    """Student distribution.
    """
    def __init__(self, df=3., loc=0., scale=1.):
        self.df = df
        self.loc = loc
        self.scale = scale

    def rvs(self, size=None):
        return stats.t.rvs(self.df, loc=self.loc, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.t.logpdf(x, self.df, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.t.ppf(u, self.df, loc=self.loc, scale=self.scale)


class Dirac(ProbDist):
    """Dirac mass.
    """
    def __init__(self, loc=0.):
        self.loc = loc

    def rvs(self, size=None):
        if isinstance(self.loc, np.ndarray):
            return self.loc.copy()
            # seems safer to make a copy here
        else:  # a scalar
            N = 1 if size is None else size
            return np.full(N, self.loc)

    def logpdf(self, x):
        return np.where(x==self.loc, 0., -np.inf)

    def ppf(self, u):
        return self.rvs(size=u.shape[0])


class TruncNormal(ProbDist):
    """Normal(mu, sigma^2) truncated to [a, b] interval.
    """
    def __init__(self, mu=0., sigma=1., a=0., b=1.):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.au = (a - mu) / sigma
        self.bu = (b - mu) / sigma

    def rvs(self, size=None):
        return stats.truncnorm.rvs(self.au, self.bu, loc=self.mu,
                                   scale=self.sigma, size=size)

    def logpdf(self, x):
        return stats.truncnorm.logpdf(x, self.au, self.bu, loc=self.mu,
                                      scale=self.sigma)

    def ppf(self, u):
        return stats.truncnorm.ppf(u, self.au, self.bu, loc=self.mu,
                                   scale=self.sigma)

    def posterior(self, x, s=1.):
        """Model is X_1,...,X_n ~ N(theta, s^2), theta~self, s fixed"""
        pr0 = 1. / self.sigma**2  # prior precision
        prd = x.size / s**2  # data precision
        varp = 1. / (pr0 + prd)  # posterior variance
        mu = varp * (pr0 * self.mu + prd * x.mean())
        return TruncNormal(mu=mu, sigma=np.sqrt(varp), a=self.a, b=self.b)

########################
# Discrete distributions
########################


class DiscreteDist(ProbDist):
    """Base class for discrete probability distributions.
    """
    dtype = 'int64'


class Poisson(DiscreteDist):
    """Poisson(rate) distribution.
    """
    def __init__(self, rate=1.):
        self.rate = rate

    def rvs(self, size=None):
        return random.poisson(self.rate, size=size)

    def logpdf(self, x):
        return stats.poisson.logpmf(x, self.rate)

    def ppf(self, u):
        return stats.poisson.ppf(u, self.rate)


class Binomial(DiscreteDist):
    """Binomial(n,p) distribution.
    """
    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p

    def rvs(self, size=None):
        return random.binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.binom.logpmf(x, self.n, self.p)

    def ppf(self, u):
        return stats.binom.ppf(u, self.n, self.p)


class Geometric(DiscreteDist):
    """Geometric(p) distribution.
    """
    def __init__(self, p=0.5):
        self.p = p

    def rvs(self, size=None):
        return random.geometric(self.p, size=size)

    def logpdf(self, x):
        return stats.geom.logpmf(x, self.p)

    def ppf(self, u):
        return stats.geom.ppf(u, self.p)

class NegativeBinomial(DiscreteDist):
    """Negative Binomial distribution.

    Parameters
    ----------
    n:  int, or array of ints
        number of failures until the experiment is run
    p:  float, or array of floats
        probability of success

    Note:
        Returns the distribution of the number of successes: support is
        0, 1, ...

    """
    def rvs(self, size=None):
        return random.negative_binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.nbinom.logpmf(x, self.p, self.n)

    def ppf(self, u):
        return stats.nbinom.ppf(u, self.p, self.n)


class Categorical(DiscreteDist):
    """Categorical distribution.

    Parameter
    ---------
    p:  (k,) or (N,k) float array
        vector(s) of k probabilities that sum to one
    """
    def __init__(self, p=None):
        self.p = p

    def logpdf(self, x):
        return np.log(self.p[x])

    def rvs(self, size=None):
        if self.p.ndim == 1:
            N = 1 if size is None else size
            u =random.rand(N)
            return np.searchsorted(np.cumsum(self.p), u)
        else:
            N = self.p.shape[0] if size is None else size
            u = random.rand(N)
            cp = np.cumsum(self.p, axis=1)
            return np.array([np.searchsorted(cp[i], u[i])
                             for i in range(N)])

#########################
# distribution transforms
#########################

class TransformedDist(ProbDist):
    """Base class for transformed distributions.

    A transformed distribution is the distribution of Y=f(X) for a certain
    function f, and a certain (univariate) base distribution for X.
    Must be sub-classed; see below.

    Example::

        dst = LogitD(Beta(3.,2.))

    returns a dist object corresponding to the distribution of
    Y=logit(X), for X~Beta(3, 2)

    """

    def __init__(self, base_dist):
        self.base_dist = base_dist

    def error_msg(self, method):
        return 'method %s not defined in class %s' % (method, self.__class__)

    def f(self, x):
        raise NotImplementedError(self.error_msg('f'))

    def finv(self, x):
        """ Inverse of f."""
        raise NotImplementedError(self.error_msg('finv'))

    def logJac(self, x):
        """ Log of Jacobian.

        Obtained by differentiating finv, and then taking the log."""
        raise NotImplementedError(self.error_msg('logJac'))

    def rvs(self, size=None):
        return self.f(self.base_dist.rvs(size=size))

    def logpdf(self, x):
        return self.base_dist.logpdf(self.finv(x)) + self.logJac(x)

    def ppf(self, u):
        return self.f(self.base_dist.ppf(u))


class LinearD(TransformedDist):
    """Distribution of Y = a*X + b.

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    a, b: float (a should be != 0)
    """
    def __init__(self, base_dist, a=1., b=0.):
        self.a, self.b = a, b
        self.base_dist = base_dist

    def f(self, x):
        return self.a * x + self.b

    def finv(self, x):
        return (x - self.b) / self.a

    def logJac(self, x):
        return -np.log(self.a)


class LogD(TransformedDist):
    """Distribution of Y = log(X).

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    """
    def f(self, x):
        return np.log(x)

    def finv(self, x):
        return np.exp(x)

    def logJac(self, x):
        return x


class LogitD(TransformedDist):
    """Distributions of Y=logit((X-a)/(b-a)).

    See base class `TransformedDist`.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X
    a, b: float
        interval [a, b] is the support of base_dist

    """

    def __init__(self, base_dist, a=0., b=1.):
        self.a, self.b = a, b
        self.base_dist = base_dist

    def f(self, x):
        p = (x - self.a) / (self.b - self.a)
        return np.log(p / (1. - p))  # use built-in?

    def finv(self, x):
        return self.a + (self.b - self.a) / (1. + np.exp(-x))

    def logJac(self, x):
        return np.log(self.b - self.a) + x - 2. * np.log(1. + np.exp(x))


############################
# Multivariate distributions
############################

class MvNormal(ProbDist):
    """Multivariate Normal distribution.

    Parameters
    ----------
    loc: ndarray
        location parameter (see below)
    scale: ndarray
        scale parameter (see below)
    cov: (d, d) ndarray
        covariance matrix (see below)

    Note
    ----
    The parametrisation used here is slightly unusual. In short,
    the following line::

        x = dists.MvNormal(loc=m, scale=s, cov=Sigma).rvs(size=30)

    is equivalent to::

        x = m + s * dists.MvNormal(cov=Sigma).rvs(size=30)

    The idea is that they are many cases when we may want to pass
    varying means and scales (but a fixed correlation matrix).

    dx (dimension of vectors x) is determined by matrix cov; for rvs,
    size must be (N, ), otherwise an error is raised.

    Notes:
    * if du<dx, fill the remaining dimensions by location
        (i.e. scale should be =0.)
    * cov does not need to be a correlation matrix; more generally
    > mvnorm(loc=x, scale=s, cor=C)
    correspond to N(m,diag(s)*C*diag(s))

    In addition, note that x and s may be (N, d) vectors;
    i.e for each n=1...N we have a different mean, and a different scale.
    """

    def __init__(self, loc=0., scale=1., cov=None):
        self.loc = loc
        self.scale = scale
        self.cov = cov
        err_msg = 'mvnorm: argument cov must be a dxd ndarray, \
                with d>1, defining a symmetric positive matrix'
        try:
            self.L = cholesky(cov, lower=True)  # L*L.T = cov
            self.halflogdetcor = np.sum(np.log(np.diag(self.L)))
        except:
            raise ValueError(err_msg)
        assert cov.shape == (self.dim, self.dim), err_msg

    @property
    def dim(self):
        return self.cov.shape[0]

    def linear_transform(self, z):
        return self.loc + self.scale * np.dot(z, self.L.T)

    def logpdf(self, x):
        z = solve_triangular(self.L, np.transpose((x - self.loc) / self.scale),
                             lower=True)
        # z is dxN, not Nxd
        if np.asarray(self.scale).ndim == 0:
            logdet = self.dim * np.log(self.scale)
        else:
            logdet = np.sum(np.log(self.scale), axis=-1)
        logdet += self.halflogdetcor
        return - 0.5 * np.sum(z * z, axis=0) - logdet - self.dim * HALFLOG2PI

    def rvs(self, size=None):
        if size is None:
            sh = np.broadcast(self.loc, self.scale).shape
            # sh=() when both loc and scale are scalars
            N = 1 if len(sh) == 0 else sh[0]
        else:
            N = size
        z = stats.norm.rvs(size=(N, self.dim))
        return self.linear_transform(z)

    def ppf(self, u):
        """
        Note: if dim(u) < self.dim, the remaining columns are filled with 0
        Useful in case the distribution is partly degenerate
        """
        N, du = u.shape
        if du < self.dim:
            z = np.zeros((N, self.dim))
            z[:, :du] = stats.norm.ppf(u)
        else:
            z = stats.norm.ppf(u)
        return self.linear_transform(z)

    def posterior(self, x, Sigma=None):
        """Posterior for model: X1, ..., Xn ~ N(theta, Sigma).

        Parameters
        ----------
        x: (n, d) ndarray
            data
        Sigma: (d, d) ndarray
            (fixed) covariance matrix in the model
        """
        n = x.shape[0]
        Sigma = np.eye(self.dim) if Sigma is None else Sigma
        Siginv = inv(Sigma)
        Qpost = inv(self.cov) + n * Siginv
        Sigpost = inv(Qpost)
        mupost = (np.matmul(Siginv, self.mean) +
                  np.matmu(Siginv, np.sum(x, axis=0)))
        return MvNormal(loc=mupost, cov=Sigpost)

##################################
# product of independent dists


class IndepProd(ProbDist):
    """Product of independent univariate distributions.

    The inputs/outputs of IndeProd are numpy ndarrays of shape (N,d), or (d),
    where d is the number of univariate distributions that are
    passed as arguments.

    Parameters
    ----------
    dists: list of `ProbDist` objects
        The probability distributions of each component

    Example
    -------
    To define a bivariate distribution::

        prior = IndepProd(Normal(scale=2.), Gamma(2., 3.))
        theta = prior.rvs(size=9)  # returns a (9, 2) ndarray

    Note
    ----
    This is used mainly to define multivariate state-space models,
    see module `state_space_models`.

    """
    def __init__(self, *dists):
        self.dists = dists
        if all(d.dtype == 'int64' for d in dists):
            self.dtype = 'int64'
        else:
            self.dtype = 'float64'

    @property
    def dim(self):
        return len(self.dists)

    def rvs(self, size=None):
        return np.stack([d.rvs(size=size) for d in self.dists], axis=1)

    def logpdf(self, x):
        return sum([d.logpdf(x[..., i]) for i, d in enumerate(self.dists)])

    def ppf(self, u):
        return np.stack([d.ppf(u[..., i]) for i, d in enumerate(self.dists)],
                        axis=1)


###################################
# structured array distributions
# (mostly to define prior distributions)
###################################

class Cond(ProbDist):
    """Conditional distributions.

    see StructDist
    """
    def __init__(self, law, dim=1, dtype='float64'):
        self.law = law
        self.dim = dim
        self.dtype = dtype

    def __call__(self, x):
        return self.law(x)

class StructDist(ProbDist):
    """A distribution such that inputs/outputs are structured arrays.

    A structured array is basically a numpy array with named fields.
    We use structured arrays to represent particles that are
    vectors of (named) parameters; see modules :mod:`smc_samplers`
    and :mod:`mcmc`. And we use StructDist to define prior distributions
    with respect to such parameters.

    To specify a distribution such that parameters are independent,
    we pass a dictionary::

        prior = StructDist({'mu':Normal(), 'sigma':Gamma(a=1., b=1.)})
        # means mu~N(0,1), sigma~Gamma(1, 1) independently
        x = prior.rvs(size=30)  # returns a stuctured array of length 30
        print(x['sigma'])  # prints the 30 values for sigma

    We may also define a distribution using a chain rule decomposition.
    For this, we pass an ordered dict, since the order of components
    become relevant::

        chain_rule = OrderedDict()
        chain_rule['mu'] = Normal()
        chain_rule['tau'] = Cond(lambda x: Normal(loc=x['mu'])
        prior = StructDist(chain_rule)
        # means mu~N(0,1), tau|mu ~ N(mu,1)

    In the third line, ``Cond`` is a ``ProbDist`` class that represents
    a conditional distribution; it is initialized with a function that
    returns for each ``x`` a distribution that may depend on fields in ``x``.

    Parameters
    ----------
    laws: dict or ordered dict (as explained above)
        keys are parameter names, values are `ProbDist` objects

    """

    def __init__(self, laws):
        if isinstance(laws, OrderedDict):
            self.laws = laws
        elif isinstance(laws, dict):
            self.laws = OrderedDict([(key, laws[key])
                                     for key in sorted(laws.keys())])
        else:
            raise ValueError('recdist class requires a dict or'
                             ' an ordered dict to be instantiated')
        formats = [str(law.dim) + law.dtype
                   for law in self.laws.values()]
        self.dtype = {'names': list(self.laws.keys()), 'formats': formats}
        # list added for python 3 compatibility

    def rvs(self, size=1):  # Default for size is 1, not None
        out = np.empty(size, dtype=self.dtype)
        for par, law in self.laws.items():
            cond_law = law(out) if callable(law) else law
            out[par] = cond_law.rvs(size=size)
        return out

    def logpdf(self, theta):
        l = 0.
        for par, law in self.laws.items():
            cond_law = law(theta) if callable(law) else law
            l += cond_law.logpdf(theta[par])
        return l
