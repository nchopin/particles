# -*- coding: utf-8 -*-

"""Objects that collect summaries at each iteration of a SMC algorithm.

Overview
========

This module implements "summary collectors", that is, objects that collect at
every time t certain summaries of the particle system.  Important applications
are **fixed-lag smoothing** and **on-line smoothing**. However, the idea is a
bit more general that that. Here is a simple example::

    import particles
    from particles import collectors as col

    # ...
    # define some_fk_model
    # ...
    alg = particles.SMC(fk=some_fk_model, N=100, 
                        collect=[col.Moments(), col.Online_smooth_naive()])

    alg.run()
    print(alg.summaries.moments)  # list of moments
    print(alg.summaries.naive_online_smooth)  # list of smoothing estimates

Once the algorithm is run, the object `alg.summaries` contains the computed
summaries, stored in lists of length T (one component for each iteration t).
Note that:

    * argument ``collect`` expects a **list** of Collector objects;
    * the name of the collector classes are capitalised, e.g. ``Moments``;
    * by default, the name of the corresponding summaries are not, e.g.
      `pf.summaries.moments`.

Default summaries
=================

By default, the following summaries are collected (even if argument `collect`
is not used): 
    * ``ESSs``: ESS (effective sample size) at each iteration;
    * ``rs_flags``: whether resampling was triggered or not at each time t;
    * ``logLts``: log-likelihood estimates.

For instance::

    print(alg.summaries.ESSs) # sequence of ESSs

You may turn off summary collection entirey::

    alg = particles.SMC(fk=some_fk_model, N=100, collect='off')

This might be useful in very specific cases when you need to keep a large
number of SMC objects in memory (as in SMC^2). In that case, even the default
summaries might take too much space.

Computing moments
=================

To compute moments (functions of the current particle sample)::

    def f(W, X):  # expected signature for the moment function
        return np.average(X, weights=W)  # for instance

    alg = particles.SMC(fk=some_fk_model, N=100,
                           collect=[Moments(mom_func=f)])

Without an argument, i.e. ``Moments()``, the collector computes the default
moments defined by the `FeynmanKac` object; for instance, for a `FeynmanKac`
object derived from a state-space model, the default moments at time t consist
of a dictionary, with keys ``'mean'`` and ``'var'``, containing the particle
estimates (at time t) of the filtering mean and variance.

It is possible to define different defaults for the moments. To do so,
override method `default_moments` of the considered FeynmanKac class::

    from particles import state_space_models as ssms
    class Bootstrap_with_better_moments(ssms.Bootstrap):
        def default_moments(W, X):
            return np.average(X**2, weights=W)
    #  ...
    #  define state-space model my_ssm
    #  ...
    my_fk_model = Bootstrap_with_better_moments(ssm=my_ssm, data=data)
    alg = particles.SMC(fk=my_fk_model, N=100, moments=True)

In that case, ``my_fk_model.summaries.moments`` is a list of weighed averages
of the squares of the components of the particles.

Fixed-lag smoothing
===================

Fixed-lag smoothing means smoothing of the previous h states; that is, computing
(at every time t) expectations of

.. math::
    \mathbb{E}[\phi_t(X_{t-h:t}) | Y_{0:t} = y_{0:t}]

for a fixed integer h (at times t <= h; if t<h, replace h by t).

This requires keeping track of the h previous states for each particle;
this is achieved by using a rolling window history, by setting option
``store_history`` to an int equals to h+1 (the length of the trajectories)::

    alg = particles.SMC(fk=some_fk_model, N=100,
                           collect=[col.Fixed_lag_smooth(phi=phi)],
                           store_history=3)  # h = 2

See module `smoothing` for more details on rolling window and other types of
particle history. Function phi takes as an input the N particles, and returns
a numpy.array::

    def phi(X):
        return np.exp(X - 2.)

If no argument is provided, test function :math:`\varphi(x)=x` is used.

Note however that X is a deque of length at most h; it behaves like a list,
except that its length is always at most h + 1.  Of course this function
could simply return its arguments ``W`` and ``X``; in that case you simply
record the fixed-lag trajectories (and their weights) at every time t.

On-line smoothing
=================

On-line smoothing is the task of approximating, at every time t,
expectations of the form:

.. math::
    \mathbb{E}[\phi_t(X_{0:t}) | Y_{0:t} = y_{0:t}]

On-line smoothing is covered in Sections 11.1 and 11.3 in the book. Note that
on-line smoothing is typically restricted to *additive* functions :math:`\phi`,
see below.

The following collectors implement online-smoothing algorithms:

* `Online_smooth_naive`: basic forward smoothing (carry forward full trajectories);
  cost is O(N) but performance may be poor for large t.
* `Online_smooth_ON2`: O(N^2) on-line smoothing. Expensive (cost is O(N^2),
  so big increase of CPU time), but better performance.
* `Paris`: on-line smoothing using Paris algorithm. (Warning: current
  implementation is very slow, work in progress).

These algorithms compute the smoothing expectation of a certain additive
function, that is a function of the form:

.. math::
    \phi_t(x_{0:t}) = \psi_0(x_0) + \psi_1(x_0, x_1) + ... + \psi_t(x_{t-1}, x_t)

The elementary function :math:`\psi_t` is specified by defining method
`add_func` in considered state-space model. Here is an example::

    class BootstrapWithAddFunc(ssms.Bootstrap):
        def add_func(self, t, xp, x):  # xp means x_{t-1} (p=past)
            if t == 0:
                return x**2
            else:
                return (xp - x)**2

The reason why additive functions are specified in this way is that
additive functions often depend on fixed parameters of the state-space model
(which are available in the closure of the ``StateSpaceModel`` object, but
not outside).

The two first algorithms do not have any parameter, the third one (Paris) have one
(default: 2). To use them simultaneously::

    alg = particles.SMC(fk=some_fk_model, N=100,
                           collect=[col.Online_smooth_naive(),
                                    col.Online_smooth_ON2(),
                                    col.Paris(Nparis=5)])

Variance estimators
===================

The variance estimators of Chan & Lai (2013), Lee & Whiteley (2018), etc., are
implemented as collectors in  module `variance_estimators`; see the
documentation of that module for more details.


User-defined collectors
=======================

You may implement your own collectors as follows::

    import collectors

    class Toy(collectors.Collector):
        # optional, default: toy (same name without capital)
        summary_name = 'toy'

        # signature of the __init__ function (optional, default: {})
        signature = {phi=None}

        # fetch the quantity to collect at time t
        def fetch(self, smc):  # smc is the particles.SMC instance
            return np.mean(self.phi(smc.X))

Once this is done, you may use this new collector exactly as the other
ones::

    alg = particles.SMC(N=30, fk=some_fk_model, collect=[col.Moments(), Toy(phi=f)])

Then ``pf.summaries.toy`` will be a list of the summaries collected at each
time by the ``fetch`` method.

"""

from __future__ import division, print_function

from numpy import random
import numpy as np

from particles import resampling as rs


class Summaries(object):
    """Class to store and update summaries.

    Attribute ``summaries`` of ``SMC`` objects is an instance of this class.
    """

    def __init__(self, cols):
        self._collectors = [cls() for cls in default_collector_cls]
        if cols is not None:
            # call each collector to get a fresh instance
            self._collectors.extend(col() for col in cols)
        for col in self._collectors:
            setattr(self, col.summary_name, col.summary)

    def collect(self, smc):
        for col in self._collectors:
            col.collect(smc)


class Collector(object):
    """Base class for collectors.

    To subclass `Collector`:
    * implement method `fetch(self, smc)` which computes the summary that
      must be collected (from object smc, at each time).
    * (optionally) define class attribute `summary_name` (name of the collected summary;
      by default, name of the class, un-capitalised, i.e. Moments > moments)
    * (optionally) define class attribute `signature` (the signature of the
      constructor, by default, an empty dict)
    """
    signature = {}

    @property
    def summary_name(self):
        cn = self.__class__.__name__
        return cn[0].lower() + cn[1:]

    def __init__(self, **kwargs):
        self.summary = []
        for k, v in self.signature.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            if k in self.signature.keys():
                setattr(self, k, v)
            else:
                raise ValueError('Collector %s: unknown parameter %s' %
                                 (self.__class__.__name__, k))

    def __call__(self):
        # clone the object
        return self.__class__(**{k: getattr(self, k) for k in
                                 self.signature.keys()})

    def collect(self, smc):
        self.summary.append(self.fetch(smc))

# Default collectors
####################

class ESSs(Collector):
    summary_name = 'ESSs'
    def fetch(self, smc):
        return smc.wgts.ESS

class LogLts(Collector):
    def fetch(self, smc):
        return smc.logLt

class Rs_flags(Collector):
    def fetch(self, smc):
        return smc.rs_flag

default_collector_cls = [ESSs, LogLts, Rs_flags]

# Moments
#########

class Moments(Collector):
    """Collects empirical moments (e.g. mean and variance) of the particles.

    Moments are defined through a function phi with the following signature:

        def mom_func(W, X):
           return np.average(X, weights=W)  # for instance

    If no function is provided, the default moment of the Feynman-Kac class
    is used (mean and variance of the particles, see ``core.FeynmanKac``).
    """
    signature = {'mom_func': None}

    def fetch(self, smc):
        f = smc.fk.default_moments if self.mom_func is None else self.mom_func
        return f(smc.W, smc.X)

# Smoothing collectors
######################

class Fixed_lag_smooth(Collector):
    """Compute some function of fixed-lag trajectories.

    Must be used in conjunction with a rolling window history (store_history=k,
    with k an int, see module ``smoothing``).
    """
    signature = {'phi': None}

    def test_func(self, x):
        if self.phi is None:
            return x
        else:
            return self.phi(x)

    def fetch(self, smc):
        B = smc.hist.compute_trajectories()
        Xs = [X[B[i, :]] for i, X in enumerate(smc.hist.X)]
        return np.average(self.test_func(Xs), weights=smc.W)


class OnlineSmootherMixin(object):
    """Mix-in for on-line smoothing algorithms.
    """
    def fetch(self, smc):
        if smc.t == 0:
            self.Phi = smc.fk.add_func(0, None, smc.X)
        else:
            self.update(smc)
        out = np.average(self.Phi, axis=0, weights=smc.W)
        self.save_for_later(smc)
        return out

    def update(self, smc):
        """The part that varies from one (on-line smoothing) algorithm to the
        next goes here.
        """
        raise NotImplementedError

    def save_for_later(self, smc):
        """Save certain quantities that are required in the next iteration.
        """
        pass


class Online_smooth_naive(Collector, OnlineSmootherMixin):
    def update(self, smc):
        self.Phi = self.Phi[smc.A] + smc.fk.add_func(smc.t, smc.Xp, smc.X)


class Online_smooth_ON2(Collector, OnlineSmootherMixin):
    def update(self, smc):
        prev_Phi = self.Phi.copy()
        for n in range(smc.N):
            lwXn = (self.prev_logw
                    + smc.fk.logpt(smc.t, self.prev_X, smc.X[n]))
            WXn = rs.exp_and_normalise(lwXn)
            self.Phi[n] = np.average(
                prev_Phi + smc.fk.add_func(smc.t, self.prev_X, smc.X[n]),
                axis=0, weights=WXn)

    def save_for_later(self, smc):
        self.prev_X = smc.X
        self.prev_logw = smc.wgts.lw


class Paris(Collector, OnlineSmootherMixin):
    signature = {'Nparis': 2}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nprop = [0.]

    def update(self, smc):
        prev_Phi = self.Phi.copy()
        mq = rs.MultinomialQueue(self.prev_W)
        nprop = 0
        for n in range(self.N):
            As = np.empty(self.Nparis, dtype=np.int64)
            for m in range(self.Nparis):
                while True:
                    a = mq.dequeue(1)
                    nprop += 1
                    lp = (smc.fk.logpt(smc.t, self.prev_X[a], smc.X[n])
                          - smc.fk.upper_bound_log_pt(smc.t))
                    if np.log(random.rand()) < lp:
                        break
                As[m] = a
            mod_Phi = (prev_Phi[As]
                       + smc.fk.add_func(smc.t, self.prev_X[As], smc.X[n]))
            self.Phi[n] = np.average(mod_Phi, axis=0)
        self.nprop.append(nprop)

    def save_for_later(self, smc):
        self.prev_X = smc.X
        self.prev_W = smc.W
