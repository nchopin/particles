# -*- coding: utf-8 -*-

"""
Overview
========

This module implements "summary collectors", that is, objects that 
collect at every time t certain summaries of the particle system. An 
important application is **on-line smoothing**. However, the idea is a 
bit more general that that. Here is a simple example:: 

    import particles
    # ...
    # define some_fk_model 
    # ... 
    my_alg = particles.SMC(fk=some_fk_model, N=100, moments=True,
                        naive_online_smooth=True)
    my_alg.run() 
    print(my_alg.summaries.moments)  # print a list of moments
    print(my_alg.summaries.naive_online_smooth)  # print a list of estimates

Once the algorithm is run, the object `my_alg.summaries` contains the
computed summaries, stored in lists of length T (one component for each
iteration t). 

Turning off summary collection
==============================

You may set option ``summaries`` of class ``SMC`` to False to avoid collecting 
any summary. This might be useful in cases when you need to keep a large number 
of SMC objects in memory (as in SMC^2). In that case, even the default
summaries (see below) might take too much space. 
    
Default summaries
=================

By default, the following summaries are collected: 
    * ``ESSs``: ESS at each iteration;  
    * ``rs_flags``: whether resampling was triggered or not at each time t; 
    * ``logLts``: log-likelihood estimates. 

Computing moments
=================

To compute moments (weighted averages, or other functions of the particle 
sample), use option ``moments`` as follows:: 

    my_alg = particles.SMC(fk=some_fk_model, N=100, moments=mom_func)
                        
where ``mom_func`` is a function with the following signature:: 

    def mom_func(W, X):
        return np.average(X, weights=W)

If option ``moments`` is set to ``True``, the default moments  are computed. 
For instance, for a ``FeynmanKac`` object  derived from a state-space model, 
the default moments at time t consist of a dictionary, with keys 'mean', and 
'var', containing the particle estimates (at time t) of the filtering mean 
and variance. 

It is possible to define different defaults for the moments. To do so, 
override method `default_moments` of the considered FeynmanKac class:: 

    class Bootstrap_with_better_moments(Bootstrap):
        def summary(W, X): 
            return np.average(X**2, weights=W)
    # ...
    # define state-space model my_ssm
    # ...
    my_fk_model = Bootstrap_with_better_moments(ssm=my_ssm, data=data)
    my_alg = particles.SMC(fk=my_fk_model, N=100, moments=True)

In that case, ``my_fk_model.summaries.moments`` is a list of weighed averages
of the squares of the components of the particles. 

On-line smoothing 
=================

On-line smoothing is the task of approximating, at every time t, 
expectations of the form:

.. math::
    \mathbb{E}[\phi_t(X_{0:t}) | Y_{0:t} = y_{0:t}]

On-line smoothing is covered in Sections 11.1 and 11.3 in the book.

The following three algorithms are implemented: 

* ``online_smooth``: basic forward smoothing (carry forward full trajectories); 
  cost is O(N) but performance may be poor for large t. 
* ``ON2_online_smooth``: O(N^2) on-line smoothing. Expensive (cost is O(N^2), 
  so big increase of CPU time), but better performance. 
* ``'paris'``: on-line smoothing using Paris algorithm. (Warning: current 
  implementation is very slow, work in progress). 

These algorithms compute the smoothing expectation of a certain additive
function, that is a function of the form:

.. math::
    \phi_t(x_{0:t}) = \psi_0(x_0) + \psi_1(x_0, x_1) + ... + \psi_t(x_{t-1}, x_t)

The elementary function :math:`\psi_t` is specified by defining method 
`add_func` in considered state-space model. Here is an example:: 

    class ToySSM(StateSpaceModel):
        def PX0(self):
            ... # as usual, see module `state_space_model`
        def add_func(self, t, xp, x):  # xp means x_{t-1} (p=past)
            if t == 0:
                return x**2
            else:
                return (xp - x)**2

The reason why additive functions are specified in this way is that 
additive functions often depend on fixed parameters of the state-space model 
(which are available in the closure of the ``StateSpaceModel`` object, but 
not outside).  

The two first algorithms do not require any parameter::

    my_alg = particles.SMC(fk=some_fk_model, N=100, online_smooth=True)

Paris algoritm has an optional parameter ``Nparis``, which you may specify 
as follows::

    my_alg = particles.SMC(fk=some_fk_model, N=100, paris=5)

If option `paris` is set to True, then the default value (2) is used. 

Custom collectors (collectors defined by the user)
==================================================

Not implemented, but let me know if you have user cases in mind for this 
feature. 
"""

from __future__ import division, print_function

import numpy as np

from particles import resampling as rs

class Summaries(object):
    """Class to store and update summaries.

    Attribute ``summaries`` of ``SMC`` objects is an instance of this class. 
    """
    def __init__(self, **sum_options):
        # Python magic at its finest
        col_classes = {cls.summary_name: cls 
                       for cls in Collector.__subclasses__()}
        # default summaries
        sopts = {k: True for k in ['ESSs', 'rs_flags', 'logLts']}
        sopts.update(sum_options)
        self._collectors = []
        for key, val in sopts.items():
            if val:  # ignores if False or None
                arg = None if val is True else val
                col = col_classes[key](arg)
                self._collectors.append(col)
                self.__dict__[col.summary_name] = col.summary

    def collect(self, smc):
        for s in self._collectors:
            s.collect(smc)

class Collector(object):
    """Base class for collectors.

    To subclass `Collector`: 
        * define summary name in string `summary_name` (should not clash
          with other summary names)
        * implement method `fetch(self, smc)` which fetches (in object smc the 
          summary that must be collected 
    """
    @property
    def summary(self):
        return getattr(self, self.summary_name)

    @summary.setter
    def summary(self, s):
        setattr(self, self.summary_name, s)

    def __init__(self, arg):
        self.summary = []
        self.arg = arg

    def collect(self, smc):
        self.summary.append(self.fetch(smc))

class ESSCollector(Collector):
    summary_name = 'ESSs'
    def fetch(self, smc):
        return smc.wgts.ESS

class LogLikCollector(Collector):
    summary_name = 'logLts'
    def fetch(self, smc):
        return smc.logLt

class RSFlagsCollector(Collector):
    summary_name = 'rs_flags'
    def fetch(self, smc):
        return smc.rs_flag

class MomentsCollector(Collector):
    """Collect the moments (weighted mean and variance) of the particles, or
    some other moment (as specified by function func). 
    """
    summary_name = 'moments'
    def fetch(self, smc):
        f = smc.fk.default_moments if self.arg is None else self.arg
        return f(smc.W, smc.X)

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

class NaiveOnLineSmoother(Collector, OnlineSmootherMixin):
    summary_name = 'naive_online_smooth'
    
    def update(self, smc):
        self.Phi = self.Phi[smc.A] + smc.fk.add_func(smc.t, smc.Xp, smc.X)

class ON2OnlineSmoother(Collector, OnlineSmootherMixin):
    summary_name = 'ON2_online_smooth'

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

class ParisOnlineSmoother(Collector, OnlineSmootherMixin):
    summary_name = 'paris'

    def __init__(self, arg):
        self.paris = []
        self.nprop = [0.]
        self.Nparis = 2 if arg is None else arg

    def update(self, smc):
        prev_Phi = self.Phi.copy()
        mq = rs.MultinomialQueue(self.prev_W)
        nprop = 0
        for n in range(self.N):
            As = np.empty(self.Nparis, 'int')
            for m in range(self.Nparis):
                while True:
                    a = mq.dequeue(1)
                    nprop += 1
                    lp = (smc.fk.logpt(smc.t, self.prev_X[a], smc.X[n])
                            - smc.fk.upper_bound_log_pt(t))
                    if np.log(random.rand()) < lp:
                        break
                As[m] = a
            mod_Phi = (self.prev_Phi[As] 
                       + smc.fk.add_func(smc.t, self.prev_X[As], smc.X[n]))
            self.Phi[n] = np.average(mod_Phi, axis=0)
        self.nprop.append(nprop)

    def save_for_later(self, smc):
        self.prev_X = smc.X
        self.prev_W = smc.W
