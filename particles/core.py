# -*- coding: utf-8 -*-

"""
Core module.

Overview
========

This module defines the following core objects:

* `FeynmanKac`: the base class for Feynman-Kac models;
* `SMC`: the base class for SMC algorithms;
* `multiSMC`: a function to run a SMC algorithm several times, in
  parallel and/or with varying options.

You don't need to import this module: these objects
are automatically imported when you import the package itself::

    import particles
    help(particles.SMC)  # should work

Each of these three objects have extensive docstrings (click on the links
above if you are reading the HTML version of this file).  However, here is a
brief summary for the first two.

The FeynmanKac abstract class
=============================

A Feynman-Kac model is basically a mathematical model for the operations that
we want to perform when running a particle filter. In particular:

    * The distribution *M_0(dx_0)* says how we want to simulate the particles at
      time 0.
    * the Markov kernel *M_t(x_{t-1}, dx_t)* says how we want to simulate
      particle X_t at time t, given an ancestor X_{t-1}.
    * the weighting function *G_t(x_{t-1}, x_t)* says how we want to reweight
      at time t a particle X_t and its ancestor is X_{t-1}.

For more details on Feynman-Kac models and their properties, see Chapter 5 of
the book.

To define a Feynman-Kac model in particles, one should, in principle:

    (a) sub-class `FeynmanKac` (define a class that inherits from it)
        and define certain methods such as `M0`, `M`, `G`, see
        the documentation of `FeynmanKac` for more details;
    (b) instantiate (define an object that belongs to) that sub-class.

In many cases however, you do not need to do this manually:

    * module `state_space_models` defines classes that automatically
      generate the bootstrap, guided or auxiliary Feynman-Kac model associated
      to a given state-space model; see the documentation of that module.
    * Similarly, module `smc_samplers` defines classes that automatically
      generates `FeynmanKac` objects for SMC tempering, IBIS and so on. Again,
      check the documentation of that module.

That said, it is not terribly complicated to define manually a Feynman-Kac
model, and there may be cases where this might be useful. There is even a basic
example in the tutorials if you are interested.

SMC class
=========

`SMC` is the class that define SMC samplers. To get you started::

    import particles
    ... # define a FeynmanKac object in some way, see above
    pf = particles.SMC(fk=my_fk_model, N=100)
    pf.run()

The code above simply runs a particle filter with ``N=100`` particles for the
chosen Feynman-Kac model. When this is done, object ``pf`` contains several
attributes, such as:

    * ``X``: the current set of particles (at the final time);
    * ``W``: their weights;
    * ``cpu_time``: as the name suggests;
    * and so on.

`SMC` objects are iterators, making it possible to run the algorithm step by
step: replace the last line above by::

    next(step) # do iteration 0
    next(step) # do iteration 1
    pf.run() # do iterations 2, ... until completion (dataset is exhausted)

All options, minus ``model``, are optional. Perhaps the most important ones are:
    * ``qmc``: if set to True, runs SQMC (the quasi-Monte Carlo version of SMC)
    * ``resampling``: the chosen resampling scheme; see `resampling` module.
    * ``store_history``: whether we should store the particles at all iterations;
        useful in particular for smoothing, see `smoothing` module.

See the documentation of `SMC` for more details.

"""

from __future__ import division, print_function

from functools import wraps

import numpy as np

from particles import collectors
from particles import hilbert
from particles import qmc
from particles import resampling as rs
from particles import smoothing
from particles import utils

err_msg_missing_trans = """
    Feynman-Kac class %s is missing method logpt, which provides the log-pdf
    of Markov transition X_t | X_{t-1}. This is required by most smoothing
    algorithms."""


class FeynmanKac(object):
    """Abstract base class for Feynman-Kac models.

    To actually define a Feynman-Kac model, one must sub-class FeymanKac,
    and define at least the following methods:

        * `M0(self, N)`: returns a collection of N particles generated from the
          initial distribution M_0.
        * `M(self, t, xp)`: generate a collection of N particles at time t,
           generated from the chosen Markov kernel, and given N ancestors (in
           array xp).
        * `logG(self, t, xp, x)`: log of potential function at time t.

    To implement a SQMC algorithm (quasi-Monte Carlo version of SMC), one must
    define methods:

        * `Gamma0(self, u)`: deterministic function such that, if u~U([0,1]^d),
        then Gamma0(u) has the same distribution as X_0
        * `Gamma(self, xp, u)`: deterministic function that, if U~U([0,1]^d)
        then Gamma(xp, U) has the same distribution as kernel M_t(x_{t-1}, dx_t)
        for x_{t-1}=xp

    Usually, a collection of N particles will be simply a numpy array of
    shape (N,) or (N,d). However, this is not a strict requirement, see
    e.g. module `smc_samplers` and the corresponding tutorial in the on-line
    documentation.
    """
    mutate_only_after_resampling = False

    # by default, we mutate at every time t

    def __init__(self, T):
        self.T = T

    def _error_msg(self, meth):
        return 'method/property %s missing in class %s' % (
            meth, self.__class__.__name__)

    def M0(self, N):
        """Sample N times from initial distribution M_0 of the FK model"""
        raise NotImplementedError(self._error_msg('M0'))

    def M(self, t, xp):
        """Generate X_t according to kernel M_t, conditional on X_{t-1}=xp
        """
        raise NotImplementedError(self._error_msg('M'))

    def logG(self, t, xp, x):
        """Evaluates log of function G_t(x_{t-1}, x_t)"""
        raise NotImplementedError(self._error_msg('logG'))

    def Gamma0(self, u):
        """Deterministic function that transform a uniform variate of dimension
        d_x into a random variable with the same distribution as M0."""
        raise NotImplementedError(self._error_msg('Gamma0'))

    def Gamma(self, t, xp, u):
        """Deterministic function that transform a uniform variate of dimension
        d_x into a random variable with the same distribution as M(self, t, xp).
        """
        raise NotImplementedError(self._error_msg('Gamma'))

    def logpt(self, t, xp, x):
        """Log-density of X_t given X_{t-1}.
        """
        raise NotImplementedError(err_msg_missing_trans %
                                  self.__class__.__name__)

    @property
    def isAPF(self):
        """Returns true if model is an APF"""
        return 'logetat' in dir(self)

    def done(self, smc):
        """Time to stop the algorithm"""
        return smc.t >= self.T

    def default_moments(self, W, X):
        """Default moments (see module ``collectors``).

        Computes weighted mean and variance (assume X is a Numpy array).
        """
        return rs.wmean_and_var(W, X)

    def summary_format(self, smc):
        return 't=%i: resample:%s, ESS (end of iter)=%.2f' % (smc.t,
                                                              smc.rs_flag,
                                                              smc.wgts.ESS)

    def weights_obj(self, lw=None):
        return rs.Weights(lw=lw)


class SMC(object):
    """Metaclass for SMC algorithms.

       Parameters
       ----------
       fk: FeynmanKac object
           Feynman-Kac model which defines which distributions are
           approximated
       N: int, optional (default=100)
           number of particles
       qmc: bool, optional (default=False)
           if True use the Sequential quasi-Monte Carlo version (the two
           options resampling and ESSrmin are then ignored)
       resampling: {'multinomial', 'residual', 'stratified', 'systematic', 'ssp'}
           the resampling scheme to be used (see `resampling` module for more
           information; default is 'systematic')
       ESSrmin: float in interval [0, 1], optional
           resampling is triggered whenever ESS / N < ESSrmin (default=0.5)
       store_history: bool, int or callable (default=False)
           whether and when history should be saved; see module `smoothing`
       verbose: bool, optional
           whether to print basic info at every iteration (default=False)
       collect: list of collectors, or 'off' (for turning off summary collections)
           see module ``collectors``

       Attributes
       ----------

       t : int
          current time step
       X : typically a (N,) or (N, d) ndarray (but see documentation)
           the N particles
       A : (N,) ndarray (int)
          ancestor indices: A[n] = m means ancestor of X[n] has index m
       wgts: `Weights` object
           An object with attributes lw (log-weights), W (normalised weights)
           and ESS (the ESS of this set of weights) that represents
           the main (inferential) weights
       aux: `Weights` object
           the auxiliary weights (for an auxiliary PF, see FeynmanKac)
       cpu_time : float
           CPU time of complete run (in seconds)
       hist: `ParticleHistory` object (None if option history is set to False)
           complete history of the particle system; see module `smoothing`
       summaries: `Summaries` object (None if option summaries is set to False)
           each summary is a list of estimates recorded at each iteration. The
           following summaries are computed by default:
               + ESSs (the ESS at each time t)
               + rs_flags (whether resampling was performed or not at each t)
               + logLts (estimates of the normalising constants)
           Extra summaries may also be computed (such as moments and online
           smoothing estimates), see module `collectors`.

       Methods
       -------
       run():
           run the algorithm until completion
       step()
           run the algorithm for one step (object self is an iterator)

    """

    def __init__(self,
                 fk=None,
                 N=100,
                 qmc=False,
                 resampling="systematic",
                 ESSrmin=0.5,
                 store_history=False,
                 verbose=False,
                 collect=None):

        self.fk = fk
        self.N = N
        self.qmc = qmc
        self.resampling = resampling
        self.ESSrmin = ESSrmin
        self.verbose = verbose

        # initialisation
        self.t = 0
        self.rs_flag = False  # no resampling at time 0, by construction
        self.logLt = 0.
        self.wgts = fk.weights_obj()
        self.aux = None
        self.X, self.Xp, self.A = None, None, None

        # summaries computed at every t
        if collect == 'off':
            self.summaries = None
        else:
            self.summaries = collectors.Summaries(collect)
        self.hist = smoothing.generate_hist_obj(store_history, self)

    def __str__(self):
        return self.fk.summary_format(self)

    @property
    def W(self):
        return self.wgts.W

    def reset_weights(self):
        """Reset weights after a resampling step.
        """
        if self.fk.isAPF:
            lw = (rs.log_mean_exp(self.logetat, W=self.W)
                  - self.logetat[self.A])
            self.wgts = self.fk.weights_obj(lw=lw)
        else:
            self.wgts = self.fk.weights_obj()

    def setup_auxiliary_weights(self):
        """Compute auxiliary weights (for APF).
        """
        if self.fk.isAPF:
            self.logetat = self.fk.logeta(self.t - 1, self.X)
            self.aux = self.wgts.add(self.logetat)
        else:
            self.aux = self.wgts

    def generate_particles(self):
        if self.qmc:
            u = qmc.sobol(self.N, self.fk.du).squeeze()
            # squeeze: must be (N,) if du=1
            self.X = self.fk.Gamma0(u)
        else:
            self.X = self.fk.M0(self.N)

    def reweight_particles(self):
        self.wgts = self.wgts.add(self.fk.logG(self.t, self.Xp, self.X))

    def resample_move(self):
        self.rs_flag = bool(self.aux.ESS < self.N * self.ESSrmin)
        if self.rs_flag:  # if resampling
            self.A = self.aux.resample(self.resampling)
            self.Xp = self.X[self.A]
            self.reset_weights()
            self.X = self.fk.M(self.t, self.Xp)
        elif not self.fk.mutate_only_after_resampling:
            self.A = np.arange(self.N)
            self.Xp = self.X
            self.X = self.fk.M(self.t, self.Xp)

    def resample_move_qmc(self):
        self.rs_flag = True  # we *always* resample in SQMC
        u = qmc.sobol(self.N, self.fk.du + 1)
        tau = np.argsort(u[:, 0])
        self.h_order = hilbert.hilbert_sort(self.X)
        self.A = self.h_order[rs.inverse_cdf(u[tau, 0], self.aux.W[self.h_order])]
        self.Xp = self.X[self.A]
        v = u[tau, 1:].squeeze()
        #  v is (N,) if du=1, (N,d) otherwise
        self.X = self.fk.Gamma(self.t, self.Xp, v)
        self.reset_weights()

    def compute_summaries(self):
        if self.t > 0:
            prec_log_mean_w = self.log_mean
        self.log_mean_w = self.wgts.mean()
        if self.t == 0 or self.rs_flag:
            self.loglt = self.log_mean_w
        else:
            self.loglt = self.log_mean_w - prec_log_mean_w
        self.logLt += self.loglt
        if self.verbose:
            print(self)
        if self.hist:
            self.hist.save(self)
        # must collect summaries *after* history, because a collector (e.g.
        # FixedLagSmoother) may needs to access history
        if self.summaries:
            self.summaries.collect(self)

    def __next__(self):
        """One step of a particle filter.
        """
        if self.fk.done(self):
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.setup_auxiliary_weights()  # APF
            if self.qmc:
                self.resample_move_qmc()
            else:
                self.resample_move()
        self.reweight_particles()
        self.compute_summaries()
        self.t += 1

    def next(self):
        return self.__next__()  #  Python 2 compatibility

    def __iter__(self):
        return self

    @utils.timer
    def run(self):
        """Runs particle filter until completion.

           Note: this class implements the iterator protocol. This makes it
           possible to run the algorithm step by step::

               pf = SMC(fk=...)
               next(pf)  # performs one step
               next(pf)  # performs one step
               for _ in range(10):
                   next(pf)  # performs 10 steps
               pf.run()  # runs the remaining steps

           In that case, attribute `cpu_time` records the CPU cost of the last
           command only.
        """
        for _ in self:
            pass


####################################################


class _picklable_f(object):

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, **kwargs):
        pf = SMC(**kwargs)
        pf.run()
        return self.fun(pf)


@_picklable_f
def _identity(x):
    return x


def multiSMC(nruns=10, nprocs=0, out_func=None, collect=None, **args):
    """Run SMC algorithms in parallel, for different combinations of parameters.


    `multiSMC` relies on the `multiplexer` utility, and obeys the same logic.
    A basic usage is::

        results = multiSMC(fk=my_fk_model, N=100, nruns=20, nprocs=0)

    This runs the same SMC algorithm 20 times, using all available CPU cores.
    The output, ``results``, is a list of 20 dictionaries; a given dict corresponds
    to a single run, and contains the following (key, value) pairs:
        + ``'run'``: a run identifier (a number between 0 and nruns-1)
        + ``'output'``: the corresponding SMC object (once method run was completed)

    Since a `SMC` object may take a lot of space in memory (especially when
    the option ``store_history`` is set to True), it is possible to require
    `multiSMC` to store only some chosen summary of the SMC runs, using option
    `out_func`. For instance, if we only want to store the estimate
    of the log-likelihood of the model obtained from each particle filter::

        of = lambda pf: pf.logLt
        results = multiSMC(fk=my_fk_model, N=100, nruns=20, out_func=of)

    It is also possible to vary the parameters. Say::

        results = multiSMC(fk=my_fk_model, N=[100, 500, 1000])

    will run the same SMC algorithm 30 times: 10 times for N=100, 10 times for
    N=500, and 10 times for N=1000. The number 10 comes from the fact that we
    did not specify nruns, and its default value is 10. The 30 dictionaries
    obtained in results will then contain an extra (key, value) pair that will
    give the value of N for which the run was performed.

    It is possible to vary several arguments. Each time a list must be
    provided. The end result will amount to take a *cartesian product* of the
    arguments::

        results = multiSMC(fk=my_fk_model, N=[100, 1000], resampling=['multinomial',
                           'residual'], nruns=20)

    In that case we run our algorithm 80 times: 20 times with N=100 and
    resampling set to multinomial, 20 times with N=100 and resampling set to
    residual and so on.

    Finally, if one uses a dictionary instead of a list, e.g.::

        results = multiSMC(fk={'bootstrap': fk_boot, 'guided': fk_guided}, N=100)

    then, in the output dictionaries, the values of the parameters will be replaced
    by corresponding keys; e.g. in the example above, {'fk': 'bootstrap'}. This is
    convenient in cases such like this where the parameter value is some non-standard
    object.

    Parameters
    ----------
    * nruns: int, optional
        number of runs (default is 10)
    * nprocs: int, optional
        number of processors to use; if negative, number of cores not to use.
        Default value is 1 (no multiprocessing)
    * out_func: callable, optional
        function to transform the output of each SMC run. (If not given, output
        will be the complete SMC object).
    * collect: list of collectors, or 'off'
        this particular argument of class SMC may be a list, hence it is "protected"
        from Cartesianisation
    * args: dict
        arguments passed to SMC class (except collect)

    Returns
    -------
    A list of dicts

    See also
    --------
    `utils.multiplexer`: for more details on the syntax.
    """
    f = _identity if out_func is None else _picklable_f(out_func)
    return utils.multiplexer(f=f, nruns=nruns, nprocs=nprocs, seeding=True,
                             protected_args={'collect': collect},
                             **args)
