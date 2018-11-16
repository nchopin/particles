# -*- coding: utf-8 -*-

r"""
Overview
========

This module defines:  

    1. the `StateSpaceModel` class, which lets you define a state-space model
       object has a Python object. 

    2. `FeynmanKac` classes that automatically define the Bootstrap, guided or
       auxililiary Feynman-Kac models associated to a given state-space model.

    3. several standard state-space models (stochastic volatility, 
       bearings-only tracking, and so on)

The recommended import is::

    from particles import state_space_models as ssm

For more details on state-space models and their properties, see Chapter 2 of
the book. 

Defining a state-space model
============================

Consider the following (simplified) stochastic volatility model: 

.. math::

     Y_t|X_t=x_t         &\sim N(0, e^{x_t})                   \\
     X_t|X_{t-1}=x_{t-1} &\sim N(0, \rho x_{t-1})             \\
     X_0                 &\sim N(0, \sigma^2 / (1 - \rho^2))

To define this particular model, we sub-class `StateSpaceModel` as follows:: 

    from particles import distributions as dists 

    class SimplifiedStochVol(ssm.StateSpaceModel):
        default_parameters = {'sigma': 1., 'rho': 0.8}  # optional
        def PY(self, t, xp, x):  # dist of Y_t at time t, given X_t and X_{t-1}
            return dists.Normal(scale=np.exp(x))
        def PX(self, t, xp):  # dist of X_t at time t, given X_{t-1}
            return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), 
                                scale=self.sigma)
        def PX0(self):  # dist of X_0 
            return dists.Normal(scale=self.sigma / np.sqrt(1. - self.rho**2))

Then we define a particular object (model) by instantiating this class::

    my_stoch_vol_model = SimplifiedStochVol(sigma=0.3, rho=0.9)

Hopefully, the code above is fairly transparent, but here are some noteworthy
details: 

    * probability distributions are defined through `ProbDist` objects, which
      are defined in module `distributions`. Most basic probability
      distributions are defined there; the syntax is very close to the syntax
      of the Scipy distributions; see module `distributions` for more details. 
    * `StochVol` actually defines a **parametric** class of models; note in
      particular that ``self.sigma`` and ``self.rho`` are **attributes** of
      this class that are set when we define object `my_stoch_vol_model`. 
      Default values for these parameters may be defined in a dictionary called 
      ``default_parameters``. When this dictionary is defined, any un-defined 
      parameter will be replaced by its default value::

          default_stoch_vol_model = SimplifiedStochVol()  # sigma=1., rho=0.8
    * There is no need to define a ``__init__()`` method, as it is already
      defined by the parent class. (This parent ``__init__()`` simply takes
      care of the default parameters, and may be overrided if needed.) 

Now that our state-space model is properly defined, what can we do with it? 
First, we may simulate states and data from it::

    x, y = my_stoch_vol_model.simulate(20)

This generates two lists of length 20: a list of states, X_0, ..., X_{19} and 
a list of observations (data-points), Y_0, ..., Y_{19}. 

Associated Feynman-Kac models
=============================

Now that our state-space model is defined, we obtain the associated Bootstrap 
Feynman-Kac model as follows:: 

    my_fk_model = ssm.Bootstrap(ssm=my_stoch_vol_model, data=y)

That's it! You are now able to run a bootstrap filter for this model:: 

    my_alg = particles.SMC(fk=my_fk_model, N=200)
    my_alg.run()

In case you are not clear about what are Feynman-Kac models, and how one may
associate a Feynman-Kac model to a given state-space model, see Chapter 5 of
the book.

To generate a guided Feynman-Kac model, we must provide proposal kernels (that
is, Markov kernels that define how we simulate particles X_t at time t, given 
an ancestor X_{t-1}):: 

    class StochVol_with_prop(StochVol):
        def proposal0(self, data):
            return dists.Normal(scale = self.sigma)
        def proposal(t, xp, data):  # a silly proposal
            return dists.Normal(loc=rho * xp + data[t], scale=self.sigma) 

    my_second_ssm = StochVol_with_prop(sigma=0.3)
    my_better_fk_model = ssm.Guided(ssm=my_second_ssm, data=y)
    # then run a SMC as above

Voilà! You have now implemented a guided filter. 

Of course, the proposal distribution above does not make much sense; we use it
to illustrate how proposals may be defined. Note in particular that it depends
on `data`, an object that represents the complete dataset. Hence the proposal
kernel at time ``t`` may depend on y_t but also y_{t-1}, or any other
datapoint. 

For auxiliary particle filters (APF), one must in addition specify auxiliary
functions, that is functions ::

    class StochVol_with_prop_and_aux_func(StochVol_with_prop):
        def logetat(self, t, x, data): 
            "Log of auxiliary function eta_t at time t"
            return -(x-data[t])**2

    my_third_ssm = StochVol_with_prop_and_aux_func()
    apf_fk_model = ssm.AuxiliaryPF(ssm=my_third_ssm, data=y)

Again, this particular choice does not make much sense, and is just given to
show how to define an auxiliary function. 

"""

from __future__ import division, print_function

import numpy as np

import particles
from particles import distributions as dists
from particles import kalman  # for linear Gaussian state space models

err_msg_missing_cst = """
    State-space model %s is missing method upper_bound_log_pt, which provides
    log of constant C_t, such that 
    p(x_t|x_{t-1}) <= C_t 
    This is required for smoothing algorithms based on rejection
    """

class StateSpaceModel(object):
    """Base class for state-space models. 

    To define a state-space model class, you must sub-class `StateSpaceModel`, 
    and at least define methods PX0, PX, and PY. Here is an example::

        class LinearGauss(StateSpaceModel):
            def PX0(self):  # The law of X_0
                return dists.Normal(scale=self.sigmaX)
            def PX(self, t, xp):  # The law of X_t conditional on X_{t-1} 
                return dists.Normal(loc=self.rho * xp, scale=self.sigmaY)
            def PY(self, t, xp, x):  # the law of Y_t given X_t and X_{t-1}
                return dists.Normal(loc=x, scale=self.sigmaY) 

    These methods return ``ProbDist`` objects, which are defined in the module
    `distributions`. The model above is a basic linear Gaussian SSM; it
    depends on parameters rho, sigmaX, sigmaY (which are attributes of the
    class). To define a particular instance of this class, we do:: 

        a_certain_ssm = LinearGauss(rho=.8, sigmaX=1., sigmaY=.2)

    All the attributes that appear in ``PX0``, ``PX`` and ``PY`` must be 
    initialised in this way. Alternatively, it it possible to define default 
    values for these parameters, by defining class attribute
    ``default_parameters`` to be a dictionary as follows:: 

        class LinearGauss(StateSpaceModel):
            default_parameters = {'rho': .9, 'sigmaX': 1., 'sigmaY': .1} 
            # rest as above 

    Optionally, we may also define methods: 

    * `proposal0(self, data)`: the (data-dependent) proposal dist at time 0
    * `proposal(self, t, xp, data)`: the (data-dependent) proposal distribution at
      time t, for X_t, conditional on X_{t-1}=xp
    * `logeta(self, t, x, data)`: the auxiliary weight function at time t

    You need these extra methods to run a guided or auxiliary particle filter.

    """

    def __init__(self, **kwargs):
        if hasattr(self, 'default_params'):
            self.__dict__.update(self.default_params)
        self.__dict__.update(kwargs)

    def _error_msg(self, method):
        return ('method ' + method + ' not implemented in class%s' % 
                self.__class__.__name__)

    @classmethod
    def state_container(cls, N, T):
        law_X0 = cls().PX0()
        dim = law_X0.dim
        shape = [N, T] 
        if dim>1:
            shape.append(dim)
        return np.empty(shape, dtype=law_X0.dtype)

    def PX0(self):
        "Law of X_0 at time 0"
        raise NotImplementedError(self._error_msg('PX0'))

    def PX(self, t, xp):
        " Law of X_t at time t, given X_{t-1} = xp"
        raise NotImplementedError(self._error_msg('PX'))

    def PY(self, t, xp, x):
        """Conditional distribution of Y_t, given the states. 
        """
        raise NotImplementedError(self._error_msg('PY'))

    def proposal0(self, data):
        raise NotImplementedError(self._error_msg('proposal0'))

    def proposal(self, t, xp, data):
        """Proposal kernel (to be used in a guided or auxiliary filter). 

        Parameter
        ---------
        t: int
            time
        x: 
            particles
        data: list-like
            data 
        """
        raise NotImplementedError(self._error_msg('proposal'))
    
    def upper_bound_log_pt(self, t):
        """Upper bound for log of transition density. 

        See `smoothing`. 
        """
        raise NotImplementedError(err_msg_missing_cst % self.__class__.__name__)

    def add_func(self, t, xp, x):
        """Additive function.""" 
        raise NotImplementedError(self._error_msg('add_func'))

    def simulate_given_x(self, x):
        lag_x = [None] + x[:-1]
        return [self.PY(t, xp, x).rvs()
                for t, (xp, x) in enumerate(zip(lag_x, x))]

    def simulate(self, T):
        """Simulate state and observation processes. 

        Parameters
        ----------
        T: int
            processes are simulated from time 0 to time T-1

        Returns
        -------
        x, y: lists
            lists of length T
        """
        x = []
        for t in range(T):
            law_x = self.PX0() if t == 0 else self.PX(t, x[-1])
            x.append(law_x.rvs())
        y = self.simulate_given_x(x)
        return x, y


class Bootstrap(particles.FeynmanKac):
    """Bootstrap Feynman-Kac formalism of a given state-space model.
    
    Parameters
    ----------

    ssm: `StateSpaceModel` object
        the considered state-space model
    data: list-like
        the data

    Returns
    -------
    `FeynmanKac` object 
        the Feynman-Kac representation of the bootstrap filter for the 
        considered state-space model 
    """
    def __init__(self, ssm=None, data=None):
        self.ssm = ssm
        self.data = data
        self.du = self.ssm.PX0().dim

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def M0(self, N):
        return self.ssm.PX0().rvs(size=N)

    def M(self, t, xp):
        return self.ssm.PX(t, xp).rvs(size=xp.shape[0])

    def logG(self, t, xp, x):
        return self.ssm.PY(t, xp, x).logpdf(self.data[t])

    def Gamma0(self, u):
        return self.ssm.PX0().ppf(u)

    def Gamma(self, t, xp, u):
        return self.ssm.PX(t, xp).ppf(u)

    def logpt(self, t, xp, x):
        """PDF of X_t|X_{t-1}=xp"""
        return self.ssm.PX(t, xp).logpdf(x)

    def upper_bound_trans(self, t):
        return self.ssm.upper_bound_log_pt(t)

    def add_func(self, t, xp, x):
        return self.ssm.add_func(t, xp, x)


class GuidedPF(Bootstrap):
    """Guided filter for a given state-space model.
    
    Parameters
    ----------

    ssm: StateSpaceModel object
        the considered state-space model
    data: list-like
        the data

    Returns
    -------
    FeynmanKac object 
        the Feynman-Kac representation of the bootstrap filter for the 
        considered state-space model 

    Note
    ----
    Argument ssm must implement methods `proposal0` and `proposal`. 
    """

    def M0(self, N):
        return self.ssm.proposal0(self.data).rvs(size=N)

    def M(self, t, xp):
        return self.ssm.proposal(t, xp, self.data).rvs(size=xp.shape[0])

    def logG(self, t, xp, x):
        if t == 0:
            return (self.ssm.PX0().logpdf(x)
                    + self.ssm.PY(0, xp, x).logpdf(self.data[0])
                    - self.ssm.proposal0(self.data).logpdf(x))
        else:
            return (self.ssm.PX(t, xp).logpdf(x)
                    + self.ssm.PY(t, xp, x).logpdf(self.data[t])
                    - self.ssm.proposal(t, xp, self.data).logpdf(x))

    def Gamma0(self, u):
        return self.ssm.proposal0(self.data).ppf(u)

    def Gamma(self, t, xp, u):
        return self.ssm.proposal(t, xp, self.data).ppf(u)


class AuxiliaryPF(GuidedPF):
    """Auxiliary particle filter for a given state-space model.
    
    Parameters
    ----------

    ssm: StateSpaceModel object
        the considered state-space model
    data: list-like
        the data

    Returns
    -------
    `FeynmanKac` object 
        the Feynman-Kac representation of the APF (auxiliary particle filter)
        for the considered state-space model 

    Note
    ----
    Argument ssm must implement methods `proposal0`, `proposal` and `logeta`.
    """

    def logeta(self, t):
        return self.ssm.logeta(t, self.data)


class AuxiliaryBootstrap(Bootstrap):
    """Base class for auxiliary bootstrap particle filters

    This is an APF, such that the proposal kernel is set to the transition 
    kernel of the model
    """

    def logeta(self, t):
        return self.ssm.logeta(t, self.data)


############################################
# EXAMPLES

class StochVol(StateSpaceModel):
    r"""Univariate stochastic volatility model. 

    .. math::

        X_0 & \sim N(\mu, \sigma^2/(1-\rho^2)) \\
        X_t & = \mu + \rho(X_{t-1}-\mu) + \sigma U_t, \quad U_t\sim N(0,1) \\
        Y_t|X_t=x_t & \sim N(0, e^{x_t}) \\
    """
    default_params = {'mu': -1.02, 'rho': 0.9702, 'sigma': .178}
    # values taken from Pitt & Shephard (1999)

    def sig0(self):
        """std of X_0"""
        return self.sigma / np.sqrt(1. - self.rho**2)

    def PX0(self):
        return dists.Normal(loc=self.mu, scale=self.sig0())

    def EXt(self, xp):
        """compute E[x_t|x_{t-1}]"""
        return (1. - self.rho) * self.mu + self.rho * xp

    def PX(self, t, xp):
        return dists.Normal(loc=self.EXt(xp), scale=self.sigma)

    def PY(self, t, xp, x):
        return dists.Normal(loc=0., scale=np.exp(0.5 * x))

    def _xhat(self, xst, sig, yt):
        return xst + 0.5 * sig**2 * (yt**2 * np.exp(-xst) - 1.)

    def proposal0(self, data):
        # Pitt & Shephard
        return dists.Normal(loc=self._xhat(0., self.sig0(), data[0]), 
                            scale=self.sig0())

    def proposal(self, t, xp, data):
        # Pitt & Shephard
        return dists.Normal(loc=self._xhat(self.EXt(xp),
                                           self.sigma, data[t]),
                            scale=self.sigma)

    def logeta(self, t, x, data):
        # Pitt & Shephard
        xst = self.EXt(x)
        xstmmu = xst - self.mu
        xhat = self._xhat(xst, self.sigma, data[t + 1])
        xhatmmu = xhat - self.mu
        return (0.5 / self.sigma**2 * (xhatmmu**2 - xstmmu**2)
                - 0.5 * self.data[t + 1]**2 * np.exp(-xst) * (1. + xstmmu))


class StochVolLeverage(StochVol):
    r"""Univariate stochastic volatility model with leverage effect.
    
    .. math::

        X_0                         & \sim N(\mu, \sigma^2/(1-\rho^2))     \\
        X_t|X_{t-1}=x_{t-1}         & \sim N(\mu + \rho (x-\mu), \sigma^2) \\
        Y_t|X_{t-1:t} =x_{t-1:t}    & \sim N( s*\phi*z, s^2*(1-\phi**2) ) 

    with :math:`s=\exp(x_t/2), z = [x_t-\mu-\rho*(x_{t-1}-\mu)]/\sigma

    Note
    ----

    This is equivalent to assuming that the innovations of X_t and Y_t
    are correlated, with correlation :math:`\phi`:

    .. math::

        X_t & = \mu + \rho(X_{t-1}-\mu) + \sigma U_t \\ 
        Y_t & = \exp(X_t/2) * V_t 

    and :math:`Cor(U_t, V_t) = \phi`

    Warning
    -------
    This class inherits from StochVol, but methods proposal, proposal0 
    and logeta were constructed for StochVol only, and should not work properly
    for this class. 
    """

    default_params = {'mu': -1.02, 'rho': 0.9702, 'sigma': .178, 'phi': 0.}

    def PY(self, t, xp, x):
        # u is realisation of noise U_t, conditional on X_t, X_{t-1}
        if t==0:
            u = (x - self.mu) / self.sig0()
        else:
            u = (x - self.EXt(xp)) / self.sigma
        std_x = np.exp(0.5 * x)
        return dists.Normal(loc=std_x * self.phi * u,
                            scale=std_x * np.sqrt(1. - self.phi**2))


class MVLinearGauss(StateSpaceModel):
    """Multivariate linear Gaussian model.

    .. math::
        X_0 & \sim N(\mu_0, cov_X) \\
        X_t & = F * X_{t-1} + U_t, \quad   U_t\sim N(0, cov_X) \\
        Y_t & = G * X_t + V_t,     \quad   V_t \sim N(0, cov_Y)

    Note
    ----
    When instantiated, this class initialise a Kalman filter, from which
    it is possible to compute the exact filtering and smoothing distributions.
    """

    default_parameters = {'F': None, 'G': None, 'covX': None, 'covY': None,
                          'mu0': None, 'cov0': None}

    def __init__(self, **kwargs):
        StateSpaceModel.__init__(self, **kwargs)
        self.kf = kalman.Kalman(F=self.F, G=self.G, covX=self.covX,
                                covY=self.covY, mu0=self.mu0, cov0=self.cov0)

    def kalman_filter(self, data):
        return self.kf.filter(data)

    def kalman_smoother(self, data):
        return self.kf.smoother(data=data)

    def PX0(self):
        return dists.MvNormal(loc=self.mu0, cov=self.covX)

    def PX(self, t, xp):
        return dists.MvNormal(loc=np.dot(xp, self.F.T), cov=self.covX)

    def PY(self, t, xp, x):
        return dists.MvNormal(loc=np.dot(x, self.G.T), cov=self.covY)

    def proposal(self, t, xp, data):
        fm, fc, _ = self.kf.posterior_t(xp, data[t])
        return dists.MvNormal(loc=fm, cov=fc)

    def proposal0(self, data):
        fm, fc, _ = self.kf.posterior_0(data[0])
        return dists.MvNormal(loc=fm, cov=fc)

    def logeta(self, t, x, data):
        _, _, logpyt = self.kf.posterior_t(x, data[t + 1])
        return logpyt


class MVLinearGauss_Guarniero_etal(MVLinearGauss):
    """Special case of a MV Linear Gaussian ssm from Guarnierio et al. 

    .. math:: 
        G = cov_X = cov_Y = cov_0 = I_{d_x}

        F_{i, j} = \alpha^ { 1 + |i-j|}

    See `MVLinearGauss` for the definition of these quantities. 

    Reference
    ---------
    Guarnierio et al (2015), arxiv:1511.06286
    """
    def __init__(self, alpha=0.4, dx=2, data=None):
        F = np.empty((dx, dx))
        for i in range(dx):
            for j in range(dx):
                F[i, j] = alpha**(1 + abs(i - j))
        MVLinearGauss.__init__(self, F=F, G=np.eye(dx), covX=np.eye(dx),
                               covY=np.eye(dx), mu0=np.zeros(dx), 
                               cov0=np.eye(dx), data=data) 

class LinearGauss(MVLinearGauss):
    r"""A simple (univariate) linear Gaussian model.

        .. math::
            X_0                 & \sim N(0, \sigma_0^2) \\
            X_t|X_{t-1}=x_{t-1} & \sim N(\rho * X_{t-1},\sigma_X^2) \\
            Y_t |X_t=x_t        & \sim N(x_t, \sigma_Y^2)

        Parameters
        ----------
        sigma0 
        sigmaX
        sigmaY
        rho

        Note
        ----
        If parameter sigma0 is set to None, it is replaced by 
        :math:`\sigma_X^2 / (1 - \rho^2)` 
    """
    default_params = {'sigmaY': .2, 'rho': 0.9, 'sigmaX': 1., 
                      'sigma0': None}

    def __init__(self, **kwargs):
        StateSpaceModel.__init__(self, **kwargs)
        if self.sigma0 is None:
            self.sigma0 = self.sigmaX / np.sqrt(1. - self.rho**2)
        self.kf = kalman.Kalman(F=self.rho, G=1., covX=self.sigmaX**2,
                                covY=self.sigmaY**2,
                                cov0=self.sigma0**2)
        # methods kalman_filter and kalman_smoother inherited from
        # MVLinearGauss

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

###############################


class Gordon_etal(StateSpaceModel):
    r"""Popular toy example that appeared initially in Gordon et al (1993).

    .. math:: 

        X_0 & \sim N(0, 2^2) \\ 
        X_t & = b X_{t-1} + c X_{t-1}/(1+X_{t-1}^2) + d*\cos(e*(t-1)) + \sigma_X V_t, \quad V_t \sim N(0,1) \\
        Y_t|X_t=x_t         & \sim N(a*x_t^2, 1)
    """
    default_params = {'a': 0.05, 'b': .5, 'c': 25., 'd': 8., 'e': 1.2,
                      'sigmaX': 3.162278}  # = sqrt(10)

    def PX0(self):
        return dists.Normal(scale=2.)

    def PX(self, t, xp):
        return dists.Normal(loc=self.b * xp + self.c * xp / (1. + xp**2)
                            + self.d * np.cos(self.e * (t - 1)),
                            scale=self.sigmaX)

    def PY(self, t, xp, x):
        return dists.Normal(loc=self.a * x**2)

###############################


class Tracking(StateSpaceModel):
    """ Bearings-only tracking SSM.

    """
    default_params = {'sigmaX': 2.e-4, 'sigmaY': 1e-3,
                      'x0': np.array([3e-3, -3e-3, 1., 1.])}

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=self.x0[0], scale=self.sigmaX),
                               dists.Normal(loc=self.x0[1], scale=self.sigmaX),
                               dists.Dirac(loc=self.x0[2]),
                               dists.Dirac(loc=self.x0[3])
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale=self.sigmaX),
                               dists.Normal(loc=xp[:, 1], scale=self.sigmaX),
                               dists.Dirac(loc=xp[:, 0] + xp[:, 2]),
                               dists.Dirac(loc=xp[:, 1] + xp[:, 3])
                               )

    def PY(self, t, xp, x):
        angle = np.arctan(x[:, 3] / x[:, 2])
        angle[x[:, 2] < 0.] += np.pi
        return dists.Normal(loc=angle, scale=self.sigmaY)

###############################

class DiscreteCox(StateSpaceModel):
    r"""A discrete Cox model. 

    .. math::
        Y_t | X_t=x_t   & \sim Poisson(e^{x_t}) \\
        X_t             & = \mu + \phi(X_{t-1}-\mu) + U_t,   U_t ~ N(0,1) \\
        X_0             & \sim N(\mu, \sigma^2/(1-\phi**2)) 
    """
    default_params = {'mu': 0., 'sigma': 1., 'phi': 0.95}

    def PX0(self):
        return dists.Normal(loc=self.mu,
                            scale=self.sigma / np.sqrt(1. - self.phi**2))

    def PX(self, t, xp):
        return dists.Normal(loc=self.mu + self.phi * (xp - self.mu),
                            scale=self.sigma)

    def PY(self, t, xp, x):
        return dists.Poisson(rate=np.exp(x))

################################

class MVStochVol(StateSpaceModel):
    """Multivariate stochastic volatility model. 

    X_0 ~ N(mu,covX)
    X_t-mu = F*(X_{t-1}-mu)+U_t   U_t~N(0,covX)
    Y_t(k) = exp(X_t(k)/2)*V_t(k) for k=1,...,d
    V_t ~ N(0,corY)
    """
    default_params = {'mu': 0., 'covX': None, 'corY': None, 'F': None}  # TODO

    def offset(self):
        return self.mu - np.dot(self.F, mu)

    def PX0(self):
        return dists.MvNormal(loc=self.mu, cov=self.corX)

    def PX(self, t, xp):
        return dists.MvNormal(loc=np.dot(xp, self.F.T) + self.offset(),
                              cov=self.covX)

    def PY(self, t, xp, x):
        return dists.MvNormal(scale=np.exp(0.5 * x), cov=self.corY)

###############################

class ThetaLogistic(StateSpaceModel):
    r""" Theta-Logistic state-space model (used in Ecology). 

    .. math::

        X_0 & \sim N(0, 1) \\
        X_t & = X_{t-1} + \tau_0 - \tau_1 * \exp(\tau_2 * X_{t-1}) + U_t, \quad U_t \sim N(0, \sigma_X^2) \\
        Y_t & \sim X_t + V_t, \quad   V_t \sim N(0, \sigma_Y^2) 
    """
    default_params = {'tau0':.15, 'tau1':.12, 'tau2':.1, 'sigmaX': 0.47, 
                      'sigmaY': 0.39}  # values from Peters et al (2010)

    def PX0(self):
        return dists.Normal(loc=0., scale=1.)

    def PX(self, t, xp):
        return dists.Normal(loc=xp + self.tau0 - self.tau1 * 
                            np.exp(self.tau2 * xp), scale=self.sigmaX)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=self.sigmaY)

    def proposal0(self, data):
        return self.PX0().posterior(data[0], sigma=self.sigmaY)

    def proposal(self, t, xp, data):
        return self.PX(t, xp).posterior(data[t], sigma=self.sigmaY)
