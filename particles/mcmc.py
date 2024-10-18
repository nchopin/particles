"""
MCMC (Markov chain Monte Carlo) and related algorithms.

Overview
========

This module contains various classes that implement MCMC samplers:

    * `MCMC`: the base class for all MCMC samplers;

    * `GenericRWHM`: base class for random-walk Hastings-Metropolis;

    * `GenericGibbs`: base class for Gibbs samplers;

    * `PMMH`, `ParticleGibbs`: base classes for the PMCMC (particle MCMC
      algorithms) with the same name.

For instance, here is how to run 200 iterations of an adaptive random-walk
sampler::

    # ...
    # define some_static_model, some_prior
    # ...
    my_mcmc = BasicRWHM(model=some_static_model, prior=some_prior, niter=200,
                        adaptive=True)
    my_mcmc.run()


Upon completion, object ``my_mcmc`` have an attribute called ``chain``, which
is a `ThetaParticles` object (see module `smc_samplers`). In particular,
``my_mcmc.chain`` has the following attributes:

    * ``theta``: a structured array that contains the 200 simulated parameters;

    * ``lpost``: an array that contains the log-posterior density at these 200
      parameters.

See the dedicated notebook tutorial_ (on Bayesian inference for state-space
models) for more examples and explanations.

.. _tutorial: notebooks/Bayes_estimation_ssm.ipynb

Random walk Metropolis
======================

Both `GenericRWHM` and `PMMH` rely on a random walk proposal; that is, given
the current point theta, the proposed point is sampled from a Gaussian
distribution, with mean theta, and some covariance matrix Sigma (or
Sigma_t at iteration t the adaptive case, see below). Various parameters
may be set to tune this type of proposal:

* to run a standard random walk Metropolis algorithm, set ``adaptive=False``, and
  specify the covariance matrix Sigma through parameter ``rw_cov``.
  Otherwise, Sigma is set by default to the identity matrix (which could a
  terrible choice in certain problems).

* to run an adaptive random walk Metropolis algorithm, where the matrix Sigma_t
  is progressively adapted to the past states of the chain, set
  ``adaptive=True``. In that case, Sigma_t is set to a fraction of the running
  estimate of the covariance matrix of the *target* distribution, and ``rw_cov``
  is used as a preliminary estimate for that target covariance. See parameter
  ``scale`` (in the documentation of `GenericRWHM`) for more details on the
  factor in front of the running estimate, and `VanishCovTracker` for how the
  running estimate is computed recursively. 

By default, the adaptive version is used. 

Beyond the bootstrap filter within `PMMH`
========================================

`PMMH` runs, at each iteration, a particle filter to approximate the likelihood
of the considered state-space model. By default, a bootstrap filter is used
(with default choices for the resampling scheme, and so on). It is possible to
change the settings of this filtering algorithm, or run a different type of
algorithm, as follows:

1. You can set parameter ``fk_cls`` to a `FeynmanKac` subclass such as e.g. 
   `ssms.GuidedPF` if you wish to use a guided filter (rather than a bootstrap
   filter). See module `state_space_models` for more details on these
   `FeynmanKac` subclasses derived from a given state-space model.

2. You can use parameter ``smc_options`` (dict-like) to pass various parameters 
   to class `SMC` when the algorithm is instantiated.

3. You can even use parameter ``smc_cls`` to specify a different class for the
  algorithm itself (a `SMC` subclass instead of `SMC` itself).

4. Finally, if you need something even more general, you can also subclass
  `PMMH` and redefine method `alg_instance`, which takes as argument ``theta`` (a
  dict-like object) and returns an algorithm (an instance of class `SMC` or one
  its subclasses). 
"""


import numpy as np
from scipy import stats
from scipy.linalg import LinAlgError, cholesky

import particles
import particles.state_space_models as ssms
from particles import smc_samplers as ssp
from particles import utils


def msjd(theta):
    """Mean squared jumping distance.

    Parameters
    ----------
    theta: structured array

    Returns
    -------
    float
    """
    s = 0.0
    for p in theta.dtype.names:
        s += np.sum(np.diff(theta[p], axis=0) ** 2)
    return s


class MCMC:
    """MCMC base class.

    To subclass MCMC, define methods:
        * `step0(self)`: initial step
        * `step(self, n)`: n-th step, n>=1

    """

    def __init__(self, niter=10, verbose=0):
        """
        Parameters
        ----------
        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        """
        self.niter = niter
        self.verbose = verbose

    def step0(self):
        raise NotImplementedError

    def step(self, n):
        raise NotImplementedError

    def mean_sq_jump_dist(self, discard_frac=0.1):
        """Mean squared jumping distance estimated from chain.

        Parameters
        ----------
        discard_frac: float
            fraction of iterations to discard at the beginning (as a burn-in)

        Returns
        -------
        float
        """
        discard = int(self.niter * discard_frac)
        return msjd(self.chain.theta[discard:])

    def print_progress(self, n):
        params = self.chain.theta.dtype.fields.keys()
        msg = "Iteration %i" % n
        if hasattr(self, "nacc") and n > 0:
            msg += ", acc. rate=%.3f" % (self.nacc / n)
        for p in params:
            msg += f', {p}={self.chain.theta[p][n]}'
        print(msg)

    @utils.timer
    def run(self):
        for n in range(self.niter):
            if n == 0:
                self.step0()
            else:
                self.step(n)
            if self.verbose > 0 and (n * self.verbose) % self.niter == 0:
                self.print_progress(n)


##################################
# Random walk Metropolis samplers


class VanishCovTracker:
    r"""Tracks the vanishing mean and covariance of a sequence of points.

    Computes running mean and covariance of points
    t^(-alpha) * X_t
    for some alpha \in [0,1] (typically)
    """

    def __init__(self, alpha=0.6, dim=1, mu0=None, Sigma0=None):
        self.alpha = alpha
        self.t = 0
        self.mu = np.zeros(dim) if mu0 is None else mu0
        if Sigma0 is None:
            self.Sigma = np.eye(dim)
            self.L0 = np.eye(dim)
        else:
            self.Sigma = Sigma0
            self.L0 = cholesky(Sigma0, lower=True)
        self.L = self.L0.copy()

    def gamma(self):
        return (self.t + 1) ** (-self.alpha)  # not t, otherwise gamma(1)=1.

    def update(self, v):
        """Adds point v"""
        self.t += 1
        g = self.gamma()
        self.mu = (1.0 - g) * self.mu + g * v
        mv = v - self.mu
        self.Sigma = (1.0 - g) * self.Sigma + g * np.dot(
            mv[:, np.newaxis], mv[np.newaxis, :]
        )
        try:
            self.L = cholesky(self.Sigma, lower=True)
        except LinAlgError:
            self.L = self.L0


class GenericRWHM(MCMC):
    """Base class for random walk Hasting-Metropolis samplers.

    must be subclassed; the subclass must provide attribute self.prior
    """

    def __init__(
        self, niter=10, verbose=0, theta0=None, adaptive=True, scale=1.0, rw_cov=None
    ):
        """
        Parameters
        ----------
        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        theta0: structured array of size=1 or None
            starting point, simulated from the prior if set to None
        adaptive: True/False
            If true, random walk covariance matrix is adapted recursively
            based on past samples; see also scale and rw_cov for extra info.
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38^2 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array (defaults to Identity matrix if not provided)
            covariance matrix of the random walk proposal if adaptive=False;
            if adaptive=True, rw_cov is used as a preliminary guess for the
            covariance matrix of the target.
        """
        for k in ["niter", "verbose", "theta0", "adaptive"]:
            setattr(self, k, locals()[k])
        self.chain = ssp.ThetaParticles(
            theta=np.empty(shape=niter, dtype=self.prior.dtype),
            lpost=np.empty(shape=niter),
        )
        self.nacc = 0
        self.arr = ssp.view_2d_array(self.chain.theta)
        self.dim = self.arr.shape[-1]
        if self.adaptive:
            optim_scale = 2.38 / np.sqrt(self.dim)
            self.scale = scale * optim_scale
            self.cov_tracker = VanishCovTracker(dim=self.dim, Sigma0=rw_cov)
            self.L = self.scale * self.cov_tracker.L
        elif rw_cov is None:
            self.L = np.eye(self.dim)
        else:
            self.L = cholesky(rw_cov, lower=True)

    def step0(self):
        th0 = self.prior.rvs(size=1) if self.theta0 is None else self.theta0
        self.prop = ssp.ThetaParticles(theta=th0, lpost=np.zeros(1))
        self.prop_arr = ssp.view_2d_array(th0)
        self.compute_post()
        self.chain.copyto_at(0, self.prop, 0)

    def compute_post(self):
        """Computes posterior density at point self.prop"""
        raise NotImplementedError

    def step(self, n):
        z = stats.norm.rvs(size=self.dim)
        self.prop_arr[0] = self.arr[n - 1] + np.dot(self.L, z)
        self.compute_post()
        lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
        if np.log(stats.uniform.rvs()) < lp_acc:  # accept
            self.chain.copyto_at(n, self.prop, 0)
            self.nacc += 1
        else:  # reject
            self.chain.copyto_at(n, self.chain, n - 1)
        if self.adaptive:
            self.cov_tracker.update(self.arr[n])
            self.L = self.scale * self.cov_tracker.L

    @property
    def acc_rate(self):
        return self.nacc / (self.chain.N - 1)


class BasicRWHM(GenericRWHM):
    """Basic random walk Hastings-Metropolis sampler."""

    def __init__(
        self,
        niter=10,
        verbose=0,
        theta0=None,
        adaptive=True,
        scale=1.0,
        rw_cov=None,
        model=None,
    ):
        """
        Parameters
        ----------
        niter: int
            number of MCMC iterations
        verbose: int (default=0)
            progress report printed every (niter/verbose) iterations (never if 0)
        theta0: structured array of length=1 (default=None)
            starting point (if None, starting point is simulated from the
            prior)
        adaptive: bool
            whether the adaptive version should be used
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array
            covariance matrix of the random walk proposal (set to I_d if None)
        model: StaticModel object
            model that defines the target distribution

        For all the other parameters, see the docstring of the parent class,
        `GenericRWHM`.
        """
        if model is None:
            raise ValueError("Metropolis(MCMC): model not provided")
        else:
            self.model = model
        self.prior = model.prior
        GenericRWHM.__init__(
            self,
            niter=niter,
            verbose=verbose,
            theta0=theta0,
            adaptive=adaptive,
            scale=scale,
            rw_cov=rw_cov,
        )

    def compute_post(self):
        self.prop.lpost = self.model.logpost(self.prop.theta)


class PMMH(GenericRWHM):
    """Particle Marginal Metropolis Hastings.

    PMMH is class of Metropolis samplers where the intractable likelihood of
    the considered state-space model is replaced by an estimate obtained from
    a particle filter.
    """

    def __init__(
        self,
        niter=10,
        verbose=0,
        ssm_cls=None,
        smc_cls=particles.SMC,
        prior=None,
        data=None,
        smc_options=None,
        fk_cls=ssms.Bootstrap,
        Nx=100,
        theta0=None,
        adaptive=True,
        scale=1.0,
        rw_cov=None,
    ):
        """
        Parameters
        ----------
        niter: int
            number of iterations
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        ssm_cls: StateSpaceModel class
            the considered parametric class of state-space models
        smc_cls: class (default: particles.SMC)
            SMC class
        prior: StructDist
            the prior
        data: list-like
            the data
        smc_options: dict
            options to pass to class SMC
        fk_cls: (default=ssms.Bootstrap)
            FeynmanKac class associated to the model
        Nx: int
            number of particles (for the particle filter that evaluates the
            likelihood)
        theta0: structured array of length=1
            starting point (generated from prior if =None)
        adaptive: True/False
            If true, random walk covariance matrix is adapted recursively
            based on past samples; see also scale and rw_cov for extra info.
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38^2 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array (defaults to Identity matrix if not provided)
            covariance matrix of the random walk proposal if adaptive=False;
            if adaptive=True, rw_cov is used as a preliminary guess for the
            covariance matrix of the target.
        """
        self.ssm_cls = ssm_cls
        self.smc_cls = smc_cls
        self.fk_cls = fk_cls
        self.prior = prior
        self.data = data
        # do not collect summaries, no need
        self.smc_options = {"collect": "off"}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        self.Nx = Nx
        GenericRWHM.__init__(
            self,
            niter=niter,
            verbose=verbose,
            theta0=theta0,
            adaptive=adaptive,
            scale=scale,
            rw_cov=rw_cov,
        )

    def alg_instance(self, theta):
        return self.smc_cls(
            fk=self.fk_cls(ssm=self.ssm_cls(**theta), data=self.data),
            N=self.Nx,
            **self.smc_options
        )

    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.prop.lpost[0] += pf.logLt


class CSMC(particles.SMC):
    """Conditional SMC."""

    def __init__(self, fk=None, N=100, ESSrmin=0.5, xstar=None):
        particles.SMC.__init__(
            self,
            fk=fk,
            N=N,
            resampling="multinomial",
            ESSrmin=ESSrmin,
            store_history=True,
            collect="off",
        )
        self.xstar = xstar

    def generate_particles(self):
        particles.SMC.generate_particles(self)
        self.X[0] = self.xstar[0]

    def resample_move(self):
        particles.SMC.resample_move(self)
        self.X[0] = self.xstar[self.t]
        self.A[0] = 0


#####################################
# Gibbs samplers


class GenericGibbs(MCMC):
    """Generic Gibbs sampler for a state-space model.

    Updates sequentially X and theta; generic enough to allow for
    various strategies for these updates.

    Abstract class.

    """

    def __init__(
        self,
        niter=10,
        verbose=10,
        theta0=None,
        ssm_cls=None,
        prior=None,
        data=None,
        store_x=False,
    ):
        for k in ["ssm_cls", "prior", "data", "theta0", "niter", "store_x", "verbose"]:
            setattr(self, k, locals()[k])
        theta = np.empty(shape=niter, dtype=self.prior.dtype)
        if store_x:
            x = ssm_cls.state_container(niter, len(self.data))
            self.chain = ssp.ThetaParticles(theta=theta, x=x)
        else:
            self.chain = ssp.ThetaParticles(theta=theta)

    def update_states(self, theta, x):
        # x is None means we are at iteration 0, we must generate the states
        # from scratch
        raise NotImplementedError

    def update_theta(self, theta, x):
        raise NotImplementedError

    def step0(self):
        th0 = self.prior.rvs(size=1) if self.theta0 is None else self.theta0
        self.chain.theta[0] = th0
        self.x = self.update_states(self.chain.theta[0], None)
        if self.store_x:
            self.chain.x[0] = self.x

    def step(self, n):
        self.chain.theta[n] = self.update_theta(self.chain.theta[n - 1], self.x)
        self.x = self.update_states(self.chain.theta[n - 1], self.x)
        if self.store_x:
            self.chain.x[n] = self.x


class ParticleGibbs(GenericGibbs):
    """Particle Gibbs sampler (abstract class).

    Parameters
    ----------
    niter: int (default=10)
        number of MCMC iterations
    verbose: int (default=0)
        progress report printed every (niter/verbose) iterations (never if 0)
    ssm_cls: `StateSpaceModel` subclass
        parametric class of state-space models
    prior: `StructDist` object
        prior distribution
    data: list-like
        data
    theta0: structured array
        starting point of the chain (if None, generated from the prior)
    Nx: int
        number of x-particles (in the CSMC step)
    fk_cls: FeynmanKac class (default=None)
        which Feynman-Kac model to use (if None, set to ssms.Bootstrap, however,
        one may use instead e.g. ssm.GuidedPF)
    regenerate_data: bool (default=False)
        re-generate the data at each step; in the case the algorithm samples
        from the *prior*; useful to check if the algorithm is correct (i.e.
        if the update_theta method leaves invariant the conditional
        distributions of theta given x and y)
    backward_step: bool (default=False)
        whether to run the backward step
    store_x: bool (default=False)
        store the states at each iteration (if False only the theta's are
        stored)

    Note
    ----
    To subclass `ParticleGibbs`, define method `update_theta`, which samples
    theta given a state trajectory x.

    """

    def __init__(
        self,
        niter=10,
        verbose=0,
        ssm_cls=None,
        prior=None,
        data=None,
        theta0=None,
        Nx=100,
        fk_cls=None,
        regenerate_data=False,
        backward_step=False,
        store_x=False,
        backward_step_kwargs=None,
    ):
        GenericGibbs.__init__(
            self,
            niter=niter,
            verbose=verbose,
            ssm_cls=ssm_cls,
            prior=prior,
            data=data,
            theta0=theta0,
            store_x=store_x,
        )
        if backward_step_kwargs is None:
            backward_step_kwargs = {}
        self.Nx = Nx
        self.fk_cls = ssms.Bootstrap if fk_cls is None else fk_cls
        self.regenerate_data = regenerate_data
        self.backward_step = backward_step
        self._backward_step_kwargs = backward_step_kwargs if backward_step_kwargs is not None else {}

    def fk_mod(self, theta):
        ssm = self.ssm_cls(**ssp.rec_to_dict(theta))
        return self.fk_cls(ssm=ssm, data=self.data)

    def update_states(self, theta, x):
        fk = self.fk_mod(theta)
        if x is None:
            cpf = particles.SMC(fk=fk, N=self.Nx, store_history=True)
        else:
            cpf = CSMC(fk=fk, N=self.Nx, xstar=x)
        cpf.run()
        if isinstance(self.backward_step, str):
            if hasattr(cpf.hist, self.backward_step):
                method = getattr(cpf.hist, self.backward_step)
            else:
                method = getattr(cpf.hist, "backward_sampling_" + self.backward_step)
            new_x = method(1, **self._backward_step_kwargs)
        elif self.backward_step:  # need to check if it is exactly the True object
            new_x = cpf.hist.backward_sampling_ON2(1)
        else:
            new_x = cpf.hist.extract_one_trajectory()

        if self.regenerate_data:
            self.data = fk.ssm.simulate_given_x(new_x)
        return new_x
