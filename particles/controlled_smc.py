# -*- coding: utf-8 -*-

"""
Controlled Sequential Monte Carlo models as Python objects.

The policy function is an attribute of StateSpaceModel class.


******************************************************************************
Overview
========
This module defines:

    1. the `ControlledSMC` class, which lets you define a controlled sequential Monte Carlo model
       as a Python object;

    3. `TwistedSMC` class that define a kind of Bootstrap Feynman-Kac models

    2. `FeynmanKac` classes based on the previous classes

The recommended import is::

    from particles import ControlledSMC module  as CtSMC

For more details on ControlledSMC models and their properties, see the article: https://arxiv.org/abs/1708.08396

TODO: Defining a ControlledSMC model
==============================

Consider the following (simplified) stochastic volatility model:

.. math::

     Y_t|X_t=x_t         &\sim N(0, e^{x_t})                   \\
     X_t|X_{t-1}=x_{t-1} &\sim N(0, \rho x_{t-1})             \\
     X_0                 &\sim N(0, \sigma^2 / (1 - \rho^2))

To define this particular model, we sub-class `StateSpaceModel` as follows::

    import numpy as np
    from particles import distributions as dists

    class SimplifiedStochVol(ssms.StateSpaceModel):
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
      distributions are defined there;  see module `distributions` for more details.
    * The class above actually defines a **parametric** class of models; in
      particular,  ``self.sigma`` and ``self.rho`` are **attributes** of
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

TODO: Associated Feynman-Kac models
=============================

Now that our state-space model is defined, we obtain the associated Bootstrap
Feynman-Kac model as follows:

    my_fk_model = ssms.Bootstrap(ssm=my_stoch_vol_model, data=y)

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
    my_better_fk_model = ssms.Guided(ssm=my_second_ssm, data=y)
    # then run a SMC as above

Voilà! You have now implemented a guided filter.

Of course, the proposal distribution above does not make much sense; we use it
to illustrate how proposals may be defined. Note in particular that it depends
on ``data``, an object that represents the complete dataset. Hence the proposal
kernel at time ``t`` may depend on y_t but also y_{t-1}, or any other
datapoint.

For auxiliary particle filters (APF), one must in addition specify auxiliary
functions, that is the (log of) functions :math:`\eta_t` that modify the
resampling probabilities (see Section 10.3.3 in the book)::

    class StochVol_with_prop_and_aux_func(StochVol_with_prop):
        def logeta(self, t, x, data):
            "Log of auxiliary function eta_t at time t"
            return -(x-data[t])**2

    my_third_ssm = StochVol_with_prop_and_aux_func()
    apf_fk_model = ssms.AuxiliaryPF(ssm=my_third_ssm, data=y)

Again, this particular choice does not make much sense, and is just given to
show how to define an auxiliary function.

Already implemented the module
======================================

This module implements a few basic state-space models that are often used as
numerical examples:

===================       =====================================================
Class                     Comments
===================       =====================================================
`NeuroScience`
`StochasticVol`

===================       =====================================================

.. note::
    In C-SMC, proposal and weights are changed by the twisted functions.
    The policy function Psi (in Twisted SMC) is   an attribute of StateSpaceModel.
    The user also need to define the proposal function in this class as well.
    This proposal shoub be overidden dynamically with ADP !

"""

from __future__ import division, print_function
import state_space_models as modelssm
import numpy as np
from   collectors import Moments
import utils
import kalman as kalman
import particles
from particles import distributions as dists

err_msg_missing_cst = """
    State-space model %s is missing method upper_bound_log_pt, which provides
    log of constant C_t, such that
    p(x_t|x_{t-1}) <= C_t
    This is required for smoothing algorithms based on rejection
    """
err_msg_missing_policy = """
    State-space model %s is missing method policy for controlled SMC, specify a policy
    """

"""
TODO:
=====
 - code is failing in resampling module in the function wmean_and_var() or 2D.
                      m = np.average(x, weights=W, axis=0)
                      m2 = np.average(x**2, weights=W, axis=0) # x**2
                      v = m2 - m**2
                      return {'mean': m, 'var': v}

 - Policy input shoud be refined 
 - Management of Policy function interaction.
 - Reshape def run(self) in ControlledSMC class.

 QUESTIONS
 ---------
 -Observation space need to be updated !
 
"""

# Define the ψ-twisted model || ψ-observation space, ψ-Proposal0, ψ-Proposal, G-ψ ?
class TwistedSMC(modelssm.Bootstrap):
    """Twisted SMC for a given state-space model.
    TODO:
     matmul is not working for 1D. Adjust this.
     define a template class to accomodate 1D and nD.

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
    Argument ssm must implement methods `proposal0` and `proposal` and define a Policy function.
    """

    def M0(self, N):  # TODO:   t = 0 is not the right thing to do.
        return self.M(0, self.ssm.proposal0(self.data).rvs(size=N))

    def M(self, t, xp):
        At, Bt, Ct = self.ssm.policy()
        Mean = self.ssm.proposal(t, xp, self.data).loc
        dimension = self.ssm.proposal(t, xp, self.data).dim

        if dimension == 1:
            Var = self.ssm.proposal(t, xp, self.data).scale
            VarInv = 1.00 / Var
            V = np.dot(VarInv, Mean) - Bt
            Alpha = VarInv + 2*At
            AlphaInv = 1.00 / Alpha
        else:
            Var = self.ssm.proposal(t, xp, self.data).cov
            VarInv = np.linalg.inv(Var)
            V = np.dot(VarInv, Mean) - Bt
            Alpha = VarInv + 2*At
            AlphaInv = np.linalg.inv(Alpha)

        mbar = np.dot(V, np.dot(VarInv, Mean) - Bt)

        expo = 0.5 * self.Quadratic(self, AlphaInv,
                                    np.zeros((dimension, 1)), - Ct, mbar)
        if dimension == 1:
            sqrtDet = np.sqrt(np.abs(Alpha)/np.abs(Var))
        else:
            sqrtDet = np.sqrt(np.linalg.det(Alpha)/np.linalg.det(Var))

        normalisation = self.Expectation(t, xp)

        if dimension == 1:
            ProposalxPsi = dists.Normal(
                mbar, (sqrtDet*expo/normalisation)**2*Alpha)
        else:
            ProposalxPsi = dists.MvNormal(
                mbar, sqrtDet*expo/normalisation, Alpha)
        # Proposal =  self.ssm.proposal(t, xp, self.data).rvs(size=xp.shape[0])
        # psiFactor = 0.5 * self.Quadratic(At, Bt, Ct, x)
        # return Proposal*psiFactor/self.Expectation(t, xp)
        # return ProposalxPsi / normalisation
        return ProposalxPsi

    def logG(self, t, xp, x):
        At, Bt, Ct = self.ssm.policy()
        du = self.ssm.PX0().dim

        # Dimension adjustement
        # Is breaking but something is happing in the code core.  Weights are appended !
        if self.du == 1:
            LogPolicy = self.Quadratic(self, At, Bt, Ct, x)  # TODO:   unsupported operand type(s) for ** or pow(): 'Normal' and 'int'
        else:
            LogPolicy = self.Quadratic(self, At, Bt.reshape(
                self.du, 1), Ct.reshape(1, 1),  x[-1])  # TODO:  x[1]

        LogNormalisation = np.log(self.Expectation(t, x[-1]))

        #  TODO:
        if t == 0:
            return (self.ssm.PX0().logpdf(x)
                    + self.ssm.proposal0(self.data).logpdf(x) + LogNormalisation - LogPolicy)
        if t == self.T:
            LogPotential = self.ssm.PY(t, xp, x).logpdf(
                self.data[t])
            return LogPotential - LogPolicy
        else:
            LogPotential = self.ssm.PY(t, xp, x).logpdf(
                self.data[t])
            return LogPotential + LogNormalisation - LogPolicy

    @staticmethod # TODO: unsupported operand type(s) for ** or pow(): 'Normal' and 'int'
    def Quadratic(self, A, B, c, x):
        if self.ssm.PX0().dim == 1:
            return A*x**2 + B*x + c
        else:
            return np.sum(x * np.dot(A, np.transpose(x))) + np.sum(B*np.transpose(x)) + c


    def Expectation(self, t, xp): # \E[ψ(t, x, xp)| xp] = \E {\exp[(Ax,x)+ Bx + C] | xp}
        """Conditional expectation with respect to the Markov kernel at time t

        Args:
            t (_type_): _description_
            xp (_type_): _description_

        Returns:
            _type_: _description_
        """
        At, Bt, Ct = self.ssm.policy()  # policy depends on (self, t, xp, x))
        dimension =  self.du

        # TODO:
        if t == 0:
            return xp  # This should be adjusted TODO:
        if t == self.T:  # G_T/Psi_T
            Mean = self.ssm.proposal(t, xp, self.data[t]).loc
            Var = self.ssm.proposal(t, xp, self.data[t]).cov  
            VarInverse = np.linalg.inv(Var)
            V = np.dot(VarInverse, Mean) - Bt
            Alpha = VarInverse + 2*At
            Identity = np.identity(dim)
            sqrtDet = np.sqrt(np.linalg.det(Identity + 2 * np.dot(Var, At)))
            expo = self.Quadratic(Alpha, np.zeros((dimension, 1)), - Ct, V)

            return np.exp(expo) / np.sqrt(sqrtDet)
        else:
            Mean = self.ssm.proposal(t, xp, self.data).loc 
            if dimension == 1:
                Var = self.ssm.proposal(t, xp, self.data).scale 
            else:
                Var = self.ssm.proposal(t, xp, self.data).cov  

            if dimension == 1:
                VarInverse = 1.00/Var
                V = VarInverse*Mean - Bt  # ! Mean can be a random variable
                Identity = dimension
                sqrtDet = np.sqrt(np.abs(Identity + 2 * Var * At))
                Alpha = VarInverse + 2*At
                expo = self.Quadratic(self, Alpha, np.zeros((dimension, 1)), - Ct, np.transpose(V))
            else:
                VarInverse = np.linalg.inv(Var)
                V = np.dot(VarInverse, np.transpose(Mean)) - \
                    (np.transpose(Bt)).reshape(dimension, 1)
                Identity = np.identity(dimension)
                sqrtDet = np.sqrt(np.linalg.det(
                    Identity + 2 * np.dot(Var, At)))
                Alpha = VarInverse + 2*At

                expo = self.Quadratic(self, Alpha, np.zeros(
                    (dimension, 1)), - Ct.reshape(1, 1), np.transpose(V))

            return np.exp(expo) / sqrtDet


class ControlledSMC(TwistedSMC):

    """ Controlled SMC algorithm
    Proposal distributions are determined by approximating the solution to an associated
    optimal control problem using an iterative scheme = > You use APF where the proposal
    is updated at every iteration.
    Parameters + Inputs
    -------------------
    ssm: StateSpaceModel object
        the considered state-space model  (-ssm with proposal and logEta(the psi)),
    data: list-like
        the data


    Returns
    -------
        [type]: [description]
    FeynmanKac object
        the Feynman-Kac representation of the  filter for the
        considered state-space model

    Note
    ----
    In C-SMC, proposal and weights are changed by the twisted functions.
    """

    def __init__(self, ssm=None, data=None):
        self.ssm = ssm  # Argument ssm must implement methods `proposal0`, `proposal`
        self.data = data
        self.du = self.ssm.PX0().dim
        self.policy = self.ssm.policy

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    @property
    def isPolicyMissing(self):
        """Returns true if model parameter contains policy in the argument dictionary in ssm constructor"""
        if (hasattr(self,self.ssm.policy) == False):
          raise NotImplementedError(self._error_msg('missing policy'))


    def run(self):  # make this iterator()
        # Policy Initialisation
        AO, BO, CO = self.ssm.policy()

        # TODO: Dynamic SMC with update proposal via policy modulo coeffs A, B, C of Policy
        for t in range(self.T):
            # Run  Twisted SMC  for t different of T
            fk_model = modelssm.TwistedSMC(self.ssm, self.data)
            PsiSMC = particles.SMC(fk=fk_model, N=100, resampling='stratified',
                              collect=[Moments()], store_history=True)
            PsiSMC.run()
            
            # TODO: Look at the code and pick the right params
            settings = {'N': 100, 'sample_trajectory': False, 'regularization': 1e-4}

            # run ADP to refine previous policy
            # add feed the right staffs from PsiSMC.run()
            adp = self.RefinePsiEstimation(
                fk_model, self.data, self.ssm.policy, PsiSMC.summaries, settings, inverse_temperature=0.0) # add feed the right staffs from PsiSMC.run()
            """
                model = ssm or any kind of model
                observations = data
                psi_smc = fk.run().results (derived from fk.run())
                settings = parameters of the model you define yourself
            """
            # Construct refined policy (It is why we get/set policy), Use set function
            refinedPolicy = self.ssm.policy() * adp['policy_refined'] # update A, B, C normally !
            self.ssm.set_policy(self, refinedPolicy)

            # Run ψ -twisted SMC method for t = T,
            if t == self.T:
                fk_model = modelssm.TwistedSMC(self.ssm, self.data)
                PsiSMC = particles.SMC(fk=fk_model, N=100, resampling='stratified',
                                  collect=[Moments()], store_history=True)
                PsiSMC.run()

            # return PsiSMC.result # Seems that PsiSMC is void :) as desire ! Smoothing, Filtering etc..
        pass

    # python compatibility
    def next(self):
        return self.__next__()  #  Python 2 compatibility

    def __iter__(self):
        return self

    # Compute the Backward Filter
    def RefinePsiEstimation(model, observations, policy, psi_smc, settings, inverse_temperature=1.0):
        """
        model = ssm or any kind of model
        observations = data
        psi_smc = fk.run().results (derived from fk.run())
        settings = parameters of the model you define yourself

        Approximate dynamic programming to refine a policy.

        In Python method overriding occurs by simply defining in the child class  a method with the same name of a method
        in the parent class. When you  define a method in the object you make this latter able to satisfy that  method call,
        so the implementations of its ancestors do not come in  play.

        Parameters
        ----------
        model : controlledpsi_smc.models.LRR.ssm
            A ssm class instance

        observations : numpy.array (T+1, dim_y)
            Time series

        settings : dict
            Particle filtering settings contain:
                'N' : int specifying number of particles
                'sample_trajectory' : bool specifying whether a trajectory is to be sampled

        policy : list of dicts of length T+1
            Coefficients specifying policy

        inverse_temperature : float
            The inverse temperature controls the annealing of the observation densities

        Returns

        -------
        output : dict
            Algorithm output contain:
            'policy_refined' : list of dicts of length T+1 containing coefficients specifying refined policy
            'r_squared' : numpy.array (T+1,) containing coefficient of determination values
        """

        # get model properties and algorithmic settings
        dim_s = model.dim_s
        T = observations.shape[0] - 1
        N = settings['N']

        # pre-allocate
        policy_refined = [{} for t in range(T+1)]
        r_squared = np.ones([T+1])

        # initialize
        log_conditional_expectation = np.zeros([N])
        ancestors = psi_smc['ancestry'][:, T-1]
        states_previous = psi_smc['states'][ancestors, :, T-1]
        states_current = psi_smc['states'][:, :, T]

        # iterate over time periods backwards
        for t in range(T, 0, -1):
            # compute uncontrolled weights of reference proposal transition
            log_weights_uncontrolled = model.log_weights_uncontrolled(
                t, observations[t, :], states_previous, states_current, inverse_temperature)

            # evaluate log-policy function
            log_policy = model.log_policy(
                policy[t], states_previous, states_current)

            # target function values
            target_values = log_weights_uncontrolled + \
                log_conditional_expectation - log_policy

            # perform regression to learn refinement
            (refinement, r_squared[t]) = model.learn_refinement(
                states_previous, states_current, target_values, settings)

            # refine current policy
            policy_refined[t] = model.refine_policy(policy[t], refinement)

            # compute log-conditional expectation of refined policy
            if t != 1:
                ancestors = psi_smc['ancestry'][:, t-2]
                states_previous = psi_smc['states'][ancestors, :, t-2]
                states_current = psi_smc['states'][:, :, t-1]
                (log_conditional_expectation, _) = model.log_conditional_expectation(
                    policy_refined[t], states_current)

        # also refine initial policy for random initial distributions
        if model.initial_type == 'random':
            # compute log-conditional expectation of refined policy
            states_current = psi_smc['states'][:, :, 0]
            (log_conditional_expectation, _) = model.log_conditional_expectation(
                policy_refined[1], states_current)

            # compute uncontrolled weights of reference proposal distribution
            log_weights_uncontrolled = model.log_weights_uncontrolled_initial(
                observations[0, :], states_current, inverse_temperature)

            # evaluate log-policy function
            log_policy = model.log_initial_policy(policy[0], states_current)

            # target function values
            target_values = log_weights_uncontrolled + \
                log_conditional_expectation - log_policy

            # perform regression to learn refinement
            (refinement, r_squared[0]) = model.learn_initial_refinement(
                states_current, target_values, settings)

            # refine current policy
            policy_refined[0] = model.refine_initial_policy(
                policy[0], refinement)

        # algorithm output
        output = {'policy_refined': policy_refined, 'r_squared': r_squared}

        return output
