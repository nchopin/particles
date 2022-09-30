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
       
    2. `TwistedFK` class that define a twisted Feynman-Kac models


The recommended import is::

    from particles import ControlledSMC module  as cSMC

For more details on ControlledSMC models and their properties, see the article: https://arxiv.org/abs/1708.08396

Steps to define a ControlledSMC model 
==============================
Step 1 : - Define your own model (for example a state space model [See basic tutorial lesson]). For example ssm = stovolModel() 
         - Define your policy functions. 
         
Step 2 :  Create the ControlledSMC object as follow:
 
     myCtrlSMC = cSMC.ControlledSMC(ssm=stovolModel, data = data, iterations = 5)
       ssm = your original defined model 
       data: is your data
       iterations = fixed at your convenience. 
       
Example: 
--------
Consider the following (simplified) stochastic volatility model:
.. math::

     Y_t|X_t=x_t         &\sim N(0, e^{x_t})                   \\
     X_t|X_{t-1}=x_{t-1} &\sim N(0, \rho x_{t-1})             \\
     X_0                 &\sim N(0, \sigma^2 / (1 - \rho^2))

To define this particular model, we sub-class from `ControlledSMC` as follows::

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

    stovolModel = SimplifiedStochVol(sigma=0.3, rho=0.9)

    myCtrlSMC = cSMC.ControlledSMC(ssm=stovolModel, data = data, iterations = 10)

Run the algorithm
=======================
To run the algorithm: 
    myCtrlSMC.runAll()
"""
from __future__ import division, print_function
import particles
from particles import state_space_models as ssm
import numpy as np
from particles import collectors
from particles import utils
from particles import distributions as dists
from sklearn.linear_model import Ridge

err_msg_missing_policy = " Model %s is missing method for policy ! Algorithm will use default policy "


class TwistedFK(ssm.Bootstrap):
    """Twisted SMC for a given state-space model.
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
    """

    def M0(self, N):  # ψ-Proposal0
        '''  Initial-Distribution  '''
        return self.M(0, self.ssm.PX0().rvs(size=N))
 
    def M(self, t, xp):
        '''  ψ-Distribution of X_t given X_{t-1}=xp   '''

        if hasattr(self.ssm, 'policy'):
            myPolicy = self.ssm.policy
            At, Bt, Ct = myPolicy[t] if type(
                myPolicy) is np.ndarray else self.ssm.policy(t)
        else:
            At, Bt, Ct = 0.0, 0.0, 0.0

        if t == 0:
            Mean = self.ssm.PX0().loc
            Var = self.ssm.PX0().scale**2
        else:
            Mean = self.ssm.PX(t, xp).loc
            Var = self.ssm.PX(t, xp).scale**2

        VarInv = 1.0 / Var
        V = VarInv * Mean - Bt
        Alpha = 1.0/(VarInv + 2*At)
        mbar = Alpha * V

        ProposalxPsi = dists.Normal(loc=mbar, scale=np.sqrt(Alpha))

        return ProposalxPsi.rvs(size = xp.shape[0])

    def logG(self, t, xp, x):  # Log Potentials
        if hasattr(self.ssm, 'policy'):
            myPolicy = self.ssm.policy  # return my log policy coefs
            At, Bt, Ct = myPolicy[t] if type(
                myPolicy) is np.ndarray else self.ssm.policy(t)
        else:
            At, Bt, Ct = 0.0, 0.0, 0.0

        # initialisation
        LogPolicy = np.zeros(x.shape[0])
        LogExpect = np.zeros(x.shape[0])
        LogForwardExpt = np.zeros(x.shape[0])

        for v in range(x.shape[0]):
            if self.du == 1:
                LogPolicy[v] = -self.Quadratic(At, Bt, Ct, x[v])
                if t != self.T-1:
                    LogForwardExpt[v] = self.logCondExp(t+1, x[v])
                if t == 0:
                    LogExpect[v] = self.logCondExp(t, x[v])
            else:
                LogPolicy[v] = - \
                    self.Quadratic(At, Bt.reshape(self.du, 1), Ct,  x[v])
                if t != self.T-1:
                    LogForwardExpt[v] = self.logCondExp(t+1, x[v])
                if t == 0:
                    LogExpect[v] = self.logCondExp(t, x[v])

        if t == 0:
            LogNuPsiOnPolicy = LogExpect - LogPolicy
            LogPotential = self.ssm.PY(t, xp, x).logpdf(self.data[t])
            LogForwardExp = LogForwardExpt
            LogGPsi = LogPotential + LogForwardExp + LogNuPsiOnPolicy
            return LogGPsi

        if t == self.T-1:
            LogPotential = self.ssm.PY(t, xp, x).logpdf(self.data[t])
            LogGPsi = LogPotential - LogPolicy
            return LogPotential - LogPolicy
        else:
            LogForwardExp = LogForwardExpt
            LogPotential = self.ssm.PY(t, xp, x).logpdf(self.data[t])
            LogGPsi = LogPotential + LogForwardExpt - LogPolicy
            return LogGPsi

    def logPolicy(self, t, xp, x, policy_t):
        At = policy_t[0]
        Bt = policy_t[1]
        Ct = policy_t[2]
        output = np.squeeze(- At * x ** 2 - Bt * x - Ct)
        return output

    def logCondExp(self, t, xp):
        """ Log Conditional expectation with respect to the Markov kernel at time t
        summary_ \E_M(ψ(Xp_t,X_t))

        Args:
            t (_type_): _description_
            xp (_type_): _description_

        Returns:
           \E_M(ψ(Xp_t,X_t))
        """
        dim = self.du
        if hasattr(self.ssm, 'policy'):
            myPolicy = self.ssm.policy  # return my log policy coefs
            A, B, C = myPolicy[t] if type(
                myPolicy) is np.ndarray else self.ssm.policy(t)
        else:
            A, B, C = 0.0, 0.0, 0.0

        if t == 0:
            Mean = self.ssm.PX0().loc
            Cov = self.ssm.PX0().scale**2
        else:
            Mean = self.ssm.PX(t, xp).loc
            Cov = self.ssm.PX(t, xp).scale**2
            Mean = Mean.reshape(self.du, 1)

        result = self.logCondFun(t, A, B, C, Mean, Cov)

        return result

    @property
    def isADP(self):
        """Returns true if we perform an ADP"""
        return 'ADP' in dir(self)

    @staticmethod
    def Quadratic(A, B, C, x):
        if type(x) is np.ndarray:
            return np.sum(x * np.dot(A, np.transpose(x))) + np.sum(B*np.transpose(x)) + C
        else:
            result = A*x**2 + B*x + C
            return result

    def logCondFun(self, t, A, B, C, Mean, Cov):
        """Log conditional expectation function"""
        dim = 1
        CovInv = 1.0/Cov
        V = CovInv * Mean - B
        Alpha = 1.0 / (CovInv + 2*A)
        quadraV = 0.5 * \
            self.Quadratic(Alpha, np.zeros([dim, 1]), 0, np.transpose(V))
        quadraGamaMean = - 0.5 * \
            self.Quadratic(CovInv, np.zeros([dim, 1]), 0, np.transpose(Mean))

        return quadraV + quadraGamaMean - 0.5 * np.log(1+2*Cov*A) - C


class ControlledSMC(TwistedFK):
    """ Controlled SMC class algorithm

    Parameters + Inputs.
    -------------------------------------------------------------
    It is the same as of the class TwistedFK  +  iterations (number of iterations)  

    ssm: StateSpaceModel object
        the considered state-space model (-ssm with proposal and logEta(the psi)),
    data: list-like
        the data

    Returns
    -------
        [type]: [description]
    FeynmanKac object
        the Feynman-Kac representation of the  filter for the
        considered state-space model
    """

    def __init__(self, ssm=None, data=None,  maxTime=None, iterations=None):
        self.ssm = ssm
        self.data = data
        self.maxTime = maxTime
        self.iterations = iterations
        self.du = self.ssm.PX0().dim
        self.policy = self.ssm.policy if hasattr(self.ssm, 'policy') else print(
            NotImplementedError(err_msg_missing_policy % self.__class__.__name__))
        self.iter = 0

    @property
    def T(self):
        return len(self.data) if self.maxTime is None else self.maxTime

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    @utils.timer
    def run(self):
        for _ in self:
            pass

    def RunAll(self):
        N = len(self.data)

        if hasattr(self.ssm, 'policy'):
            myPolicy = self.ssm.policy
            policy = np.array([myPolicy[t] if type(
                myPolicy) is np.ndarray else self.ssm.policy(t) for t in range(self.T)])
        else:
            myPolicy = np.array([0.0, 0.0, 0.0])
            policy = np.array([myPolicy for t in range(self.T)])

        setattr(self.ssm, 'policy', policy)

        # Construct and run the Psi Model for initialisation
        fk_model = TwistedFK(self.ssm, self.data)
        self.isADP == True
        fk_model.isADP == True

        PsiSMC = particles.SMC(fk=fk_model, N=N, resampling='multinomial',
                               collect=[collectors.Moments()], store_history=True)
        PsiSMC.run()
        settings = {'N': N, 'sample_trajectory': False,
                    'regularization': 0.009}

        schedule = np.exp(np.linspace(-10, 0, self.iterations))

        for it in range(self.iterations):
            # run ADP
            adp = self.ADP(fk_model, self.data, self.ssm.policy,
                           PsiSMC, settings, schedule[it])
            # Run ψ-twisted SMC with refined policy
            tfk_model = TwistedFK(self.ssm, self.data)
            PsiSMC = particles.SMC(fk=tfk_model, N=N, resampling='multinomial', collect=[
                                   collectors.Moments()], store_history=True)
            PsiSMC.run()

        return PsiSMC.hist, PsiSMC.summaries

    def ADP(self, model, observations, policy, psi_smc, settings, inverse_temperature=1.0):
        """
        Approximate dynamic programming to refine a policy.

        model = ssm or any kind of model 
        observations = data
        psi_smc = fk.run().results (derived from fk.run())

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

        policy : list of dicts  
            Coefficients specifying policy 

        inverse_temperature : float
            The inverse temperature controls the annealing of the observation densities

        Returns

        -------
        output : dict
            Algorithm output contain:
            'policy_refined' : list of dicts containing coefficients specifying refined policy 
            'r_squared' : numpy.array  containing coefficient of determination values    
        """

        T = model.T-1
        N = psi_smc.N
        SMC = psi_smc.hist
        policy_refined = np.array([[0.0, 0.0, 0.0] for t in range(self.T)])
        r_squared = np.ones([T+1])

        # initialize at T
        log_conditional_expectation = np.zeros([N])
        ancestors = SMC.A[T]
        states_previous = np.take(SMC.X[T-1], ancestors)
        states_current = SMC.X[T]
        SMC.A[0] = np.asarray(list(range(0, N)))

        for t in range(T, -1, -1):
            if t == 0:
                states_current = SMC.X[0]
                log_conditional_expectation = self.log_conditional_expectation(
                    t, policy_refined[1], states_current)
            # compute uncontrolled weights of reference proposal transition
            log_weights_uncontrolled = self.log_weights_uncontrolled(
                t, states_previous, states_current, inverse_temperature)
            # evaluate log-policy function  (self, t, policy, xp, x ):
            log_policy = self.logPolicy(
                t, states_previous, states_current, policy[t])

            # target function values
            target_values = log_weights_uncontrolled - \
                log_policy + log_conditional_expectation

            # perform regression to learn refinement (update this function for high dimensional case)
            (refinement, r_squared[t]) = self.learn_refinement(
                states_previous, states_current, target_values, settings)
            refinement[0] = np.abs(refinement[0])
            # refine current policy # This change automatically the ssm.policy cause reference.
            policy[t] = self.refine_policy(policy[t], refinement)

            if t > 1:
                ancestors = SMC.A[t-1]
                states_current = SMC.X[t-1]
                states_previous = np.take(
                    SMC.X[t-2], ancestors)  # SMC.X[T-1 ancestors]
                log_conditional_expectation = self.log_conditional_expectation(
                    t, policy[t], states_current)

        output = {'policy_refined': policy, 'r_squared': r_squared}

        return output

    """ 
    FONCTIONS USED FOR ADP FUNCTION ABOVE
    """

    def log_policy(self, t, policy, xp, x):
        return self.logPolicy(t, xp, x, policy)

    def log_weights_uncontrolled(self, t, xp, x, temp):
        return temp * self.ssm.PY(t, xp, x).logpdf(self.data[t])

    def log_conditional_expectation(self, t, policy_refined, x):
        it = np.repeat(t, x.shape[0])
        result = np.squeeze(np.array(list(map(self.logCondExp, it, x))))
        return result

    def learn_refinement(self, xp, x, target_values, settings):
        """
        Learn policy refinement using ridge regression.

        Parameters
        ----------                       
        xp : numpy.array (N, dim_s)
            Latent states at previous time period 
        x : numpy.array (N, dim_s)
            Latent states at current time period 
        target_values : numpy.array (N,)
            Target function values at latent states
        settings : dict
            Regression settings

        Returns
        -------    
        refinement : dict 
            Coefficients specifying the refinement at the current time period 
        r_squared : float
            Coefficient of determination
        """
        # construct design matrix
        if self.du == 1:
            x = x.reshape(x.shape[0], 1)
            xp = xp.reshape(xp.shape[0], 1)

        design_matrix = self.design_matrix_Quadratic_univariate(x)

        # perform ridge regression
        ridge_regressor = Ridge(
            alpha=settings['regularization'], fit_intercept=False)
        ridge_regressor.fit(design_matrix, - target_values)

        # get refinement coefficients from regression coefficients
        refinement = self.get_coef_Quadratic_univariate(ridge_regressor.coef_)

        # compute R-squared
        r_squared = np.corrcoef(ridge_regressor.predict(
            design_matrix), target_values)[0, 1]**2

        return (refinement, r_squared)

    def get_coef_Quadratic_univariate(self, regression_coef):
        """
        Get coefficients (a, b, c) of the Quadratic function of a univariate variable x 
        Q(x) = a * x^2 + b * x + c 
        given an array of regression coefficients.

        Parameters
        ----------    
        regression_coef : numpy.array (num_features,) where num_features = 3
            Array of regression coefficients

        Returns
        -------
        output : dict 
        """
        # get coefficients
        # output = {}
        output = np.zeros(3)

        output[0] = regression_coef[2]
        output[1] = regression_coef[1]
        output[2] = regression_coef[0]
        return output

    def design_matrix_Quadratic_univariate(self, x):
        """
        Construct design matrix of features for Quadratic function of a univariate variable
        Q(x) = a * x^2 + b * x + c.

        Parameters
        ----------    
        x : numpy.array (N, 1) 

        Returns
        -------
        design_matrix : numpy.array (N, num_features) where num_features = 3
        """
        # get size
        N = x.shape[0]

        # construct design matrix
        num_features = 3
        design_matrix = np.ones([N, num_features])  # for intercept c
        design_matrix[:, 1] = x[:, 0]  # for coefficient b
        design_matrix[:, 2] = x[:, 0]**2  # for coefficient a

        return design_matrix

    def refine_policy(self, policy_current, refinement):
        """
        Perform policy refinement.

        Parameters
        ----------  
        policy_current : dict
            Coefficients specifying the policy at the current time period 

        refinement : dict
            Coefficients specifying the refinement at the current time period 

        Returns
        -------
        output : dict
            Coefficients specifying the refined policy at the current time period         
        """

        outPut = refinement + policy_current
        return outPut