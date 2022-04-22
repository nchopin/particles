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

TODO: Run the algorithm
=======================
To run the algorithm: 
    myCtrlSMC.run()
    
Hopefully, the code above is fairly transparent.
 


.. note::
"""
from __future__ import division, print_function
import particles
from particles import state_space_models as ssm
import numpy as np
from particles import collectors
from particles import utils
from particles import distributions as dists
from sklearn.linear_model import Ridge

err_msg_missing_policy = """
    State-space model %s is missing method policy for controlled SMC, specify a policy
    """

""" 
TODO:
=====

 - Argument ssm must implement method: Policy function (Later on this will be taken out).
 - 
  
 QUESTIONS
 ---------
 -
 
 DISCUSION
 ---------
 - moove some functions to utils.
 - reshape the classes and simplify it
 - Make controlled SMC algo iterable
  
"""


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
    Argument ssm must implement methods a Policy function (Later on this will be taken out).
    """

    def M0(self, N):
        '''  Initial-Distribution  '''
        return self.M(0, self.ssm.PX0().rvs(size=N))

    def M(self, t, xp):
        '''  ψ-Distribution of X_t given X_{t-1}=xp   '''
        it = t*np.ones(xp.shape[0]).astype(int)
        if self.ssm.PX(t, xp).dim == 1:
            loop = [self.PsiProposal(it[i], xp[i]) for i in range(0, len(it))]
            transition = np.array(loop).reshape(xp.shape[0])
        else:
            loop = [self.PsiProposal(int(it[i]), xp[i, :])
                    for i in range(0, len(it))]
            transition = np.array(loop).reshape(xp.shape[0], xp.shape[1])

        return transition

    def PsiProposal(self, t, xp):  # M-ψ-Proposal

        myPolicy = self.ssm.policy
        At, Bt, Ct = myPolicy[t] if type(
            myPolicy) is np.ndarray else self.ssm.policy(t)

        dim = self.ssm.PX(t, xp).dim

        if t == 0:
            Mean = self.ssm.PX0().loc
            Var = self.ssm.PX0().cov if dim > 1 else self.ssm.PX0().scale**2
        else:
            Mean = self.ssm.PX(t, xp).loc
            Var = self.ssm.PX(t, xp).cov if dim > 1 else self.ssm.PX(
                t, xp).scale**2

        VarInv = np.linalg.inv(Var) if dim > 1 else 1.00 / Var
        V = np.dot(VarInv, Mean) - Bt
        Alpha = np.linalg.inv(
            VarInv + 2*At) if dim > 1 else 1.0/(VarInv + 2*At)
        mbar = np.dot(Alpha, V)

        if dim == 1:
            ProposalxPsiLaw = dists.Normal(loc=mbar, scale=np.sqrt(Alpha))
        else:
            ProposalxPsiLaw = dists.MvNormal(loc=mbar, scale=1, cov=Alpha)

        return ProposalxPsiLaw.rvs(size=1)

    def logG(self, t, xp, x):  # Log Potentials

        # retrieve policy from ssm model
        myPolicy = self.ssm.policy

        At, Bt, Ct = myPolicy[t] if type(
            myPolicy) is np.ndarray else self.ssm.policy(t)

        # initialisation
        LogPolicy = np.zeros(x.shape[0])
        LogExpect = np.zeros(x.shape[0])
        LogForwardExpt = np.zeros(x.shape[0])

        for v in range(x.shape[0]):
            if self.du == 1:
                LogPolicy[v] = -self.Quadratic(At, Bt, Ct, x[v])
                if t != self.T:
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
        LogPolicy = np.ones(x.shape[0])
        At = policy_t[0]
        Bt = policy_t[1]
        Ct = policy_t[2]
        for v in range(x.shape[0]):
            if self.du == 1:
                LogPolicy[v] = -self.Quadratic(At, Bt, Ct, x[v])
            else:
                LogPolicy[v] = - \
                    self.Quadratic(At, Bt.reshape(self.du, 1), Ct,  x[v])
        return LogPolicy

    def logCondExp(self, t, xp):
        # TODO: make the function  depends on policy !
        """ Log Conditional expectation with respect to the Markov kernel at time t
        summary_ \E_M(ψ(Xp_t,X_t))

        Args:
            t (_type_): _description_
            xp (_type_): _description_

        Returns:
           \E_M(ψ(Xp_t,X_t))
        """
        dim = self.du
        myPolicy = self.ssm.policy
        A , B , C  =   myPolicy[t-1] if type(myPolicy) is np.ndarray else  self.ssm.policy(t-1)

        if t == 0:
            Mean = self.ssm.PX0().loc
            Cov = self.ssm.PX0().cov if dim > 1 else self.ssm.PX0().scale**2
        else:
            Mean = self.ssm.PX(t, xp).loc
            Cov = self.ssm.PX(t, xp).cov if dim > 1 else self.ssm.PX(
                t, xp).scale**2

        result = self.logCondFun(t, A, B, C, Mean, Cov)

        return result

    @property
    def isADP(self):
        """Returns true if we perform an ADP"""
        return 'ADP' in dir(self)

    @staticmethod
    def Quadratic(A, B, c, x):
        if type(x) is np.ndarray:
            result = np.sum(x * np.dot(A, np.transpose(x))) + \
                np.sum(B*np.transpose(x)) + c
        else:
            result = A*x**2 + B*x + c
        return result

    def logCondFun(self, t, A, B, C, Mean, Cov):
        """Log conditional expectation function"""

        dim = Cov.shape[0] if type(Cov) is np.ndarray else 1
        Identity = np.identity(dim)
        CovInv = np.linalg.inv(Cov) if dim > 1 else 1.0/Cov
        V = np.dot(CovInv, Mean) - B
        Alpha = np.linalg.inv(
            CovInv + 2*A) if dim > 1 else 1.0 / (CovInv + 2*A)
        quadraV = 0.5 * self.Quadratic(Alpha, np.zeros([dim, 1]), 0, np.transpose(V))
        quadraGamaMean = - 0.5 * self.Quadratic(CovInv, np.zeros([dim, 1]), 0, np.transpose(Mean))

        Det = np.linalg.det(Identity + 2 * np.dot(Cov, A)) if dim > 1 else 1+2*Cov*A
        return quadraV + quadraGamaMean - 0.5 * np.log(Det) - C


class ControlledSMC(TwistedFK):

    """ Controlled SMC class algorithm

    Parameters + Inputs.
    -------------------------------------------------------------
    It is the same as of TwistedFK  +  iterations (number of iterations) to use for the controlled SMC

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

    def __init__(self, ssm=None, data=None, iterations=None):
        self.ssm = ssm
        self.data = data
        self.iterations = iterations
        self.du = self.ssm.PX0().dim
        self.policy = self.ssm.policy
        self.iter = 0

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    @property
    def isPolicyMissing(self):
        """Returns true if model parameter contains policy in the argument dictionary in ssm constructor"""
        if (hasattr(self, self.ssm.policy) == False):  # if('policy' in dir(self) == False):
            raise NotImplementedError(self._error_msg('missing policy'))

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    @utils.timer
    def run(self):  # make this iterator()
        for _ in self:
            pass

    def generateIntialParticules(self):
        N = len(self.data)
        policy_initial = np.array([[0.0, 0.0, 0.0] for t in range(self.T)])

        # Construct and run the Psi Model for initialisation to compute ADP to refine the policy
        fk_model = TwistedFK(self.ssm, self.data)
        PsiSMC = particles.SMC(fk=fk_model, N=N, resampling='multinomial',
                               collect=[collectors.Moments()], store_history=True)
        PsiSMC.run()

        # TODO: remove new field to the FK object.
        self.hist = PsiSMC.hist
        self.policy = policy_initial

    def generateParticulesWithADP(self):
        settings = {'N': len(self.data), 'sample_trajectory': False,
                    'regularization': 1e-4}
        # fk_model = self.hist
        PsiSMC = self.hist
        adp = self.ADP(self.data, self.policy, PsiSMC, settings)
        refinedPolicy = adp['policy_refined']
        self.ssm.set_policy(refinedPolicy)
        self.policy = refinedPolicy

        # Run ψ -twisted SMC with refined policy
        fk_model = TwistedFK(self.ssm, self.data)
        fk_model.isADP == True
        PsiSMC = particles.SMC(fk=fk_model, N=len(self.data), resampling='multinomial',
                               collect=[collectors.Moments()], store_history=True)
        PsiSMC.run()

        self.hist = PsiSMC.hist

    def RunAll(self):  # def __next__(self):
        # if self.done(self):
        #     raise StopIteration
        # if self.iterations == 1:
        # intialisation
        N = len(self.data)
        myPolicy = self.ssm.policy
        policy = np.array([myPolicy[t] if type(myPolicy) is np.ndarray else self.ssm.policy(t) for t in range(self.T)])  # this is the right one.

        # Construct and run the Psi Model for initialisation
        fk_model = TwistedFK(self.ssm, self.data)
        PsiSMC = particles.SMC(fk=fk_model, N=N, resampling='multinomial',
                               collect=[collectors.Moments()], store_history=True)
        PsiSMC.run()
        settings = {'N': N, 'sample_trajectory': False,
                    'regularization': 1e-4}
        # else:
        for it in range(self.iterations):
            # run ADP
            adp = self.ADP(fk_model, self.data, policy, PsiSMC, settings)
            # Construct refined policy
            refinedPolicy = adp['policy_refined']
            self.ssm.set_policy(refinedPolicy)
            policy = refinedPolicy
            TestRefinedPolicy = np.array(self.ssm.policy)
            # Run ψ-twisted SMC with refined policy
            fk_model = TwistedFK(self.ssm, self.data)
            fk_model.isADP == True
            PsiSMC = particles.SMC(fk=fk_model, N=N, resampling='multinomial', collect=[
                                   collectors.Moments()], store_history=True)
            PsiSMC.run()
        return PsiSMC.hist

    def ADP(self, model, observations, policy, psi_smc, settings, inverse_temperature=1.0):
        """
        model = ssm or any kind of model 
        observations = data
        psi_smc = fk.run().results (derived from fk.run())
        settings = parameters of the model you define yourself

        Approximate dynamic programming to refine a policy.

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
        T = len(observations) - 1  # observations.shape[0] - 1
        N = settings['N']

        HistoryData = psi_smc.hist

        # pre-allocate
        policy_refined = np.array([[0.0, 0.0, 0.0] for t in range(self.T)])

        r_squared = np.ones([T+1])

        # initialize at T
        states_previous = psi_smc.Xp
        states_current = psi_smc.X
        log_conditional_expectation = np.zeros([N])

        # iterate over time periods backwards
        for t in range(T, 0, -1):
            states_previous = HistoryData.X[t-1]
            states_current = HistoryData.X[t]

            # compute uncontrolled weights of reference proposal transition
            # (t, observations[t, :], states_previous, states_current)
            log_weights_uncontrolled = self.log_weights_uncontrolled(
                t, states_previous, states_current)

            # evaluate log-policy function
            log_policy = self.log_policy(
                t, policy[t], states_previous, states_current)

            # target function values
            target_values = log_weights_uncontrolled.reshape(len(log_policy), 1) + \
                log_conditional_expectation.reshape(len(log_policy), 1) - log_policy.reshape(
                    len(log_policy), 1)

            # perform regression to learn refinement (update this function for high dimensional case)
            (refinement, r_squared[t]) = self.learn_refinement(
                states_previous, states_current, target_values, settings)

            # refine current policy
            policy_refined[t] = self.refine_policy(policy[t], refinement)

            # set Policy
            self.ssm.set_policy(policy_refined)

            # compute log-conditional expectation of refined policy
            if t != 1:
                states_previous = HistoryData.X[t-1]
                states_current = HistoryData.X[t]
                log_conditional_expectation = self.log_conditional_expectation(
                    t, policy_refined[t], states_current)

        output = {'policy_refined': policy_refined}

        return output

    """ 
    FONCTIONS USED FOR ADP FUNCTION ABOVE
    """

    def log_weights_uncontrolled(self, t, xp, x):
        """ """
        return self.ssm.PY(t, xp, x).logpdf(self.data[t])

    def log_policy(self, t, policy, xp, x):
        """ """
        LogPolicy = self.logPolicy(t, xp, x, policy)
        return LogPolicy

    def log_conditional_expectation(self, t, policy_refined, x):
        """ """
        LogCondExpect = np.ones(x.shape[0])

        it = t*np.ones(x.shape[0]).astype(int)

        if self.ssm.PX(t, x).dim == 1:
            loop = [self.logCondExp(it[i], x[i]) for i in range(0, len(it))]
            LogCondExpect = np.array(loop).reshape(x.shape[0])
        else:
            loop = [self.logCondExp(it[i], x[i, :]) for i in range(0, len(it))]

            LogCondExpect = np.array(loop).reshape(x.shape[0], 1)

        return LogCondExpect

    def learn_refinement(self, xp, x, target_values, settings):  # ridge_regressor   here
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
        refinement = ridge_regressor.coef_

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
        output = {}

        output['a'] = regression_coef[2] 
        output['b'] = regression_coef[1]  
        output['c'] = regression_coef[0]  
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

        if self.du == 1:
            outPut =  policy_current + np.exp(-refinement)
        else: # update this 
            outPut = policy_current + refinement
        return outPut
