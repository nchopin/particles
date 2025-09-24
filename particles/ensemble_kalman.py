
import collections

import numpy as np
from scipy.linalg import solve

from particles import distributions as dists
from particles import state_space_models as ssms
from particles import utils
import particles




def EnK_step(ssm, t, xp, x_prop, y, return_weights = False):
  if x_prop.ndim == 1:
    dx = 1
    J = len(x_prop)
    mapped_X_prop = ssm.PY(t, xp, x_prop).rvs(size=x_prop.shape[0])
    x_prop = np.reshape(x_prop, (J,1))
  else:
    J, dx = x_prop.shape    
    mapped_X_prop = ssm.PY(t, xp, x_prop).rvs(size=x_prop.shape[0])
  # weights = ssm.PY(t, xp, x_prop).logpdf(mapped_X_prop)
  mapped_X_prop = np.reshape(mapped_X_prop, (J,1))
  full_cov = np.cov(x_prop,mapped_X_prop, rowvar=False)
  Cuu = np.atleast_2d(full_cov[0:dx, 0:dx])
  Cup = np.atleast_2d(full_cov[0:dx, dx:])
  CppGamma = np.atleast_2d(full_cov[dx:, dx:])
  # Cuu = np.atleast_2d(full_cov[0:dx, 0:dx])
  # Cuu = np.atleast_2d(np.cov(x_prop,mapped_X_prop, rowvar=False)[0:dx,0:dx])
  # Cup = np.atleast_2d(np.cov(x_prop,mapped_X_prop, rowvar=False)[0:dx,dx:])
  # CppGamma = np.atleast_2d(np.cov(mapped_X_prop, rowvar=False))
  K = Cup@np.linalg.inv(CppGamma)
  cov_shrinkage = (np.eye(dx) - K@Cup.T)
  Qhat = cov_shrinkage@Cuu@cov_shrinkage.T
  new_filt = x_prop - (K@(mapped_X_prop.T - y.T)).T
  # K = Cup@(np.linalg.inv(CppGamma))
  
  # noise_model = ssm.PY(t, xp, x_prop)
  # if isinstance(noise_model, dists.Normal):
  #   Gamma = np.atleast_2d(noise_model.scale**2)
  # elif  isinstance(noise_model, dists.MvNormal):
  #   Gamma = noise_model.scale**2 * noise_model.cov # this will only work for multivariate normal noise terms
  # else:
  #   raise TypeError("noise model is assumed to be Gaussian for EnKF to work. Feel free to extend this.")
  if return_weights: # possibly remove all this
    if t == 0:        
      Pxweights = ssm.PX0().logpdf(x_prop)
      Qxweights = ssm.PX0().logpdf(new_filt)  
    else:        
      Pxweights = ssm.PX(t,xp).logpdf(x_prop)
      Qxweights = ssm.PX(t,xp).logpdf(new_filt)
    correction_logweights = Pxweights - Qxweights
    return new_filt, K, correction_logweights
  else:
    return new_filt, K, None



def EnK_step_for_WEnKF(ssm, t, xp, x_prop, y, return_weights = False):
  assert isinstance(ssm, MVNonlinearGauss)
  if x_prop.ndim == 1:
    dx = 1
    J = len(x_prop)
    mapped_X_prop = ssm.F(x_prop)# ssm.PY(t, xp, x_prop).rvs(size=x_prop.shape[0])
    x_prop = np.reshape(x_prop, (J,1))
  else:
    J, dx = x_prop.shape    
    mapped_X_prop = ssm.F(x_prop)#ssm.PY(t, xp, x_prop).rvs(size=x_prop.shape[0])
  # weights = ssm.PY(t, xp, x_prop).logpdf(mapped_X_prop)
  mapped_X_prop = np.reshape(mapped_X_prop, (J,1))
  full_cov = np.cov(x_prop,mapped_X_prop, rowvar=False)
  Cuu = np.atleast_2d(full_cov[0:dx, 0:dx])
  Cup = np.atleast_2d(full_cov[0:dx, dx:])
  CppGamma = np.atleast_2d(full_cov[dx:, dx:]) + ssm.covY
  # Cuu = np.atleast_2d(full_cov[0:dx, 0:dx])
  # Cuu = np.atleast_2d(np.cov(x_prop,mapped_X_prop, rowvar=False)[0:dx,0:dx])
  # Cup = np.atleast_2d(np.cov(x_prop,mapped_X_prop, rowvar=False)[0:dx,dx:])
  # CppGamma = np.atleast_2d(np.cov(mapped_X_prop, rowvar=False))
  K = Cup@np.linalg.inv(CppGamma)
  cov_shrinkage = (np.eye(dx) - K@Cup.T)
  Qhat = cov_shrinkage@Cuu@cov_shrinkage.T
  Qhatinv = np.linalg.inv(Qhat)
  Qinv = np.linalg.inv(ssm.covX)
  betas = np.random.multivariate_normal(np.zeros(dx), Qhat, size=J)
  new_filt = x_prop - (K@(mapped_X_prop.T - y.T)).T + betas
  corr_weight1 = 0.5*np.einsum('ij,ji->i', betas, np.dot(Qhatinv, betas.T))
  diff = new_filt - x_prop
  corr_weight2 = 0.5*np.einsum('ij,ji->i', diff, np.dot(Qinv, diff.T))
  return new_filt, (corr_weight1 - corr_weight2)

error_msg = "arguments of MVNonlinearGauss.__init__ have inconsistent shapes"

class MVNonlinearGauss(ssms.StateSpaceModel):
    r"""Multivariate nonlinear additive Gaussian model.

    .. math::
        X_0 & \sim N(\mu_0, cov_0) \\
        X_t & = F(X_{t-1}) + U_t, \quad   U_t\sim N(0, cov_X) \\
        Y_t & = G(X_t) + V_t,     \quad   V_t \sim N(0, cov_Y)

    The only mandatory parameters are `covX` and `covY` (from which the
    dimensions dx and dy of, respectively, X_t, and Y_t, are deduced). The
    default values for the other parameters are:

    * `mu0` : an array of zeros (of size dx)
    * `cov0`: cov_X
    * `F` : Identity mapping  of shape (dx, dx)
    * `G` : (dy, dx) matrix such that G[i, j] = 1[i=j]

    Note
    ----
    The Kalman filter takes as an input an instance of this class (or one of
    its subclasses).
    """

    def __init__(self, F=None, G=None, covX=None, covY=None, mu0=None, cov0=None):
        self.covX, self.covY = np.atleast_2d(covX), np.atleast_2d(covY)
        self.dx, self.dy = self.covX.shape[0], self.covY.shape[0]
        self.mu0 = np.zeros(self.dx) if mu0 is None else mu0
        self.cov0 = self.covX if cov0 is None else np.atleast_2d(cov0)
        self.F = (lambda x: x) if F is None else F
        self.G = (lambda x: x[0:self.dy]) if G is None else G
        self.check_shapes()

    def check_shapes(self):
        """
        Check all dimensions are correct.
        """
        assert self.covX.shape == (self.dx, self.dx), error_msg
        assert self.covY.shape == (self.dy, self.dy), error_msg
        # assert self.F.shape == (self.dx, self.dx), error_msg
        # assert self.G.shape == (self.dy, self.dx), error_msg
        assert self.mu0.shape == (self.dx,), error_msg
        assert self.cov0.shape == (self.dx, self.dx), error_msg

    def PX0(self):
        return dists.MvNormal(loc=self.mu0, cov=self.cov0)

    def PX(self, t, xp):
        return dists.MvNormal(loc=self.F(xp), cov=self.covX)

    def PY(self, t, xp, x):
        return dists.MvNormal(loc=self.G(x), cov=self.covY)

    # def proposal(self, t, xp, data):
    #     pred = MeanAndCov(mean=np.matmul(xp, self.F.T), cov=self.covX)
    #     f, _ = filter_step_asarray(self.G, self.covY, pred, data[t])
    #     return dists.MvNormal(loc=f.mean, cov=f.cov)

    # def proposal0(self, data):
    #     pred0 = MeanAndCov(mean=self.mu0, cov=self.cov0)
    #     f, _ = filter_step(self.G, self.covY, pred0, data[0])
    #     return dists.MvNormal(loc=f.mean, cov=f.cov)

    # def logeta(self, t, x, data):
    #     pred = MeanAndCov(mean=np.matmul(x, self.F.T), cov=self.covX)
    #     _, logpyt = filter_step_asarray(self.G, self.covY, pred, data[t + 1])
    #     return logpyt
    
class NudgedPF(ssms.Bootstrap):
  """nudged particle filter for a given state-space model.
  
  Parameters
  ----------
  
  ssm: StateSpaceModel object
      the considered state-space model
  data: list-like
      the data
  
  Returns
  -------
  FeynmanKac object
      the Feynman-Kac representation of the nudged particle filter for the
      considered state-space model
  
  """
  
  def M0(self, N):
      x_prop = self.ssm.PX0().rvs(size=N)
      # return self.ssm.PX0().rvs(size=N)
      if np.isnan(self.data[0]):
        new_filt = x_prop
      else:
        # mapped_X_prop = self.ssm.PY(0, None, x_prop).rvs(size=x_prop.shape[0])
        # if x_prop.ndim == 1:
        #   ndX = 1
        #   Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:].squeeze()
        # else:
        #   ndX = x_prop.shape[1]
        #   Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:]
        # CppGamma = np.cov(mapped_X_prop, rowvar=False)
        # if mapped_X_prop.ndim == 1:
        #   new_filt = x_prop - (((mapped_X_prop - self.data[0])/CppGamma)*Cup).T
        # else:
        #   new_filt = x_prop - (Cup@(np.linalg.solve(CppGamma, mapped_X_prop.T - self.data[0].T))).T
        new_filt, K, Gamma = EnK_step(self.ssm, 0, None, x_prop, self.data[0])
      
      self.nudged_particles = new_filt 
      self.nudging = new_filt - x_prop # need for computation of logG! make sure order is always M, then logG
      self.K = K
      return new_filt
  # def M(self, t, xp):
  #     return self.ssm.proposal(t, xp, self.data).rvs(size=xp.shape[0]) # change this so we don't need the proposal0 to be a method of ssm
  
  
  def M(self, t, xp):
      x_prop = self.ssm.PX(t, xp).rvs(size=xp.shape[0])
      if np.isnan(self.data[t]):
        new_filt = x_prop
      else:
        # mapped_X_prop = self.ssm.PY(t, xp, x_prop).rvs(size=xp.shape[0])
        # if x_prop.ndim == 1:
        #   ndX = 1
        #   Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:].squeeze()
        # else:
        #   ndX = x_prop.shape[1]
        #   Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:]
        # CppGamma = np.cov(mapped_X_prop, rowvar=False)
        # if mapped_X_prop.ndim == 1:
        #   new_filt = x_prop - (((mapped_X_prop - self.data[t])/CppGamma)*Cup).T
        # else:
        #   new_filt = x_prop - (Cup@(np.linalg.solve(CppGamma, mapped_X_prop.T - self.data[t].T))).T
        new_filt, K, Gamma = EnK_step(self.ssm, t, xp, x_prop, self.data[t])
      
      self.nudged_particles = new_filt 
      self.nudging = new_filt - x_prop # need for computation of logG! make sure order is always M, then logG
      self.K = K
      return new_filt
  def logG(self, t, xp, x):
      if t == 0:
        return (
            self.ssm.PX0().logpdf(x)
            + self.ssm.PY(0, xp, x).logpdf(self.data[0])
            - self.ssm.PX0().logpdf(x-self.nudging)
        )#return self.ssm.PY(0, xp, x).logpdf(self.data[0])   
      else:
        return (
            self.ssm.PX(t, xp).logpdf(x)
            + self.ssm.PY(t, xp, x).logpdf(self.data[t])
            - self.ssm.PX(t, xp).logpdf(x-self.nudging)
        )
  # def logG(self, t, xp, x):
  #     if t == 0:
  #         return (
  #             self.ssm.PX0().logpdf(x)
  #             + self.ssm.PY(0, xp, x).logpdf(self.data[0])
  #             - self.ssm.proposal0(self.data).logpdf(x) # specify the logpdf directly
  #         )
  #     else:
  #         return (
  #             self.ssm.PX(t, xp).logpdf(x)
  #             + self.ssm.PY(t, xp, x).logpdf(self.data[t])
  #             - self.ssm.proposal(t, xp, self.data).logpdf(x)
  #         )
  
  def Gamma0(self, u): 
      return self.ssm.proposal0(self.data).ppf(u)
  
  def Gamma(self, t, xp, u):
      return self.ssm.proposal(t, xp, self.data).ppf(u)

class EnsembleKalman(particles.FeynmanKac):
  """ Ensemble Kalman formalism of a given state-space model.
  
  Note that the EnKF is an approximate algorithm! This means that the Feynman-Kac
  model is only an exact representation of the state space model if the setting is linear and Gaussian
  
  Parameters
  ----------

  ssm: `StateSpaceModel` object
      the considered state-space model
  data: list-like
      the data

  Returns
  -------
  `FeynmanKac` object
      the Feynman-Kac representation of the Ensemble Kalman filter for the
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
      x_prop = self.ssm.PX(t, xp).rvs(size=xp.shape[0])
      if np.isnan(self.data[t]):
        new_filt = x_prop
      else:
        new_filt, _, _ = EnK_step(self.ssm, t, xp, x_prop, self.data[t])
      return new_filt
    
  def logG(self, t, xp, x):
      return np.zeros(x.shape[0])

    
class WEnKF(ssms.Bootstrap):
    # NOT YET FUNCTIONAL!
    """weighted Ensemble Kalman filter for a given state-space model.

    Parameters
    ----------

    ssm: StateSpaceModel object
        the considered state-space model
    data: list-like
        the data

    Returns
    -------
    FeynmanKac object
        the Feynman-Kac representation of the WEnKF for the
        considered state-space model

    """
    def __init__(self, ssm, data):
      assert isinstance(ssm, MVNonlinearGauss)
      super().__init__(ssm, data)

    def M0(self, N):
        x_prop = self.ssm.PX0().rvs(size=N)
        # return self.ssm.PX0().rvs(size=N)
        if np.isnan(self.data[0]):
          new_filt = x_prop
          self.K = 0
        else:

          new_filt, correction_logweights = EnK_step_for_WEnKF(self.ssm, 0, None, x_prop, self.data[0], return_weights=True)
          # self.K = K
          # self.Gamma = Gamma
          # Qhatinv = np.linalg.inv(Qhat)
          # new_filt
          # corrterm1 = np.dot()
          self.correction_logweights = correction_logweights
 
        return new_filt
    # def M(self, t, xp):
    #     return self.ssm.proposal(t, xp, self.data).rvs(size=xp.shape[0]) # change this so we don't need the proposal0 to be a method of ssm


    def M(self, t, xp):
        x_prop = self.ssm.PX(t, xp).rvs(size=xp.shape[0])
        if np.isnan(self.data[t]):
          new_filt = x_prop
          self.K = 0
        else:
          # mapped_X_prop = self.ssm.PY(t, xp, x_prop).rvs(size=xp.shape[0])
          # if x_prop.ndim == 1:
          #   ndX = 1
          #   Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:].squeeze()
          # else:
          #   ndX = x_prop.shape[1]
          #   Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:]
          # CppGamma = np.cov(mapped_X_prop, rowvar=False)
          # if mapped_X_prop.ndim == 1:
          #   new_filt = x_prop - (((mapped_X_prop - self.data[t])/CppGamma)*Cup).T
          # else:
          #   new_filt = x_prop - (Cup@(np.linalg.solve(CppGamma, mapped_X_prop.T - self.data[t].T))).T

          new_filt, correction_logweights = EnK_step_for_WEnKF(self.ssm, t, xp, x_prop, self.data[t], return_weights=True)
          
          # self.K = K
          # self.Gamma = Gamma
          self.correction_logweights = correction_logweights
        
        # self.nudged_particles = new_filt 
        # self.nudging = new_filt - x_prop # need for computation of logG! make sure order is always M, then logG
        
        return new_filt
    def logG(self, t, xp, x):
        if t == 0:
          # if x.ndim == 1:
          #   mat = self.K**2 * self.Gamma
          # else:
          #   mat = self.K @ self.Gamma @ self.K
          return (
              #self.ssm.PX0().logpdf(x)
              self.ssm.PY(0, xp, x).logpdf(self.data[0])
              + self.correction_logweights
              #+ 0.5*np.dot(x-self.nudging, np.dot(np.atleast_2d(mat),( x-self.nudging)))#- self.ssm.PX0().logpdf(x-self.nudging)
          )#return self.ssm.PY(0, xp, x).logpdf(self.data[0])   
        else:
          return (
              self.ssm.PY(0, xp, x).logpdf(self.data[0])
              + self.correction_logweights
              # self.ssm.PX(t, xp).logpdf(x)
              # + self.ssm.PY(t, xp, x).logpdf(self.data[t])              
              # + 0.5*np.dot(x-self.nudging, mat@( x-self.nudging))#- self.ssm.PX(t, xp).logpdf(x-self.nudging)

          )
    # def logG(self, t, xp, x):
    #     if t == 0:
    #         return (
    #             self.ssm.PX0().logpdf(x)
    #             + self.ssm.PY(0, xp, x).logpdf(self.data[0])
    #             - self.ssm.proposal0(self.data).logpdf(x) # specify the logpdf directly
    #         )
    #     else:
    #         return (
    #             self.ssm.PX(t, xp).logpdf(x)
    #             + self.ssm.PY(t, xp, x).logpdf(self.data[t])
    #             - self.ssm.proposal(t, xp, self.data).logpdf(x)
    #         )

    def Gamma0(self, u): 
        return self.ssm.proposal0(self.data).ppf(u)

    def Gamma(self, t, xp, u):
        return self.ssm.proposal(t, xp, self.data).ppf(u)