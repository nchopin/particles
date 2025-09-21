
import collections

import numpy as np
from scipy.linalg import solve

from particles import distributions as dists
from particles import state_space_models as ssms
from particles import utils
import particles



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
        new_filt, K = EnK_step(self.ssm, t, xp, x_prop, self.data[t])
      return new_filt
    
  def logG(self, t, xp, x):
      return np.zeros(x.shape[0])


def EnK_step(ssm, t, xp, x_prop, y):
  mapped_X_prop = ssm.PY(t, xp, x_prop).rvs(size=x_prop.shape[0])
  if x_prop.ndim == 1:
    ndX = 1
    Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:].squeeze()
  else:
    ndX = x_prop.shape[1]
    Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:]
  CppGamma = np.cov(mapped_X_prop, rowvar=False)
  if mapped_X_prop.ndim == 1:
    new_filt = x_prop - (((mapped_X_prop - y)/CppGamma)*Cup).T
    K = Cup/CppGamma
  else:
    new_filt = x_prop - (Cup@(np.linalg.solve(CppGamma, mapped_X_prop.T - y.T))).T
    K = Cup@(np.linalg.inv(CppGamma))
  return new_filt, K

class NudgedPF(ssms.Bootstrap):
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
          new_filt, K = EnK_step(self.ssm, 0, None, x_prop, self.data[0])
        
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
          new_filt, K = EnK_step(self.ssm, t, xp, x_prop, self.data[t])
        
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