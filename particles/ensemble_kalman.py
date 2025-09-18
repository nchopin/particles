
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
      mapped_X_prop = self.ssm.PY(t, xp, x_prop).rvs(size=xp.shape[0])
      if x_prop.ndim == 1:
        ndX = 1
        Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:].squeeze()
      else:
        ndX = x_prop.shape[1]
        Cup = np.cov(x_prop,mapped_X_prop, rowvar=False)[0:ndX,ndX:]
      CppGamma = np.cov(mapped_X_prop, rowvar=False)
      if mapped_X_prop.ndim == 1:
        new_filt = x_prop - (((mapped_X_prop - self.data[t])/CppGamma)*Cup).T
      else:
        new_filt = x_prop - (Cup@(np.linalg.solve(CppGamma, mapped_X_prop.T - self.data[t].T))).T
      return new_filt
    
  def logG(self, t, xp, x):
      return np.zeros(x.shape[0])

