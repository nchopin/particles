"""
SMC samplers for binary spaces.

Overview
========

This module implements SMC tempering samplers for target distributions defined
with respect to a binary space, {0, 1}^d.  This is based on Schäfer & Chopin
(2014). Note however the version here also implements the waste-free version of
these SMC samplers, see Dang & Chopin (2020).

This module builds on the `smc_samplers` module. The general idea is that the N
particles are represented by a (N, d) boolean numpy array, and the different
components of the SMC sampler (e.g. the MCMC steps) operate on such arrays. 

More precisely, this module implements: 

* `NestedLogistic`: the proposal distribution used in Schäfer and Chopin
  (2014), which amounts to fit a logistic regression to each component i, based
  on the (i-1) previous components. This is a sub-class of
  `distributions.DiscreteDist`.

* `BinaryMetropolis`: Independent Metropolis step based on a NestedLogistic
  proposal. This is a sub-class of `smc_samplers.ArrayMetropolis`. 

* Various sub-classes of `smc_samplers.StaticModel` that implements Bayesian
  variable selection. 

See also the scripts in papers/binarySMC for numerical experiments. 


References
----------

* Dau, H. D., & Chopin, N. (2022). Waste-free sequential Monte Carlo. 
  Journal of the Royal Statistical Society Series B: Statistical Methodology, 
  84(1), 114-148.

* Schäfer, C., & Chopin, N. (2013). Sequential Monte Carlo on large binary sampling 
  spaces. Statistics and Computing, 23, 163-184.

"""

import numba
import numpy as np
import scipy as sp
from numpy import random
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from particles import distributions as dists
from particles import smc_samplers as ssps

def all_binary_words(p):
    out = np.zeros((2**p, p), dtype=bool)
    ns = np.arange(2**p)
    for i in range(p):
        out[:, i] = (ns % 2**(i + 1)) // 2**i
    return out

def log_no_warn(x):
    """log without the warning about x <= 0.
    """
    return np.log(np.clip(x, 1e-300, None))

class Bernoulli(dists.ProbDist):
    dtype = 'bool'  # TODO only dist to have this dtype

    def __init__(self, p):
        self.p = p

    def rvs(self, size=None):
        N = self.p.shape[0] if size is None else size 
        # TODO already done in distributions? 
        u = random.rand(N)
        return (u < self.p)

    def logpdf(self, x):
        return np.where(x, log_no_warn(self.p), log_no_warn(1. - self.p))


class NestedLogistic(dists.DiscreteDist):
    """Nested logistic proposal distribution. 

    Recursively, each component is either:
        * independent Bernoulli(coeffs[i, i]) if edgy[i]
        * or follows a logistic regression based on the (i-1) components
    """
    dtype = 'bool'
    def __init__(self, coeffs, edgy):
        self.coeffs = coeffs
        self.edgy = edgy
        self.dim = len(edgy)

    def predict_prob(self, x, i):
        if self.edgy[i]:
            return self.coeffs[i, i]
        else:
            if i == 0:
                lin = 0.
            else:
                lin = np.sum(self.coeffs[i, :i] * x[:, :i], axis=1)
            return expit(self.coeffs[i, i] + lin)

    def rvs(self, size=1):
        out = np.empty((size, self.dim), dtype=bool)
        for i in range(self.dim):
            out[:, i] = Bernoulli(self.predict_prob(out, i)).rvs(size=size)
        return out

    def logpdf(self, x):
        l = np.zeros(x.shape[0])
        for i in range(self.dim):
            l += Bernoulli(self.predict_prob(x, i)).logpdf(x[:, i])
        return l

    @classmethod
    def fit(cls, W, x, probs_thresh=0.02, corr_thresh=0.075):
        N, dim = x.shape
        coeffs = np.zeros((dim, dim))
        ph = np.average(x, weights=W, axis=0)
        edgy = (ph < probs_thresh) | (ph > 1. - probs_thresh)
        for i in range(dim):
            if edgy[i]:
                coeffs[i, i] = ph[i]
            else:
                preds = []  # a list of ints
                for j in range(i): # finally include all predecessors
                    pij = np.average(x[:, i] & x[:, j], weights=W, axis=0)
                    corr = corr_bin(ph[i], ph[j], pij)
                    if np.abs(corr) > corr_thresh:
                        preds.append(j)
                if preds: 
                    reg = LogisticRegression(penalty='None')
                    reg.fit(x[:, preds], x[:, i], sample_weight=W)
                    coeffs[i, i] = reg.intercept_
                    coeffs[i, preds] = reg.coef_
                else:
                    coeffs[i, i] = logit(ph[i])
        print(ph)
        sparsity = (np.sum(coeffs!=0.) - dim) / (0.5 * dim * (dim - 1))
        print(f'edgy: {np.average(edgy):f}, sparsity: {sparsity:f}')
        return cls(coeffs, edgy)

def corr_bin(pi, pj, pij):
    varij = pi * (1. - pi) * pj * (1. - pj)
    if varij <= 0:
        return 0.
    else:
        return (pij - pi * pj) / np.sqrt(varij)

class BinaryMetropolis(ssps.ArrayMetropolis):
    def calibrate(self, W, x):
        x.shared['proposal'] = NestedLogistic.fit(W, x.theta)

    def proposal(self, x, xprop):
        prop_dist = x.shared['proposal']
        xprop.theta = prop_dist.rvs(size=x.N)
        lp = (prop_dist.logpdf(x.theta) 
              - prop_dist.logpdf(xprop.theta))
        return lp

def chol_and_friends(gamma, xtx, xty, vm2):
    N, d = gamma.shape
    len_gam = np.sum(gamma, axis=1)
    ldet = np.zeros(N)
    wtw = np.zeros(N)
    for n in range(N):
        if len_gam[n] > 0:
            gam = gamma[n, :]
            xtxg = xtx[:, gam][gam, :] + vm2 * np.eye(len_gam[n])
            C = sp.linalg.cholesky(xtxg, lower=True, overwrite_a=True,
                                   check_finite=False)
            w = sp.linalg.solve_triangular(C, xty[gam], lower=True,
                                           check_finite=False)
            ldet[n] = np.sum(np.log(np.diag(C)))
            wtw[n] = w.T @ w
    return len_gam, ldet, wtw

@numba.njit(parallel=True)
def jitted_chol_and_fr(gamma, xtx, xty, vm2):
    N, d = gamma.shape
    len_gam = np.sum(gamma, axis=1)
    ldet = np.zeros(N)
    wtw = np.zeros(N)
    for n in range(N):
        gam = gamma[n, :]
        if len_gam[n] > 0:
            xtxg = xtx[:, gam][gam, :] + vm2 * np.eye(len_gam[n])
            C = np.linalg.cholesky(xtxg)
            b = np.linalg.solve(C, xty[gam])  # not solve_triangular
            ldet[n] = np.sum(np.log(np.diag(C)))
            wtw[n] = b.T @ b
    return len_gam, ldet, wtw

class VariableSelection(ssps.StaticModel):
    """Meta-class for variable selection. 

    Represents a Bayesian (or pseudo-Bayesian) posterior where:

    * the prior is wrt a vector of gamma of indicator variables (whether to
      include a variable or not)
    * the likelihood is typically the marginal likelihood of gamma, where
      the coefficient parameters have been integrated out.

    """
    def __init__(self, data=None):
        self.x, self.y = data
        self.n, self.p = self.x.shape
        self.xtx = self.x.T @ self.x
        self.yty = np.sum(self.y ** 2)
        self.xty = self.x.T @ self.y

    def complete_enum(self):
        gammas = all_binary_words(self.p)
        l = self.logpost(gammas)
        return gammas, l

    def chol_intermediate(self, gamma):
        if self.jitted:
            return  jitted_chol_and_fr(gamma, self.xtx, self.xty, self.iv2)
        else:
            return chol_and_friends(gamma, self.xtx, self.xty, self.iv2)

    def sig2_full(self):
        gamma_full = np.ones((1, self.p), dtype=bool)
        _, _, btb = chol_and_friends(gamma_full, self.xtx, self.xty, 0.)
        return (self.yty - btb) / self.n


class BIC(VariableSelection):
    """Likelihood is exp{ - lambda * BIC(gamma)}
    """
    def __init__(self, data=None, lamb=10.):
        super().__init__(data=data)
        self.lamb = lamb
        self.coef_len = np.log(self.n) * self.lamb
        self.coef_log = self.n * self.lamb
        self.coef_in_log = self.yty
        self.iv2 = 0.

    def loglik(self, gamma, t=None):
        len_gam, ldet, wtw = self.chol_intermediate(gamma)
        l = - (self.coef_len * len_gam
               + self.coef_log * np.log(self.coef_in_log - wtw))
        return l


class BayesianVS(VariableSelection):
    """Marginal likelihood for the following hierarchical model:
    Y = X beta + noise    noise ~ N(0, sigma^2)
    sigma^2 ~ IG(nu / 2, lambda*nu / 2)
    beta | sigma^2 ~ N(0, v2 sigma^2 I_p)

    Note: iv2 is inverse of v2

    """
    def __init__(self, data=None, prior=None, nu=4., lamb=None, iv2=None,
                 jitted=False):
        super().__init__(data=data)
        self.prior = prior
        self.jitted = jitted
        self.nu = nu
        self.lamb = self.sig2_full() if lamb is None else lamb
        self.iv2 = float(self.lamb / 10.) if iv2 is None else iv2
        self.set_constants()

    def set_constants(self):
        self.coef_len = - 0.5 * np.log(self.iv2) 
        # minus above because log(iv2) = - log(v2)
        self.coef_log = 0.5 * (self.nu + self.n)
        self.coef_in_log = self.nu * self.lamb + self.yty

    def loglik(self, gamma, t=None):
        len_gam, ldet, wtw = self.chol_intermediate(gamma)
        l = - (self.coef_len * len_gam + ldet
               + self.coef_log * np.log(self.coef_in_log - wtw))
        return l


class BayesianVS_gprior(BayesianVS):
    """
    Same model as parent class, except: 
    beta | sigma^2 ~ N(0, g sigma^2 (X'X)^-1)

    """
    def __init__(self, data=None, prior=None, nu=4., lamb=None, g=None,
                 jitted=False):
        self.g = self.n if g is None else g 
        super().__init__(data=data, prior=prior, nu=nu, lamb=lamb, iv2=0.,
                         jitted=jitted)

        def set_constants(self):
            self.coef_len = 0.5 * np.log(1 + self.g)
            self.coef_log = 0.5 * (self.n + self.nu)
            self.coef_in_log = nu * self.lamb + self.yty
            self.gogp1 = self.g / (self.g + 1.)

    def loglik(self, gamma, t=None):
        len_gam, _, wtw = self.chol_intermediate(gamma)
        l = - (self.coef_len * len_gam
               + self.coef_log * np.log(self.coef_in_log - self.gogp1 * wtw))
        return l
        

