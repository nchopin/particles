"""
SMC samplers for binary spaces.

This module implements SMC tempering samplers for target distributions defined
with respect to a binary space, {0, 1}^d.  This is based on Schäfer & Chopin
(2014). Note however the version here also implements the waste-free version of
these SMC samplers, see Dang & Chopin (2020).

This module builds on the `smc_samplers` module. The general idea is that the N
particles are represented by a (N, d) boolean numpy array, and the different
components of the SMC sampler (e.g. the MCMC steps) operate on such arrays. 

More precisely, this module implements: 

* NestedLogistic: the proposal distribution used in Schäfer and Chopin
  (2014), which amounts to fit a logistic regression to each component i, based
  on the (i-1) previous components. This is sub-class of
  `distributions.DiscreteDist`.

* BinaryMetropolis: Independent Metropolis step based on a NestedLogistic
  proposal. This is sub-class of `smc_samplers.ArrayMetropolis`. 

* Various sub-classes of `smc_samplers.StaticModel` that implements Bayesian
variable selection: BayesianVS_gprior, BayesianVS_gprior.

See the script in papers/binary for numerical experiments. 

TODO:
    * make jitted version works? 
    * check logistic reg *really* needed
    * concrete

"""

import numba
import numpy as np
from numpy import random
from scipy import linalg
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

from particles import distributions as dists
from particles import smc_samplers as ssps

def all_binary_words(p):
    out = np.zeros((2**p, p), dtype=np.bool)
    ns = np.arange(2**p)
    for i in range(p):
        out[:, i] = (ns % 2**(i + 1)) // 2**i
    return out

def log_no_warn(x):
    """log without the warning about x <= 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.log(x)
    return out

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
        out = np.empty((size, self.dim), dtype=np.bool)
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
                    reg = LogisticRegression(penalty='none')
                    reg.fit(x[:, preds], x[:, i], sample_weight=W)
                    coeffs[i, i] = reg.intercept_
                    coeffs[i, preds] = reg.coef_
                else:
                    coeffs[i, i] = logit(ph[i])
        print(ph)
        sparsity = (np.sum(coeffs!=0.) - dim) / (0.5 * dim * (dim - 1))
        print('edgy: %f, sparsity: %f' % (np.average(edgy), sparsity))
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


def vhat_and_chol(xtx, xty, yty, n):
    C = linalg.cholesky(xtx, lower=True)
    bhat = linalg.solve_triangular(C, xty, lower=True)
    vhat = (yty - np.sum(bhat**2)) / n
    return vhat, C

class VariableSelection(ssps.StaticModel):
    """Meta-class for variable selection. 
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

class BIC(VariableSelection):
    """Likelihood is exp{ - lambda * BIC(gamma)}
    """
    def __init__(self, data=None, lamb=10.):
        super().__init__(data=data)
        self.lamb = lamb

    def loglik(self, gamma, t=None):
        l = np.sum(gamma, axis=1) * np.log(self.n)
        N, d = gamma.shape
        for n in range(N):
            gam = gamma[n, :]
            xtxg = self.xtx[:, gam][gam, :]
            vh, _ = vhat_and_chol(xtxg, self.xty[gam], self.yty, self.n)
            l[n] += self.n * np.log(vh)
        return - self.lamb * l 

class BayesianVS(VariableSelection):
    """Marginal likelihood for the following hierarchical model:
    Y = X beta + noise    noise ~ N(0, sigma^2)
    sigma^2 ~ IG(nu / 2, lambda*nu / 2)
    beta | sigma^2 ~ N(0, v2 sigma^2 I_p)

    """
    def __init__(self, data=None, prior=None, nu=4., lamb=None, v2=None):
        super().__init__(data=data)
        self.prior = prior
        self.nu = nu
        if lamb is None:
            self.lamb, _ = vhat_and_chol(self.xtx, self.xty, self.yty, self.n)
        else:
            self.lamb = lamb
        self.v2 = 10. / self.lamb if v2 is None else v2
        self.logv = 0.5 * np.log(self.v2)
        self.coef1 = 0.5 * (self.nu + self.n)
        self.coef2 = self.nu * self.lamb / self.n

    def loglik(self, gamma, t=None):
        N = gamma.shape[0]
        len_gam = np.sum(gamma, axis=1)
        l = - self.logv * len_gam
        for n in range(N):
            gam = gamma[n]
            if len_gam[n] > 0:
                xtxg = (self.xtx[gam, :][:, gam] 
                        + (1. / self.v2) * np.eye(len_gam[n]))
                vh, C = vhat_and_chol(xtxg, self.xty[gam], self.yty, self.n)
                cii = np.sum(np.log(np.diag(C)))
            else:
                cii = 0.
                vh = self.yty / self.n
            l[n] -= (cii + self.coef1 * np.log(self.coef2 + vh))
        return l

    def loglik_fast(self, gamma, t=None):
        return jitted_loglik(gamma, self.xtx, self.yty, self.xty, self.n,
                             self.logv, self.v2, self.coef1, self.coef2)

@numba.jit(parallel=True)
def jitted_loglik(gamma, xtx, yty, xty, n, logv, v2, coef1, coef2):
    N = gamma.shape[0]
    len_gam = np.sum(gamma, axis=1)
    l = - len_gam * logv
    for n in range(N):
        gam = gamma[n]
        if len_gam[n] > 0:
            xtxg = (xtx[gam, :][:, gam] + (1. / v2) *
                    np.eye(len_gam[n]))
            vh, C = vhat_and_chol(xtxg, xty[gam], yty, n)
            cii = np.sum(np.log(np.diag(C)))
        else:
            cii = 0.
            vh = yty / n
        l[n] -= (cii + coef1 * np.log(coef2 + vh))
    return l

class BayesianVS_gprior(BayesianVS):
    """
    Same model as parent class, except: 
    beta | sigma^2 ~ N(0, g sigma^2 (X'X)^-1)

    """
    def __init__(self, data=None, prior=None, nu=4., lamb=None, g=None):
        super().__init__(data=data, prior=prior, nu=nu, lamb=lamb, v2=None)
        self.g = self.n if g is None else g 
        # constants
        self.cst_lg = - 0.5 * np.log(1 + self.g)
        self.nulayty = nu * self.lamb + self.yty
        self.gogp1 = self.g / (self.g + 1.)

    def loglik(self, gamma, t=None):
        N, d = gamma.shape
        l = self.cst_lg * np.sum(gamma, axis=1)
        for n in range(N):
            gam = gamma[n]
            if np.any(gam):
                xtxg = self.xtx[gam, :][:, gam]
                C = linalg.cholesky(xtxg, lower=True)
                W = linalg.solve_triangular(C, self.xty[gam], lower=True)
                WtW = W.T @ W
            else:
                WtW = 0.
            l[n] -= self.coef1 * np.log(self.nulayty - self.gogp1 * WtW)
        return l
        

