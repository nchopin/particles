# -*- coding: utf-8 -*-

"""
This module implements resampling schemes, plus some basic numerical 
functions related to weights and weighted data (ESS, weighted mean, etc). 
The recommended import is::

    from particles import resampling as rs

Resampling schemes
==================

All the resampling schemes are implemented as functions with the following
signature:: 

    A = rs.scheme(W, M=None) 

where: 

  * ``W`` is a vector of N normalised weights (i.e. positive and summing to
    one).

  * ``M`` (int) is the number of resampled indices that must be generated; 
    (optional, set to N if not provided).

  * ``A`` is a ndarray containing the M resampled indices 
    (i.e. ints in the range 0, ..., N-1). 

Here the list of currently implemented resampling schemes: 

* `multinomial`
* `residual`    
* `stratified`
* `systematic`
* `ssp`      
__
Alternative ways to sample from a multinomial distribution 
==========================================================

Function `multinomial` effectively samples M times from the multinomial
distribution that produces output n with probability ``W[n]``.  It does so
using an algorithm with complexity O(M+N), as explained in Section 9.4 of the
book. However, this function is not really suited: 

    1. if you want to draw only **once** from that distribution; 

    2. If you do not know in advance how many draws you need. 

The two functions below cover these scenarios: 

* `multinomial_once`
* `MultinomialQueue` 

Other functions related to resampling 
=====================================

The following basic functions are called by some resampling schemes, 
but they might be useful in other contexts. 

* `inverse_cdf`
* `uniform_spacings`

Other functions of interest
===========================

In `particles`, importance weights and similar quantities are always computed 
and stored on the log-scale, to avoid numerical overflow. This module also 
contains a few basic functions to deal with log-weights: 

* `essl`
* `exp_and_normalise`
* `log_mean_exp`
* `log_sum_exp`
* `wmean_and_var`
* `wmean_and_var_str_array`
* `wquantiles`

"""

from __future__ import division, print_function

import functools
import numpy as np
from numpy import random
from numba import jit

def exp_and_normalise(lw):
    """Exponentiate, then normalise (so that sum equals one).

    Arguments
    ---------
    lw: ndarray
        log weights.

    Returns
    -------
    W: ndarray of the same shape as lw
        W = exp(lw) / sum(exp(lw))

    Note
    ----
    uses the log_sum_exp trick to avoid overflow (i.e. subtract the max 
    before exponentiating)

    See also
    --------
    log_sum_exp
    log_mean_exp

    """
    w = np.exp(lw - lw.max())
    return w / w.sum()


def essl(lw):
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    lw: (N,) ndarray
        log-weights 

    Returns
    -------
    float 
        the ESS of weights w = exp(lw), i.e. the quantity  
        sum(w**2) / (sum(w))**2

    Note 
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights. 
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero. 

    """
    w = np.exp(lw - lw.max())
    return (w.sum())**2 / np.sum(w**2)

class Weights(object):
    """ A class to store N log-weights, compute the normalised weights, and
    their ESS. 
    """

    def __init__(self, lw=None):
        self.lw = lw
        if lw is not None:
            self.lw[np.isnan(self.lw)] = - np.inf
            self.W = exp_and_normalise(lw)
            self.ESS  = 1. / np.sum(self.W ** 2)

    def add(self, delta):
        if self.lw is None:
            return Weights(lw=delta)
        else:
            return Weights(lw=self.lw + delta)

def log_sum_exp(v):
    """Log of the sum of the exp of the arguments. 

    Arguments
    ---------
    v: ndarray

    Returns
    -------
    l: float 
        l = log(sum(exp(v)))

    Note
    ----
    use the log_sum_exp trick to avoid overflow: i.e. we remove the max of v
    before exponentiating, then we add it back

    See also
    --------
    log_mean_exp

    """
    m = v.max()
    return m + np.log(np.sum(np.exp(v - m)))


def log_sum_exp_ab(a, b):
    """log_sum_exp for two scalars.

    Parameters
    ----------
    a, b: float 

    Returns
    -------
    c: float
        c = log(e^a + e^b)
    """
    if a > b:
        return a + np.log(1. + np.exp(b - a))
    else:
        return b + np.log(1. + np.exp(a - b))


def log_mean_exp(v, W=None):
    """Returns log of (weighted) mean of exp(v). 

       Parameters
       ----------
       v: ndarray 
           data, should be such that v.shape[0] = N 

       W: (N,) ndarray, optional  
            normalised weights (>=0, sum to one)

       Returns
       -------
       ndarray 
           mean (or weighted mean, if W is provided) of vector exp(v) 

       See also
       --------
       log_sum_exp

    """
    m = v.max()
    V = np.exp(v - m)
    if W is None:
        return m + np.log(np.mean(V))
    else:
        return m + np.log(np.average(V, weights=W))


def wmean_and_var(W, x):
    """Component-wise weighted mean and variance.

    Parameters
    ----------
    W: (N,) ndarray
        normalised weights (must be >=0 and sum to one). 
    x: ndarray (such that shape[0]==N)
        data 

    Returns
    -------
    dictionary 
        {'mean':weighted_means, 'var':weighted_variances}
    """
    m = np.average(x, weights=W, axis=0)
    m2 = np.average(x**2, weights=W, axis=0)
    v = m2 - m**2
    return {'mean': m, 'var': v}

def wmean_and_var_str_array(W, x): 
    """Weighted mean and variance of each component of a structured array.

    Parameters
    ----------
    W: (N,) ndarray
        normalised weights (must be >=0 and sum to one). 
    x: (N,) structured array
        data 

    Returns
    -------
    dictionary 
        {'mean':weighted_means, 'var':weighted_variances}
    """
    m = np.empty(shape=x.shape[1:], dtype=x.dtype)
    v = np.empty_like(m)
    for p in x.dtype.names:
        m[p], v[p] = wmean_and_var(W, x[p]).values()
    return {'mean': m, 'var': v}

def _wquantiles(W, x, alphas):
    N = W.shape[0]
    order = np.argsort(x)
    cw = np.cumsum(W[order])
    indices = np.searchsorted(cw, alphas)
    quantiles = []
    for a, n in zip(alphas, indices):
        prev = np.clip(n - 1, 0, N - 2)
        q = np.interp(a, cw[prev:prev+2], x[order[prev:prev+2]])
        quantiles.append(q)
    return quantiles

def wquantiles(W, x, alphas=(0.25, 0.50, 0.75)):
    """Quantiles for weighted data.

    Parameters
    ----------
    W: (N,) ndarray
        normalised weights (weights are >=0 and sum to one)
    x: (N,) or (N,d) ndarray
        data
    alphas: list-like of size k (default: (0.25, 0.50, 0.75))
        probabilities (between 0. and 1.) 

    Returns
    -------
    a (k,) or (d, k) ndarray containing the alpha-quantiles 
    """
    if len(x.shape) == 1:
        return _wquantiles(W, x, alphas=alphas)
    elif len(x.shape) == 2: 
        return np.array([_wquantiles(W, x[:, i], alphas=alphas) 
                         for i in range(x.shape[1])])

def wquantiles_str_array(W, x, alphas=(0.25, 0.50, 0,75)):
    """quantiles for weighted data stored in a structured array. 

    Parameters
    ----------
    W: (N,) ndarray
        normalised weights (weights are >=0 and sum to one)
    x: (N,) structured array
        data
    alphas: list-like of size k (default: (0.25, 0.50, 0.75))
        probabilities (between 0. and 1.) 

    Returns
    -------
    dictionary {p: quantiles} that stores for each field name p
    the corresponding quantiles

    """
    return {p: wquantiles(W, x[p], alphas) for p in x.dtype.names}

###################
# Resampling schemes 
####################

rs_funcs = {}  # populated by the decorator below

# generic docstring of resampling schemes; assigned by decorator below
rs_doc = """%s resampling. 
  
             Parameters
             ----------
             W: (N,) ndarray 
                 normalized weights (>=0, sum to one)
             M: int, optional (set to N if missing)
                 number of resampled points.
 
             Returns
             -------
             (M,) ndarray
                 M ancestor variables, drawn from range 0,...,N-1
         """ 

def resampling_scheme(func):
    """Decorator for resampling schemes."""

    @functools.wraps(func)
    def modif_func(W, M=None):
        M = W.shape[0] if M is None else M
        return func(W, M)

    rs_funcs[func.__name__] = modif_func
    modif_func.__doc__ = rs_doc % func.__name__.capitalize()
    return modif_func

def resampling(scheme, W, M=None):
    try:
        return rs_funcs[scheme](W, M=M)
    except KeyError:
        raise ValueError('%s: not a valid resampling scheme' % name)


@jit
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution. 

        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]). 
        W: (N,) ndarray 
            a vector of N normalized weights (>=0 and sum to one)

        Returns
        ------- 
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, 'int')
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A


def uniform_spacings(N):
    """ Generate ordered uniform variates in O(N) time.

    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates

    Returns
    -------
    (N,) float ndarray 
        the N ordered variates (ascending order)

    Note
    ----
    This is equivalent to:

        from numpy import random
        u = sort(random.rand(N))

    but the line above has complexity O(N*log(N)), whereas the algorithm 
    used here has complexity O(N)

    """
    z = np.cumsum(-np.log(random.rand(N + 1)))
    return z[:-1] / z[-1]


def multinomial_once(W):
    """ Sample once from a Multinomial distribution

    Parameters
    ----------
    W: (N,) ndarray
        normalized weights (>=0, sum to one)

    Returns
    -------
    int 
        a single draw from the discrete distribution that generates n with 
        probability W[n] 

    Note
    ----
    This is equivalent to 

       A = multinomial(W, M=1)

    but it is faster. 
    """
    return np.searchsorted(np.cumsum(W), random.rand())


@resampling_scheme
def multinomial(W, M):
    return inverse_cdf(uniform_spacings(M), W)


@resampling_scheme
def stratified(W, M):
    su = (random.rand(M) + np.arange(M)) / M
    return inverse_cdf(su, W)


@resampling_scheme
def systematic(W, M):
    su = (random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)


@resampling_scheme
def residual(W, M):
    N = W.shape[0]
    A = np.empty(M, 'int')
    MW = M * W
    intpart = np.floor(MW).astype('int')
    sip = np.sum(intpart)
    res = MW - intpart
    sres = M - sip
    A[:sip] = np.arange(N).repeat(intpart)
    # each particle n is repeated intpart[n] times
    if sres > 0:
        A[sip:] = multinomial(res / sres, M=sres)
    return A


@resampling_scheme
@jit
def ssp(W, M):
    N = W.shape[0]
    MW = M * W
    nb_children = np.floor(MW).astype('int')
    xi = MW - nb_children
    u = random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1. - xi[i])  # increase i, decr j
        delta_j = min(xi[i], 1. - xi[j])  # the opposite
        pj = delta_i / (delta_i + delta_j) # prob we should inc j, dec i
        if u[k] < pj:  # swap i, j, so that we always inc i
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1. - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nb_children[i] += 1
            i = k + 2 
    return np.arange(N).repeat(nb_children)


class MultinomialQueue(object):
    """On-the-fly generator for the multinomial distribution. 

       To obtain k1,k2, ... draws from the multinomial distribution with
       weights W, do:: 

           g = MulinomialQueue(M,W)
           first_set_of_draws = g.dequeue(k1)
           second_set_of_draws = g.dequeue(k2)
           # ... and so on

       At initialisation, a vector of size M is created, and each time dequeue(k)
       is invoked, the next k draws are produced. When all the draws have been
       "served", a new vector of size M is generated. (If no value is given
       for M, we take M=N, the length of vector W.)

       In this way, we have on average a O(1) complexity for each draw,
       without knowing in advance how many draws will be needed.
    """

    def __init__(self, W, M=None):
        self.W = W
        self.M = W.size if M is None else M
        self.j = 0  # points to first non-consumed item in the queue
        self.enqueue()

    def enqueue(self):
        perm = random.permutation(self.M)
        self.A = multinomial(self.W, M=self.M)[perm]

    def dequeue(self, k):
        """Outputs *k* draws from the multinomial distribution."""
        if self.j + k <= self.M:
            out = self.A[self.j:(self.j + k)]
            self.j += k
        elif k <= self.M:
            out = np.empty(k, 'int')
            nextra = self.j + k - self.M
            out[:(k - nextra)] = self.A[self.j:]
            self.enqueue()
            out[(k - nextra):] = self.A[:nextra]
            self.j = nextra
        else:
            raise ValueError('MultinomialQueue: k must be <= M (the max \
                             capacity of the queue)')
        return out
