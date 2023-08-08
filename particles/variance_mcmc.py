"""
MCMC variance estimators. 

Author: Hai-Dang Dau

Various estimators of the asymptotic variance of a MCMC kernel, based on M
chains of length P:

    * initial sequence estimator of Geyer
    * Tukey-Hanning

This may be used to estimate the (asymptotic) variance of estimates generated
by waste-free SMC. 
"""


import gc

import numpy as np
from scipy.signal import choose_conv_method, correlate


def MCMC_variance(X: np.ndarray, method: str):
    """

    :param X: a (P, M) numpy array which contains M MCMC chains of lengths P
    :param method: a string in ['naive', 'init_seq', 'th']
    :return: estimation of sigma^2 in the CLT for MCMC chains, or, equivalently, M*P times the variance of the estimator produced by the whole array X.
    """
    if method == 'naive':
        return MCMC_variance_naive(X)
    if method == 'init_seq':
        return MCMC_init_seq(X)
    if method == 'th':
        return MCMC_Tukey_Hanning(X)
    raise ValueError('Unknown method.')

def _mean_with_weighted_columns(X: np.ndarray, W: np.ndarray):
    """
    :param X: array of shape (P,M)
    :param W: array of shape (M,), summing to 1.
    """
    P, M = X.shape
    W = W/P
    return np.sum(X * W)

def MCMC_variance_weighted(X: np.ndarray, W:np.ndarray, method:str):
    """Like `MCMC_variance`, but each column of `X` has a weight W that sums to 1."""
    P, M = X.shape
    return MCMC_variance(M * W * (X - _mean_with_weighted_columns(X, W)), method)

def MCMC_variance_naive(X):
    P, M = X.shape
    means = np.mean(X, axis=0)
    return np.var(means) * P

def autocovariance(X: np.ndarray, order: int, mu: float = None, bias=True):
    if mu is None:
        mu = np.mean(X)
    X = X - mu
    P, M = X.shape
    if bias: # use the biased estimator
        return np.mean(X[0:(P - order)] * X[order:P]) * (P-order)/P
    else:  # * (P-order)/P
        return np.mean(X[0:(P - order)] * X[order:P])

def autocovariance_fft_single(x, mu=None, bias=True):
    """
    :param x: numpy array of shape (n,)
    :return: numpy array `res` of shape(n,), where `res[i]` is the i-th autocorrelation
    """
    if mu is None:
        mu = np.mean(x)
    x = x - mu
    res = correlate(x, x, method='fft')
    res = np.array([res[-len(x)+i] for i in range(len(x))])
    if bias:
        return res/len(x)
    else:
        return res/np.arange(len(x),0,-1)

def autocovariance_fft_multiple(X, mu=None, bias=True):
    """
    :param X: numpy array of shape (P,M), which corresponds typically to `M` MCMC runs of length `P` each.
    :return: numpy array `res` of shape (P,), where `res[i]` is the i-th order autocorrelation
    """
    if mu is None:
        mu = np.mean(X)
    P, M = X.shape
    res = np.array([autocovariance_fft_single(x=X[:,m], mu=mu, bias=bias) for m in range(M)])
    return np.mean(res, axis=0)

class AutoCovarianceCalculator:
    """An artificial device to efficiently calculate the autocovariances based
    on (possibly) multiple runs of an MCMC method.
    """
    def __init__(self, X:np.ndarray, method:str=None, bias=True):
        """
        :param X: np array of size `(M,P)`, typically the result of `M` independent MCMC runs of length `P`
        :param method: how will the covariances be calculated. `None` to let things be chosen automatically, otherwise `direct` or `fft` must be specified.
        """
        self.X = X
        self.P, self.M = X.shape
        # noinspection PyTypeChecker
        self.mu: float = np.mean(X)
        self.method = method
        self.bias = bias
        self._covariances = np.array([np.nan]*self.P)

    def __getitem__(self, k:int):
        if k >= len(self._covariances) or k < 0:
            raise IndexError
        if np.isnan(self._covariances[k]):
            if self.method is None:
                self._choose_method()
            if self.method == 'fft':
                self._covariances = autocovariance_fft_multiple(X=self.X, mu=self.mu, 
                                                                bias=self.bias)
                assert len(self._covariances) == self.P
            elif self.method == 'direct':
                self._covariances[k] = autocovariance(X=self.X, order=k, 
                                                      mu=self.mu, bias=self.bias)
            else:
                raise AssertionError("Method must be either 'fft' or 'direct'")
        return self._covariances[k]

    def _choose_method(self):
        if self.P <= 10:
            self.method = 'direct'
            return
        test = self.X[0:self.P//2,0]
        self.method = choose_conv_method(test, test)

    def __len__(self):
        return len(self._covariances)

def MCMC_init_seq(X: np.ndarray, method=None, bias=True):
    """
    initial sequence estimator, see Practical MCMC (Geyer 1992)
    Let c_0, c_1, ... be the sequence of autocorrelations. Then:

    * i is an inadmissible index if i is odd and one of the two following conditions is proved to be False:
        * c[i] + c[i-1] >= 0
        * c[i-2] + c[i-3] - c[i] - c[i-1] >= 0

    * All c_i are admissible until the first inadmissible index, or when the list runs out.
    """
    covariances = AutoCovarianceCalculator(X=X, method=method, bias=bias)
    i = 0
    while (i< len(covariances)) and (not _inadmissible(covariances, i)):
        i = i + 1
    return -covariances[0] + 2*sum([covariances[j] for j in range(i)])

def _inadmissible(c, i:int):
    """Helper for `MCMC_init_seq`
    :param c: an indexable object
    """
    if i % 2 == 0:
        return False
    try:
        val1 = c[i] + c[i-1]
    except IndexError:
        val1 = np.inf
    try:
        val2 = c[i-2] + c[i-3] - c[i] - c[i-1]
    except IndexError:
        val2 = np.inf
    return val1 < -1e-10 or val2 < -1e-10


def MCMC_Tukey_Hanning(X, method=None, bias=True, adapt_constant=True):
    """MCMC Variance estimator using spectral variance method with Tukey_Hanning window.

    See `Batch means and spectral variance estimators in MCMC, Flegal and Johns (2010)`
    """
    if np.var(X) < 1e-12:
        return 0
    covariances = AutoCovarianceCalculator(X=X, method=method, bias=bias)
    alpha = 1/4
    P = len(covariances)
    if adapt_constant:
        c = np.sqrt(3.75*MCMC_variance_naive(X)/np.var(X)) 
        # leave this alone for the moment. In high dimensional settings, it is 
        # rare that we can run Markov chain for (a lot) more than 3 
        # autocorrelation time.
    else:
        c = 1
    b = max(c * P**0.5+1,2)
    b = int(b)
    w = [1 - 2*alpha + 2*alpha * np.cos(np.pi*k/b) for k in range(b)]
    w_cov = []
    for i in np.arange(1,b):
        try:
            w_cov.append(w[i] * covariances[i])
        except IndexError:
            w_cov.append(0)
    return w[0] * covariances[0] + 2 * sum(w_cov)

def default_collector(ls: list[np.ndarray]) -> np.ndarray:
    gc.collect()
    return np.r_[tuple(ls)]

def _weighted_variance_by_columns(x: np.ndarray, W: np.ndarray) -> float:
    """Compute variance of elements of `x` where each column of `x` is weighted by `W`.
    :param W: weights, should sum to 1
    """
    P, M = x.shape
    W = W/P
    mean_of_squares = np.sum(W * x**2)
    square_of_mean = np.sum(W*x)**2
    return mean_of_squares - square_of_mean

