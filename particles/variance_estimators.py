r"""Single-run variance estimators.

Overview
========

As discussed in Section 19.3 of the book, several recent papers (Chan & Lai,
2013; Lee & Whiteley, 2018; Olsson & Douc, 2019) have proposed variance
estimates that may be computed from a **single** run of the algorithm. These
estimates rely on genealogy tracking; more precisely they require to track eve
variables; i.e. the index of the ancestor at time 0 (or some other time, in
Olsson and Douc, 2019) of each particle. See function `var_estimate`
for the exact expression of this type of estimate.

Variance estimators (Chan & Lai, 2013; Lee & Whiteley, 2018)
============================================================

These estimates may be *collected* (see module `collectors`) as follows::

    import particles
    from particles import variance_estimators as var  # this module

    # Define a FeynmanKac object, etc.
    #  ...
    phi = lambda x: x**2  # for instance
    my_alg = particles.SMC(fk=my_fk_model, N=100,
                           collect=[var.Var(phi=phi), var.Var_logLt()])

The first collector will compute at each time t an estimate of the variance of
:math:`\sum_{n=1}^N W_t^n \varphi(X_t^n)` (which is itself a particle estimate
of expectation :math:`\mathbb{Q}_t(\varphi)`). If argument ``phi`` is not provided,
the function :math:`\varphi(x)=x` will be used.

The second collector will compute an estimate of the variance of the log
normalising constant, i.e. :math:`\log L_t`.

.. note::
    The estimators found in Chan & Lai (2013) and Lee & Whiteley (2018) differ only
    by a factor :math:`(N/(N-1))^t`; the collectors above implement the former
    version, without the factor. 

Lag-based variance estimators (Olsson and Douc, 2019)
=====================================================

The above estimators suffer from the well known problem of **particle
degeneracy**; as soon as the number of distinct ancestors falls to one, these
variance estimates equal zero. Olsson and Douc (2019) proposed a variant based
on a fixed-lag approximation.  To compute it, you need to activate the tracking
of a rolling-window history, as for fixed-lag smoothing (see below)::

    my_alg = particles.SMC(fk=my_fk_model, N=100,
                           collect=[var.Lag_based_var(phi=phi)],
                           store_history=10)

which is going to compute the same type of estimates, but using as eve
variables (called Enoch variables in Olsson and Douc) the index of the ancestor
of each particle :math:`X_t^n` as time :math:`t-l`, where :math:`l` is the lag.
This collector actually computes and stores simultaneously the estimates that
correspond to lags 0, 1, ..., k (where k is the size of the rolling window
history). This makes it easier to assess the impact of the lag on the
estimates. Thus::

    print(my_alg.lag_based_var[-1])  # prints a list of length 10

Numerical experiments
=====================

See `here`_ for a jupyter notebook that illustrates these variance estimates in a
simple example.

.. _here: notebooks/variance_estimation.html

References
==========

* Chan, H.P. and Lai, T.L. (2013). A general theory of particle filters in
  hidden Markov models and some applications. Ann. Statist. 41, pp. 2877–2904.

* Lee, A and Whiteley, N (2018). Variance estimation in the particle filter.
  Biometrika 3, pp. 609-625.

* Olsson, J. and Douc, R. (2019). Numerically stable online estimation of
  variance in particle filters. Bernoulli 25.2, pp. 1504-1535.

"""


from numba import jit
import numpy as np

import particles.collectors as col

def var_estimate(W, phi_x, B):
    r"""Variance estimate based on genealogy tracking.

    This computes the variance estimate of Chan & Lai (2013):

        .. math::

           \sum_{n=1}^N \left\{ \sum_{m:B_t^m=n} W_t^m (\varphi(X_t^m) - \mathbb{Q}_t^N(\varphi)) \right\}^2

    where :math:`\mathbb{Q}_t^N(\varphi)` is the particle estimate 
    of :math:`\mathbb{Q}_t(\varphi)`:

        .. math::
          \mathbb{Q}_t^N(\varphi) = \sum_{n=1}^N W_t^n \varphi(X_t^n)

    Parameters
    ----------
    W:  (N,) numpy.array
        normalised weights (>=0, sum to one)
    phi_x: (N) or (N, d) numpy.array
        values of phi(X_t^n)
    B: (N,) int numpy.array
        eve variables

    Returns
    -------
    variance estimate

    """
    m = np.average(phi_x, weights=W, axis=0)
    phixm = phi_x - m
    w_phi = W[:, np.newaxis] * phixm if phixm.ndim == 2 else W * phixm
    if B[0] == B[-1]:
        out = np.zeros_like(m)
    else:
        out = _sum_over_branches(w_phi, B)
    return out

@jit(nopython=True)
def _sum_over_branches(w_phi, B):
    N = w_phi.shape[0]
    s = np.zeros(N)
    for m in range(N):
        s[B[m]] += w_phi[m]
    return np.sum(s**2, axis=0)

class VarColMixin:
    def update_B(self, smc):
        if smc.t == 0:
            self.B = np.arange(smc.N)
        else:
            self.B = self.B[smc.A]


class Var(col.Collector, VarColMixin):
    """Computes and collects variance estimates for a given test function phi.

    Parameters
    ----------
    phi:  callable
       the test function (default: identity function)
   """
    signature = {'phi': None}

    def test_func(self, x):
        if self.phi is None:
            return x
        else:
            return self.phi(x)

    def fetch(self, smc):
        self.update_B(smc)
        return var_estimate(smc.W, self.test_func(smc.X), self.B)

class Var_logLt(col.Collector, VarColMixin):
    """Computes and collects estimates of the variance of the log normalising
    constant estimator.
    """
    def fetch(self, smc):
        self.update_B(smc)
        return _sum_over_branches(smc.W, self.B)

class Lag_based_var(Var):
    """Computes and collects Olsson and Douc (2019) variance estimates, which
    are based on a fixed-lag approximation.

    Must be used in conjunction with a rolling window history
    (``store_history=k``, with ``k`` an int, see module `smoothing`). The
    collector computes the estimates for all the lags 0, ..., k. Hence, it
    returns a list, such that element i is the estimate based on lag i. This
    makes it easier to assess the impact of the lag on the estimator.

    Parameters
    ----------
    phi:  callable
       the test function (default: identity function)

    """
    def fetch(self, smc):
        B = smc.hist.compute_trajectories()
        return [var_estimate(smc.W, self.test_func(smc.X), Bt) for Bt in B][::-1]
