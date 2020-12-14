"""Single-run variance estimators.

Overview
========

As discussed in Section 19.3 of the book, several recent papers (Chan & Lai,
2013; Lee & Whiteley, 2018; Olsson & Douc, 2019) have proposed variance
estimates that may be computed from a **single** run of the algorithm. These
estimates rely on genealogy tracking; more precisely they require to track eve
variables; i.e. the index of the ancestor at time 0 (or some other origin, in
Olsson and Douc, 2019) of each particle X_t^n. See function ``var_estimate``
for the exact expression of this type of estimate.

Variance estimators (Chan & Lai, 2013; Lee & Whiteley, 2018)
============================================================

These estimates may be *collected* (see module ``collectors``) as follows::

    #  same as before
    #  ...
    # phi = lambda x: x  # for instance
    my_alg = particles.SMC(fk=my_fk_model, N=100, var_est=phi)

This will compute at each time t an estimate of the variance of
:math:`\sum_{n=1}^N W_t^n \varphi(X_t^n)` (which is itself a particle estimate
of expectation :math:`\mathbb{Q}_t(\varphi)`).

You can also estimate the variance of the normalising constant as follows::

    #  same as before
    #  ...
    # phi = lambda x: x  # for instance
    my_alg = particles.SMC(fk=my_fk_model, N=100, var_est_norm_cst=True)

This one is a bit tricky: it returns at time t an estimate of the *relative*
variance (variance divided expectation squared) of the normalising constant
:math:`L_{t-1}` (at time t-1). This is inherent to the method. To get this
estimate at the final time T, you may add one extra (arbitrary) point to the
data.

In both cases, the estimators from Chan & Lai (2013) are computed. To compute
the version from Lee & Whiteley (2018), you must multiply by a factor
:math:`(N/(N-1))^t` at time t.

Lag-based variance estimators (Olsson and Douc, 2019)
=====================================================

The above estimators suffer from the well known problem of **particle
degeneracy**; as soon as the number of distinct ancestors falls to one, these
variance estimates equal zero. Olsson and Douc (2019) proposed a variant based
on a fixed-lag approximation. To compute it, you need to activate the tracking
of a rolling-window history, as for fixed-lag smoothing (see below)::

    k = 10  # for instance
    my_alg = particles.SMC(fk=my_fk_model, N=100, lag_based_var_est=phi,
                           store_history=k)

which is going to compute the same type of estimates, but using as eve
variables (called Enoch variables in Olsson and Douc) the index of the ancestor
of each particle :math:`X_t^n` as time :math:`t-l`, where `l` is the lag.
This collector actually computes and stores simultaneously the estimates that
correspond to lags 0, 1, ..., k (where `k` is the size of the rolling window
history). This makes it easier to assess the impact of the lag on the
estimates. Thus

    my_alg = particles.SMC(fk=my_fk_model, N=100, lag_based_var_est=phi,
                           store_history=10)
    my_alg.run()
    print(my_alg.lag_based_var_est[-1])  # prints a list of length 10

References
==========

* Chan, H.P. and Lai, T.L. (2013). A general theory of particle filters in
  hidden Markov models and some applications. Ann. Statist. 41, pp. 2877–2904.

* Lee, A and Whiteley, N (2018). Variance estimation in the particle filter.
  Biometrika 3, pp. 609-625.

* Olsson, J. and Douc, R. (2019). Numerically stable online estimation of
  variance in particle filters. Bernoulli 25.2, pp. 1504-1535.

"""

from __future__ import division, print_function

from numba import jit
import numpy as np

import particles.collectors as col

def var_estimate(W, phi_x, B):
    """Variance estimate based on genealogy tracking.

    This computes the variance estimate of Chan & Lai (2013):

        .. math::
          \sum_{n=1}^N \left\{ \sum_{m:B_t^m=n} W_t^m (\varphi(X_t^m) - \Q_t^N(\varphi)) \right\}^2

    where :math:`Q_t^N(\varphi)` is the particle estimate of :math:`Q_t(\varphi)`:

        .. math::
          Q_t^N(\varphi) = \sum_{n=1}^N W_t^n \varphi(X_t^n)

    Parameters
    ----------
    W:  (N,) numpy.array
        normalised weights (>=0, sum to one)
    phi_x: (N) or (N, d) numpy.array
        values of phi(X_t^n)
    B: (N,) int numpy.array
        eve variables

    Output
    ------
    (m, v): tuple
        estimate of :math:`Q_t(\varphi)` and its associated variance estimate

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
    return np.sum(s**2)

class VarColMixin(object):
    def update_B(self, smc):
        if smc.t == 0:
            self.B = np.arange(smc.N)
        else:
            self.B = self.B[smc.A]

class Var(col.Collector, VarColMixin):
    """Computes and collects variance estimates for a given function phi.
   """
    signature = {'phi': None}

    def fetch(self, smc):
        self.update_B(smc)
        return var_estimate(smc.W, self.phi(smc.X), self.B)

class Var_logLt(col.Collector, VarColMixin):
    """Computes and collects estimates of the relative variance of the normalising
    constant estimator.

    Note: the estimate at time t corresponds to the relative variance (variance
    divided by squared expectation) of the normalising constant estimate at
    time t-1. This *shift* is inherent to the method.

    """
    def fetch(self, smc):
        self.update_B(smc)
        return _sum_over_branches(smc.W, self.B)

class Lag_based_var(col.Collector):
    """Computes and collects Olsson and Douc (2019) variance estimates, which
    are based on a fixed-lag approximation.

    Must be used in conjunction with a rolling window history (store_history=k,
    with k an int, see module ``smoothing``). The collector computes the
    estimates for all the lags 0, ..., k. Hence, it returns a list, such that
    element i is the estimate based on lag i. This makes it easier to assess
    the impact of the lag on the estimator.

    See the module doc for more details on variance estimation.

    """
    signature = {'phi': None}
    def fetch(self, smc):
        B = smc.hist.compute_trajectories()
        return [var_estimate(smc.W, self.phi(smc.X), Bt) for Bt in B][::-1]
