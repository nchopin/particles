# -*- coding: utf-8 -*-

"""
This module implements off-line smoothing algorithms as methods
of class `ParticleHistory`. Off-line smoothing amounts to approximate the
distribution of the complete trajectory X_{0:T}, given data y_{0:T}, at some
fixed time horizon T. The corresponding algorithms require: 

    1. to run a particle filter forward in time (from time 0 to time T), 
    and store its complete *history* (at each time t, the particles, their
    weights, and their ancestor indices);

    2. to do some computations on that history, e.g. reconstructing
    trajectories backward. 

In practice, to do Step 1, we initialize a SMC algorithm with option 
``store_history`` set to ``True`` (see :mod:`core` module). Then, upon completion,
the SMC object has a ``hist`` attribute, which is an instance of 
`ParticleHistory`; then to do Step 2, we use the appropriate method. 
Here is a quick example::

    # forward pass 
    pf = particles.SMC(fk=my_fk_model, N=100, store_history=True)
    # generate 20 smoothing trajectories
    trajectories = pf.backward_sampling(20)  

For more details, see the documentation of `ParticleHistory`. 

.. warning:: the complete history of a particle filter may take a lot of
  memory. 

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import random
from scipy import stats  # worker
import time

import particles # worker 
from particles import hilbert
from particles import qmc
from particles import resampling as rs


class ParticleHistory(object):
    """Particle history. 

    A class that stores the history of the particle system, i.e.
    at each time t=0,...T, the N particles, their weights, and their ancestors.
    Smoothing algorithms are methods of this class.

    `SMC` creates an object of this class when invoked with
    ``store_history=True``, and then save at every time t the set of particles,
    their weights (and their logarithm), and the ancestor variables. 

    Attributes
    ----------
    X: list
        X[t] is the object that represents the N particles at iteration t
    wgt: list
        wgt[t] is a Weights object (see module resampling) that represents 
        the N weights at time t
    A: list
        A[t] is the vector of ancestor indices at time t

    """

    def __init__(self, model, N):
        """
        Parameters
        ----------
        N: int
            the number of particles
        model: `FeynmanKac` object
            the Feynman-Kac model 
        """
        self.A, self.X, self.wgt, self.h_orders = [], [], [], []
        self.model = model
        self.N = N

    @property
    def T(self):
        """Current length of history.
        """
        return len(self.X)

    def save(self, X=None, w=None, A=None):
        """Save one "page" of history at a given time. 
        """
        self.X.append(X)
        self.wgt.append(w)
        self.A.append(A)

    def extract_one_trajectory(self):
        """Extract a single trajectory from the particle history.

        The final state is chosen randomly, then the corresponding trajectory 
        is constructed backwards, until time t=0. 
        """
        traj = []
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                n = rs.multinomial_once(self.wgt[-1].W)
            else:
                n = self.A[t + 1][n]
            traj.append(self.X[t][n])
        return traj[::-1]

    def compute_trajectories(self):
        """Compute the N trajectories that constitute the current genealogy.

        Compute and add attribute B to *self* where B is an array such that
        B[t,n] is the index of ancestor at time t of particle X_T^n,
        where T is the current length of history
        """
        self.B = np.empty((self.T, self.N), 'int')
        self.B[-1, :] = self.A[-1]
        for t in reversed(range(self.T - 1)):
            self.B[t, :] = self.A[t + 1][self.B[t + 1]]

    def _check_h_orders(self):
        if not self.h_orders:
            raise ValueError('QMC FFBS requires particles have been Hilbert\
                             ordered during the forward pass')

    def backward_sampling(self, M, linear_cost=False, return_ar=False):
        """Generate trajectories using the FFBS (forward filtering backward
        sampling) algorithm. 

        Arguments
        ---------
        M: int
            number of trajectories we want to generate
        linear_cost: bool 
            if set to True, the O(N) version is used, see below. 

        return_ar: bool (default=False)
            if set to True, change the output, see below. 
        
        Returns
        -------
        paths: a list of ndarrays
            paths[t][n] is component t of trajectory m. 
        ar: float
            the overall acceptance rate of the rejection procedure

        Notes
        -----

        1. if linear_cost=False, complexity is O(TMN); i.e. O(TN^2) for M=N;
           if =True, complexity is O(T(M+N)), i.e. O(TN) for M=N.
           This requires that model has method `upper_bound_trans(self,t)`, which
           provides the log of a constant C_t such that p_t(x_t|x_{t-1})<=C_t

        2. main output is *paths*, a list of T arrays such that
           paths[t][m] is component t of trajectory m.

        3. if linear_cost=True and return_ar=True, output is tuple (paths, ar),
           where paths is as above, and ar is the overall acceptance rate 
           (of the rejection steps that choose the ancestors); otherwise 
           output is simply paths
        """
        idx = np.empty((self.T, M), dtype=int)
        idx[-1, :] = rs.multinomial(self.wgt[-1].W, M=M)
        if linear_cost:
            ar = self._backward_sampling_ON(M, idx)
        else:
            self._backward_sampling_ON2(M, idx)
        # When M=1, we want a list of states, not a list of arrays containing 
        # one state
        if M == 1:
            idx = idx.squeeze(axis=1)
        paths = [self.X[t][idx[t]] for t in range(self.T)]
        if linear_cost and return_ar:
            return (paths, ar)
        else:
            return paths

    def _backward_sampling_ON(self, M, idx):
        """O(N) version of backward sampling 

        not meant to be called directly, see backward_sampling
        """
        nattempts = 0
        for t in reversed(range(self.T - 1)):
            where_rejected = np.arange(M)
            who_rejected = self.X[t + 1][idx[t + 1, :]]
            nrejected = M
            gen = rs.MultinomialQueue(self.wgt[t].W, M=M)
            while nrejected > 0:
                nattempts += nrejected
                nprop = gen.dequeue(nrejected)
                lpr_acc = (self.model.logpt(t + 1, self.X[t][nprop], 
                                            who_rejected)
                           - self.model.upper_bound_trans(t + 1))
                newly_accepted = np.log(random.rand(nrejected)) < lpr_acc
                still_rejected = np.logical_not(newly_accepted)
                idx[t, where_rejected[newly_accepted]] = nprop[newly_accepted]
                where_rejected = where_rejected[still_rejected]
                who_rejected = who_rejected[still_rejected]
                nrejected -= sum(newly_accepted)
        return (M * (self.T - 1)) / nattempts

    def _backward_sampling_ON2(self, M, idx):
        """O(N^2) version of backward sampling 

        not meant to be called directly, see backward_sampling
        """
        for m in range(M):
            for t in reversed(range(self.T - 1)):
                lwm = (self.wgt[t].lw + self.model.logpt(t + 1, self.X[t],
                                                     self.X[t + 1][idx[t + 1, m]]))
                idx[t, m] = rs.multinomial_once(rs.exp_and_normalise(lwm))

    def backward_sampling_qmc(self, M):
        """QMC version of backward sampling.
        
        Parameters
        ----------
        M : int
            number of trajectories

        Note
        ----
        Use this only on the history of a SQMC algorithm.
        """
        self._check_h_orders()
        u = qmc.sobol(M, self.T)
        # the final particles have not been sorted
        hT = hilbert.hilbert_sort(self.X[-1])
        # searchsorted to avoid having to sort in place u according to u[:,T-1]
        idx = np.searchsorted(np.cumsum(self.wgt[-1].W[hT]), u[:, -1])
        paths = [self.X[-1][hT][idx], ]
        for t in reversed(range(self.T - 1)):
            idx = np.empty(M, 'int')
            for m, xn in enumerate(paths[-1]):
                lwm = self.wgt[t].lw + self.model.logpt(t + 1, self.X[t], xn)
                # use ordered version here
                cw = np.cumsum(rs.exp_and_normalise(lwm[self.h_orders[t]]))
                idx[m] = np.searchsorted(cw, u[m, t])
            paths.append(self.X[t][self.h_orders[t]][idx])
        paths.reverse()
        return paths


#     def backward_sampling_lincost_pedagogical(self, M):
#         """ O(N) FFBS
#
#             Don't use this! This is the *pedagogical* version of O(N) FFBS:
#             code is simpler to understand, but quite slow, because it has loops
#         """
#         if not hasattr(self.model, 'upper_bound_trans'):
#             raise ValueError('O(N) version of backward smoothing'
#                              +'requires to specify constant upper_bound_trans(t)'
#                              +' s.t. log p_t(x_t|x_{t-1})<upper_bound_trans(t)')
#         idx = np.empty((self.T, M), dtype=int)
#         idx[-1, :] = rs.multinomial(M, self.wgt[-1].W)
#         nattempts = 0
#         for t in xrange(self.T - 2, -1, -1):
#             gen = rs.MulinomialQueue(M, self.wgt[t].W)
#             for m in xrange(M):
#                 while True:
#                     nattempts += 1
#                     nprop = gen.dequeue(1)
#                     lpr_acc = (self.model.logpt(t+1, self.X[t][nprop],
#                                                 self.X[t+1][idx[t+1, m]])
#                                -self.model.upper_bound_trans(t+1))
#                     if np.log(random.rand()) < lpr_acc:
#                         break
#                 idx[t, m] = nprop
#         print('O(N) FFBS: acceptance rate is %1.2f' %
#               (M * (self.T - 1) / nattempts))
#         return [self.X[t][idx[t, :]] for t in range(self.T)]

    def twofilter_smoothing(self, t, info, phi, loggamma, linear_cost=False,
                            return_ess=False, modif_forward=None,
                            modif_info=None):
        """Two-filter smoothing.

        Parameters
        ----------
        t: time, in range 0 <= t < T-1
        info: SMC object 
            the information filter
        phi: function
            test function, a function of (X_t,X_{t+1})
        loggamma: function
            a function of (X_{t+1})
        linear_cost: bool
            if True, use the O(N) variant (basic version is O(N^2))

        Returns
        -------
        Two-filter estimate of the smoothing expectation of phi(X_t,x_{t+1})
        """
        ti = self.T - 2 - t  # t+1 in reverse
        if t < 0 or t >= self.T - 1:
            raise ValueError(
                'two-filter smoothing: t must be in range 0,...,T-2')
        lwinfo = info.hist.wgt[ti].lw - loggamma(info.hist.X[ti])
        if linear_cost:
            return self._twofilter_smoothing_ON(t, ti, info, phi, lwinfo,
                                               return_ess,
                                               modif_forward, modif_info)
        else:
            return self._twofilter_smoothing_ON2(t, ti, info, phi, lwinfo)

    def _twofilter_smoothing_ON2(self, t, ti, info, phi, lwinfo):
        """O(N^2) version of two-filter smoothing. 

        This method should not be called directly, see twofilter_smoothing.
        """
        sp, sw = 0., 0.
        upb = lwinfo.max() + self.wgt[t].lw.max()
        if hasattr(self.model, 'upper_bound_trans'):
            upb += self.model.upper_bound_trans(t + 1)
        # Loop over n, to avoid having in memory a NxN matrix
        for n in range(self.N):
            omegan = np.exp(lwinfo + self.wgt[t].lw[n] - upb
                            + self.model.logpt(t + 1, self.X[t][n],
                                               info.hist.X[ti]))
            sp += np.sum(omegan * phi(self.X[t][n], info.hist.X[ti]))
            sw += np.sum(omegan)
        return sp / sw

    def _twofilter_smoothing_ON(self, t, ti, info, phi, lwinfo, return_ess,
                               modif_forward, modif_info):
        """O(N) version of two-filter smoothing.

        This method should not be called directly, see twofilter_smoothing.
        """
        if modif_info is not None:
            lwinfo += modif_info
        Winfo = rs.exp_and_normalise(lwinfo)
        I = rs.multinomial(Winfo)
        if modif_forward is not None:
            lw = self.wgt[t].lw + modif_forward
            W = rs.exp_and_normalise(lw)
        else:
            W = self.wgt[t].W
        J = rs.multinomial(W)
        log_omega = self.model.logpt(t + 1, self.X[t][J], info.hist.X[ti][I])
        if modif_forward is not None:
            log_omega -= modif_forward[J]
        if modif_info is not None:
            log_omega -= modif_info[I]
        Om = rs.exp_and_normalise(log_omega)
        est = np.average(phi(self.X[t][J], info.hist.X[ti][I]), axis=0,
                         weights=Om)
        if return_ess:
            return (est, 1. / np.sum(Om**2))
        else:
            return est


def smoothing_worker(method=None, N=100, seed=None, fk=None, fk_info=None,
                     add_func=None, log_gamma=None):
    """Generic worker for off-line smoothing algorithms.

    This worker may be used in conjunction with utils.multiplexer in order to 
    run in parallel (and eventually compare) off-line smoothing algorithms.
    
    Parameters
    ----------
    method: string
         ['FFBS_ON', 'FFBS_ON2', 'FFBS_QMC', 
           'two-filter_ON', 'two-filter_ON_prop', 'two-filter_ON2']
    N: int
        number of particles 
    seed: int
        random generator seed; if None, generator is not seeded
    fk: Feynman-Kac object
        The Feynman-Kac model for the forward filter
    fk_info: Feynman-Kac object (default=None)
        the Feynman-Kac model for the information filter; if None, 
        set to the same Feynman-Kac model as fk, with data in reverse
    add_func: function, with signature (t, x, xf) 
        additive function, at time t, for particles x=x_t and xf=x_{t+1}
    log_gamma: function
        log of function gamma (see book)

    Returns
    -------
    a dict with fields:
        est: a ndarray of length T
        cpu_time 
    """
    T = fk.T  
    if fk_info is None: 
        fk_info = fk.__class__(ssm=fk.ssm, data=fk.data[::-1])
    if seed:
        random.seed(seed)
    est = np.zeros(T - 1)
    if method=='FFBS_QMC':
        pf = particles.SQMC(fk=fk, N=N, store_history=True)
    else:
        pf = particles.SMC(fk=fk, N=N, store_history=True)
    tic = time.clock()
    pf.run()
    if method in ['FFBS_ON', 'FFBS_ON2', 'FFBS_QMC']:
        if method.startswith('FFBS_ON'): 
            z = pf.hist.backward_sampling(N, linear_cost=(method == 'FFBS_ON'))
        else:
            z = pf.hist.backward_sampling_qmc(N)
        for t in range(T - 1):
            est[t] = np.mean(add_func(t, z[t], z[t + 1]))
    elif method in ['two-filter_ON2', 'two-filter_ON', 'two-filter_ON_prop']:
        infopf = particles.SMC(fk=fk_info, N=N, store_history=True)
        infopf.run()
        for t in range(T - 1):
            psi = lambda x, xf: add_func(t, x, xf)
            if method == 'two-filter_ON2':
                est[t] = pf.hist.twofilter_smoothing(t, infopf, psi, log_gamma)
            else:
                ti = T - 2 - t  # t+1 for info filter
                if method == 'two-filter_ON_prop':
                    modif_fwd = stats.norm.logpdf(pf.hist.X[t],
                                          loc=np.mean(infopf.hist.X[ti + 1]),
                                          scale=np.std(infopf.hist.X[ti + 1]))
                    modif_info = stats.norm.logpdf(infopf.hist.X[ti],
                                           loc=np.mean(pf.hist.X[t + 1]),
                                           scale=np.std(pf.hist.X[t + 1]))
                else:
                    modif_fwd, modif_info = None, None
                est[t] = pf.hist.twofilter_smoothing(t, infopf, psi, log_gamma,
                                                     linear_cost=True,
                                                     modif_forward=modif_fwd,
                                                     modif_info=modif_info)
    else:
        print('no such method?')
    cpu_time = time.clock() - tic
    print(method + ' took %.2f s for N=%i' % (cpu_time, N))
    return {'est': est, 'cpu': cpu_time}
