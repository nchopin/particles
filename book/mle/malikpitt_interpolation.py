import numpy as np
from numpy import random

""" A quick implementation of Malik and Pitt (2011)'s interpolation method. 

    We simply subclass SMC, so as to replace the resampling step by the
    interpolated version of Malik and Pitt. 

"""
import numba as nb
import particles


@nb.njit
def interpol(x1, x2, y1, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


@nb.njit
def avg_n_nplusone(x):
    """ returns x[0]/2, (x[0]+x[1])/2, ... (x[-2]+x[-1])/2, x[-1]/2 
    """

    y = np.zeros(1 + x.shape[0])
    hx = 0.5 * x
    y[:-1] = hx
    y[1:] += hx
    return y


@nb.njit
def _interpoled_resampling(cs, xs, u):
    N = u.shape[0]
    xrs = np.empty(N)
    where = np.searchsorted(cs, u)
    # costs O(N log(N)) but algorithm has O(N log(N)) complexity anyway
    for n in range(N):
        m = where[n]
        if m == 0:
            xrs[n] = xs[0]
        elif m == N:
            xrs[n] = xs[-1]
        else:
            xrs[n] = interpol(cs[m - 1], cs[m], xs[m - 1], xs[m], u[n])
    return xrs



def interpoled_resampling(W, x):
    """Resampling based on an interpolated CDF, as described in Malik and Pitt. 

    Parameters
    ----------
    W: (N,) array
        weights 
    x: (N,) array 
        particles

    Returns
    -------
    xrs: (N,) array
        the resampled particles 

    """
    N = W.shape[0]
    idx = np.argsort(x)
    xs = x[idx]
    ws = W[idx]
    cs = np.cumsum(avg_n_nplusone(ws))
    u = random.rand(N)
    return _interpoled_resampling(cs, xs, u)


class MalikPitt_SMC(particles.SMC):
    """Subclass of SMC that implements Malik and Pitt's interpolated resampling
    method. 

    May be instantiated exactly as particles.SMC. Note however this works only for 
    univariate state-space models. 
    """

    def resample_move(self):
        self.rs_flag = True
        self.Xp = interpoled_resampling(self.W, self.X)
        self.reset_weights()
        # A not properly defined in this case 
        self.X = self.fk.M(self.t, self.Xp)
