# -*- coding: utf-8 -*-

"""QMC and RQMC sequences. 

This module is a simple wrapper for LowDiscrepancy.f, a fortran piece of code
that implements the Sobol' and Halton sequence. (The same fortran code 
is used the `randtoolbox 
<https://cran.r-project.org/web/packages/randtoolbox/index.html>` package
in ``R``.)

To use this module, you must first compile LowDiscrepancy.f, as explained in
the installation notes. 
"""

import numpy as np
import warnings

qmc_warning = """
Module lowdiscrepancy does not seem to be installed (see INSTALL
notes). You will not be able to run SQMC (quasi-Monte Carlo version
of SMC).
"""

sobol_warning =  """
lowdiscrepancy.sobol(%i, %i, %i, %i, 1, 0) generated points outside (0, 1)
"""
try:
    from particles import lowdiscrepancy
except ImportError:
    warnings.warn(qmc_warning)


def sobol(N, dim, scrambled=1):
    """ Sobol sequence. 

    Parameters
    ----------
    N : int
        length of sequence
    dim: int
        dimension
    scrambled: int
        which scrambling method to use: 

            + 0: no scrambling
            + 1: Owen's scrambling
            + 2: Faure-Tezuka
            + 3: Owen + Faure-Tezuka

    Returns
    -------
    (N, dim) numpy array. 


    Notes 
    -----
    One of the argument of the underlying Fortran function is the seed for the
    random generator used for scrambling. We simply generate this seed
    uniformly (between 0 and 2^32 - 1). There is a very small number of seeds
    that generate points that are == 0 (This has been reported to the
    maintainer of randtoolbox). When this happens, we generate a warning and
    start over (i.e.  we re-generate another random seed, and compute a new
    scrambled Sobol point set. 
    """
    while(True):
        seed = np.random.randint(2**32)
        out = lowdiscrepancy.sobol(N, dim, scrambled, seed, 1, 0)
        if np.all(out < 1.) and np.all(out > 0.):
            return out
        else:
            warnings.warn(sobol_warning % (N, dim, scrambled, seed))

def halton(N, dim):
    """ Halton sequence.

    Component i of the sequence consists of a Van der Corput sequence in base b_i,
    where b_i is the i-th prime number. 

    Parameters
    ----------
    N : int
        length of sequence
    dim: int
        dimension

    Returns
    -------
    (N, dim) numpy array. 

    """
    return lowdiscrepancy.halton(N, dim, 1, 0)
