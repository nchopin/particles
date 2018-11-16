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

qmc_warning = """
Module lowdiscrepancy does not seem to be installed (see INSTALL
notes). You will not be able to run SQMC (quasi-Monte Carlo version
of SMC).
"""
try:
    from particles import lowdiscrepancy
except ImportError:
    import warnings
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
    For scrambling, seed is set randomly. 

    Fun fact: this venerable but playful piece of Fortran code occasionally
    returns numbers above 1. (i.e. for a very small number of seeds); when this
    happen we just start over (since the seed is randomly generated). 
    """
    while(True):
        seed = np.random.randint(2**32)
        out = lowdiscrepancy.sobol(N, dim, scrambled, seed, 1, 0)
        if (scrambled == 0) or ((out < 1.).all() and (out > 0.).all()):
            # no need to test if scrambled==0
            return out


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
