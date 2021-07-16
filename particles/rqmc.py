"""Randomised quasi-Monte Carlo sequences.

"""

TOL = 1e-10

from scipy.stats import qmc

def safe_generate(N, d, engine_cls):
    eng = engine_cls(d)
    u = eng.random(N)
    v = 0.5 + (1. - TOL) * (u - 0.5)
    return v

def sobol(N, d):
    return safe_generate(N, d, qmc.Sobol)

def halton(N, d):
    return safe_generate(N, d, qmc.Halton)

def latin(N, d):
    return safe_generate(N, d, qmc.LatinHybercube)
