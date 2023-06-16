"""Randomised quasi-Monte Carlo sequences.

"""
from scipy.stats import qmc

TOL = 1e-10


def _safe_generate(N, eng):
    u = eng.random(N)
    v = 0.5 + (1.0 - TOL) * (u - 0.5)
    return v


def _get_qmc_sampler(engine_cls):
    instances = dict()

    def sampler(N, d):
        eng = instances.setdefault(d, engine_cls(d))
        return _safe_generate(N, eng)

    return sampler


sobol = _get_qmc_sampler(qmc.Sobol)
halton = _get_qmc_sampler(qmc.Halton)
latin = _get_qmc_sampler(qmc.LatinHypercube)
