#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" CPU time comparison of resampling implementations. 

    See Python corner of Chapter 9 (importance resampling) for more 
    explanations.

    Notes: 
    1. this works only if numba is installed, see install notes. 
    2. Takes approx 1h 10min
    3. If run on Python 2, you should replace range by xrange in the pure
    python version 
"""

from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import timeit

nrep = 10

Ns = [np.ceil(10**n) for n in np.linspace(2, 8, 30)]

setup = """ 
import numpy as np
from scipy import stats
from particles import resampling as rs 

def pure_python_inverse_cdf(su, W):
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, 'int')
    for n in range(M):  # xrange for Python 2
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A

N = %i 
M = %i
x = stats.norm.rvs(size=N) 
lw = np.exp(-(x-1)**2)
W = np.exp(lw-lw.max())
W = W/sum(W)
A = rs.multinomial(W) 
# to make sure that the setup time of numba is not taken into account
"""

statements = {}

statements['numba'] = "E = rs.multinomial(W) "

statements['pure python'] = """
C = pure_python_inverse_cdf(rs.uniform_spacings(N), W)
"""

statements['searchsorted'] = """
D = np.searchsorted(np.cumsum(W), stats.uniform.rvs(size=M))
""" 

# For the record, the following two extra statements could 
# also be considered, but they are much slower 
extra_statements = {}
extra_statements['rv_dis'] = """ 
multi = stats.rv_discrete("resampling", values=(np.arange(N), W))
A = multi.rvs(size=M)
"""

extra_statements['ppf'] = """
multi = stats.rv_discrete("resampling", values=(np.arange(N), W))
B = multi.ppf(stats.uniform.rvs(size=M))
"""
results = []
for N in Ns:
    for method, state in statements.items():
        cputime = (1./nrep)*timeit.timeit(state, setup=setup%(N,N), number=nrep)
        results.append({'N':N, 'method':method, 'cputime':cputime})

# PLOT
# ====
savefigs = False  # change this to save the plots as PDFs 
plt.style.use('ggplot')
sb.set_palette(sb.dark_palette("lightgray", n_colors=4, reverse=True))

plt.figure()
for method in statements.keys():
    plt.plot(Ns, [ [r['cputime'] for r in results if r['method']==method and
                  r['N']==N] for N in Ns], label=method)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$N$')
plt.ylabel('CPU time (s)')
plt.legend(loc=2)
if savefigs:
    plt.savefig('comparison_numba_searchsorted.pdf')

plt.show()

