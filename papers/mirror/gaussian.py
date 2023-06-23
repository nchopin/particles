from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import particles
from particles import resampling as rs
from particles import smc_samplers as ssps
from particles import distributions as dists

HALFLOGTWOPI = 0.5 * np.log(2. * np.pi)

class BetterTempering(ssps.AdaptiveTempering):
    def __init__(self, model=None, wastefree=True, len_chain=10, move=None, 
                 chisq=1.):
        super().__init__(
            model=model, wastefree=wastefree, len_chain=len_chain, move=move
        )
        self.chisq = chisq
    def logG(self, t, xp, x):
        epn = x.shared['exponents'][-1]
        fisher = np.var(x.llik)
        if fisher > 0.:
            delta = np.sqrt(self.chisq / fisher)
            new_epn = epn + delta
        if new_epn > 1. or fisher < 0:
            new_epn = 1.
            delta = 1. - epn
        x.shared['exponents'].append(new_epn)
        return self.logG_tempering(x, delta)

class ArrayDiagRandomWalk(ssps.ArrayRandomWalk):
    def calibrate(self, W, x):
        arr = ssps.view_2d_array(x.theta)
        N, d = arr.shape
        m, cov = rs.wmean_and_cov(W, arr)
        scale = 2.38 / np.sqrt(d)
        x.shared["scale_prop"] = scale * np.sqrt(np.diag(cov))

    def proposal(self, x, xprop):
        arr = ssps.view_2d_array(x.theta)
        arr_prop = ssps.view_2d_array(xprop.theta)
        z = stats.norm.rvs(size=arr.shape)
        arr_prop[:, :] = arr + z * x.shared["scale_prop"]
        return 0.0

class GaussianBridge(ssps.TemperingBridge):
    def __init__(self, sigma=0.1, dim=10):
        self.sigma = sigma
        self.dim = dim
        self.prior = dists.IndepProd(*[dists.Normal() for _ in range(dim)])
        self.log_norm_cst = dim * (np.log(sigma) + HALFLOGTWOPI)

    def logtarget(self, theta):
        return np.sum(stats.norm.logpdf(theta, scale=self.sigma), axis=1)
        # return (-(0.5/self.sigma**2) * np.sum(theta**2, axis=1) 
        #         - self.log_norm_cst)

dims = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650, 700,
        750, 800]
nruns = 3
N = 400

fks = {d: BetterTempering(model=GaussianBridge(dim=d), 
                          move=ssps.MCMCSequenceWF(mcmc=ArrayDiagRandomWalk(),
                                                   len_chain=4 * d)) 
       for d in dims}

results = particles.multiSMC(fk=fks, nruns=nruns, N=N, verbose=True,
                             joblib_verbose=50)

## PLOTS
#######
plt.style.use('ggplot')

plt.figure()
plt.scatter([r['fk'] for r in results],
            [r['output'].t for r in results])
plt.xlim(left=1)
plt.ylim(bottom=0)
plt.xlabel('dim')
plt.ylabel('nr tempering steps')
plt.savefig('gaussian_nr_tempering_steps_vs_dim.pdf')

plt.figure()
rmax = [r for r in results if r['fk'] == max(dims)][0]
plt.plot(rmax['output'].X.shared['exponents'])
plt.xlabel('iteration t')
plt.ylabel('tempering exponent')
plt.savefig('gaussian_temperatures.pdf')
