from math import log
import numpy as np
from numpy import random
import torch

import particles
from particles import resampling as rs

cuda = torch.device('cuda:0')

def exp_and_normalise(lw):
    return torch.exp(lw - lw.max())

class TorchWeights(rs.Weights):
    def __init__(self, lw=None):
        self.lw = lw
        if lw is not None:
            self.lw[torch.isnan(self.lw)] = float('-inf')
            m = self.lw.max()
            w = torch.exp(lw - m)
            sw = float(w.sum())
            self.W = w / sw
            self.log_mean = m + log(sw) - log(len(lw))
            self.ESS = float(1. / torch.sum(self.W ** 2))

    @staticmethod
    def arange(N):
        return torch.arange(N, device=cuda)

    def resample(self, scheme='multinomial', M=None):
        if scheme != 'multinomial':
            raise ValueError('torch: only multinomial resampling available')
        if M is None:
            M = len(self.lw)  
        #  TODO what if self.lw is None
        return torch.multinomial(self.W, M, replacement=True)

class TorchFeynmanKac(particles.FeynmanKac):
    weights_cls = TorchWeights

class ToyGPUFK(TorchFeynmanKac):
    def __init__(self, T=10, d=10):
        self.T = T
        self.d = d
    def M0(self, N):
        return torch.randn(N, self.d, device=cuda)
    def M(self, t, xp):
        return 0.9 * xp + torch.randn(N, self.d, device=cuda)
    def logG(self, t, xp, x):
        return - 10. * torch.sum(x**2, axis=1)

class ToyFK(particles.FeynmanKac):
    def __init__(self, T=10, d=10):
        self.T = T
        self.d = d
    def M0(self, N):
        return random.randn(N, self.d)
    def M(self, t, xp):
        return 0.9 * xp + random.randn(N, self.d)
    def logG(self, t, xp, x):
        return - 10 * np.sum(x**2, axis=1)

if __name__ == '__main__':
    T = 1000
    N = 10**5
    fk = ToyFK(T=T)
    pf = particles.SMC(fk=fk, N=N, resampling='multinomial')
    gpu_fk = ToyGPUFK(T=T)
    gpu_pf = particles.SMC(fk=gpu_fk, N=N, resampling='multinomial')


# TODO
#
# * APF
# * summaries should *not* be tensors
