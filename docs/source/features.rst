Features
********

Here is a brief list of the features of particles: 

* state-space models may be defined as python objects, in a basic form of
  probabilistic programming. 
* Bootstrap filter, guided filter, auxiliary particle filter. 
* exact filtering/smoothing algorithms: Kalman (linear Gaussian models), 
  and forward-backward (finite hidden Markov models). 
* Several resampling schemes are implemented. 
* Sequential quasi-Monte Carlo (and related tools: Hilbert ordering, RQMC
  sampling). 
* Smoothing: on-line and off-line, O(N^2) and O(N) versions of standard
  algorithms (FFBS, two-filter).
* SMC samplers: IBIS (data-tempering) and SMC tempering. Static models may 
  be defined as Python objects. 
* Bayesian inference for state-space models: several PMCMC (particle MCMC
  algorithms are implemented), such as PMMH and Particle Gibbs. Also SMC^2. 
* Genealogy-based variance estimators (Chan & Lai, 2013; Lee & Whiteley,
  2018; Olsson & Douc, 2019). 
* A Pima indian example is included. 


