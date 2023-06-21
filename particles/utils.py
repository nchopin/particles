# -*- coding: utf-8 -*-

"""
Non-numerical utilities (mostly for parallel computation).

Overview
========

This module gathers several non-numerical utilities, mostly related to parallel
computation. The most useful such utility, `multiplexer` is described in the
next section. More low-level functions are described briefly in a following
section. 

Multiplexer
===========

Say we have some function ``f``, which takes only keyword arguments::

    def f(x=0, y=0, z=0):
        return x + y + z**2

We wish to evaluate f repetitively for a range of x, y and/or z values,
optionally in parallel (using several CPU cores). To do so, we may use 
function `multiplexer` as follows::

    results = multiplexer(f=f, x=3, y=[2, 4, 6], z=[3, 5])

which returns a list of 3*2 dictionaries of the form::

    [ {'x':3, 'y':2, 'z':3, 'out':14},  # 14=f(3, 2, 3)
      {'x':3, 'y':2, 'z':5, 'out':30},
      {'x':3, 'y':4, 'z':3, 'out':16},
       ... ]

In other words, `multiplexer` computes the **Cartesian product** of the inputs.

For each argument, you may use a dictionary instead of a list::

    results = multiplexer(f=f, z={'good': 3, 'bad': 5})

In that case, the values of the dictionaries are used in the same way as above,
but the output reports the corresponding keys, i.e.::

    [ {'z': 'good', 'out': 12},  # f(0, 0, 3)
      {'z': 'bad', 'out': 28}    # f(0, 0, 5)
    ]

This is useful when f takes as arguments complex objects that you would like to
replace by more legible labels; e.g. argument `fk` for class `SMC`.

`multiplexer` also accepts three extra keyword arguments (whose name may not
therefore be used as keyword arguments for function f):

* ``nprocs`` (default=1): if >0, number of CPU cores to use in parallel; if
  <=0, number of cores *not* to use; in particular, ``nprocs=0`` means all CPU
  cores must be used.
* ``nruns`` (default=1): evaluate f *nruns* time for each combination of arguments;
  an entry `run` (ranging from 0 to nruns-1) is added to the output dictionaries.
* ``seeding`` (default: True if ``nruns``>1, False otherwise):  if True, seeds
  the pseudo-random generator before each call of function `f` with a different
  seed; see below.

.. warning ::
    Parallel computation relies on library `joblib`. This library generates
    identical workers, up to the state of the Numpy random generator. If your
    function involves random numbers, make sure that (a) option ``seeding`` is
    set to True (otherwise, you will get identical results from workers with
    the same inputs); (b) that the function f does not rely on scipy frozen
    distributions, as these distributions also freeze the states. For instance,
    do not use any frozen distribution when defining your own Feynman-Kac
    object.

.. seealso :: `multiSMC`

Low-level utilities
===================

These low-level utilities might be useful to more advanced users in certain
specific scenarios.

* `distribute_work`: evaluate the same function for different inputs,
  optionally in parallel. (Useful if multiplexer is not flexible enough for
  you, i.e. you want something more general that a cartesian product for the
  inputs).

* `distinct_seeds`: generate (randomly) pairwise distinct RNG seeds.

* `timer`: a decorator to measure the CPU time of a function call.
"""

from __future__ import division, print_function

import functools
import itertools
import time

import joblib
import numpy as np
from numpy import random

MAX_INT_32 = np.iinfo(np.uint32).max


def timer(method):
    @functools.wraps(method)
    def timed_method(self, **kwargs):
        starting_time = time.perf_counter()
        out = method(self, **kwargs)
        self.cpu_time = time.perf_counter() - starting_time
        return out

    return timed_method


def cartesian_lists(d):
    """
    turns a dict of lists into a list of dicts that represents
    the cartesian product of the initial lists

    Example
    -------
    cartesian_lists({'a':[0, 2], 'b':[3, 4, 5]}
    returns
    [ {'a':0, 'b':3}, {'a':0, 'b':4}, ... {'a':2, 'b':5} ]

    """
    return [
        {k: v for k, v in zip(d.keys(), args)}
        for args in itertools.product(*d.values())
    ]


def cartesian_args(args, listargs, dictargs):
    """Compute a list of inputs and outputs for a function
    with kw arguments.

    args: dict
      fixed arguments, e.g. {'x': 3}, then x=3 for all inputs
    listargs: dict
        arguments specified as a list; then the inputs
        should be the Cartesian products of these lists
    dictargs: dict
        same as above, except the key will be used in the output
        (see module doc for more explanation)

    """
    ils = { k: [ v, ] for k, v in args.items() }
    ils.update(listargs)
    ils.update({k: v.values() for k, v in dictargs.items()})
    ols = listargs.copy()
    ols.update({k: v.keys() for k, v in dictargs.items()})
    return cartesian_lists(ils), cartesian_lists(ols)


def add_to_dict(d, obj, key="output"):
    if isinstance(obj, dict):
        d.update(obj)
    else:
        d[key] = obj
    return d


def worker(qin, qout, f):
    """Worker for multi-processing.

    A worker repeatedly picks a dict of arguments in the queue and computes
    f for this set of arguments, until the input queue is empty.
    """
    while True:
        i, args = qin.get()
        if i is None and args is None:
            break
        qout.put((i, f(**args)))


def distribute_work(f, inputs, outputs=None, nprocs=1, out_key="output",
                    joblib_verbose=0):
    """
    Evaluate the same function for different inputs, optionally in parallel.

    Parameters
    ----------
    f: callable (takes only keyword arguments)
        the function to evaluate 
    inputs: list of dicts
        list of input arguments for each function call
    outputs: list of dicts (default: set to inputs)
        initialise the list of outputs (one per call)
    nprocs: int (default: 1)
        if positive, number of processors is set to this value;
        if not, number of processors is set to max number + nprocs
        (e.g. nprocs=0 means all the available processors should be used)
    out_key: string
        name of the key for the output of each function call

    Returns
    -------
    outputs: list of dicts

    Note
    ----
    outputs if first initialised by argument `outputs`. Then each dict in this
    this list is updated as follows when the corresponding function call is
    performed:

    * if f returns a dict, then each (key, value) pair in that dict is added to
      to the output dict. 

    * if f returns an object which is not a dict, then a single (key, value) 
      pair is added, with key given by argument `out_key`, and value the output 
      of f.

    """
    if outputs is None:
        outputs = [ip.copy() for ip in inputs]
    if nprocs <= 0:
        nprocs += joblib.cpu_count()

    # no multiprocessing
    if nprocs <= 1:
        return [add_to_dict(op, f(**ip), key=out_key) 
                for ip, op in zip(inputs, outputs)]

    delayed_f = joblib.delayed(f)

    # multiprocessing
    pool = joblib.Parallel(n_jobs=nprocs, backend="loky",
                           verbose=joblib_verbose)
    results = pool(delayed_f(**ip) for ip in inputs)
    for i, r in enumerate(results):
        add_to_dict(outputs[i], r)

    return outputs


def distinct_seeds(k):
    """generates distinct seeds for random number generation.

    Parameters
    ----------
    k:  int
        number of requested seeds

    Note
    ----
    uses stratified sampling to make sure the seeds are distinct.
    """
    bw = MAX_INT_32 // k  # bin width
    return np.arange(0, k * bw, bw) + random.randint(bw, size=k)


class seeder(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed:
            random.seed(seed)
        return self.func(**kwargs)


def multiplexer(f=None, nruns=1, nprocs=1, seeding=None, protected_args=None, 
                joblib_verbose=0, **args):
    """Evaluate a function for different parameters, optionally in parallel.

    Parameters
    ----------
    f: function
        function f to evaluate, must take only kw arguments as inputs
    nruns: int
        number of evaluations of f for each set of arguments
    nprocs: int (default: 1)
        number of processors to use. If nprocs <=0, number of processors
        is set to nprocs plus number of available processors; e.g. nprocs=0
        means use all the cores.
        Default is 1, which means no multiprocessing
    seeding: bool (default: True if nruns > 1, False otherwise)
        whether to seed the pseudo-random generator (with distinct
        seeds) before each evaluation of function f.
    protected_args: dict
        args protected from cartesian product (even if they are lists)
    **args:
        keyword arguments for function f.

    Note
    ----
    see documentation of `utils` (especially regarding ``seeding``).

    """
    if not callable(f):
        raise ValueError("multiplexer: function f missing, or not callable")
    # extra arguments (meant to be arguments for f)
    fixedargs = {} if protected_args is None else protected_args
    listargs, dictargs = {}, {}
    listargs["run"] = list(range(nruns))
    for k, v in args.items():
        if isinstance(v, list):
            listargs[k] = v
        elif isinstance(v, dict):
            dictargs[k] = v
        else:
            fixedargs[k] = v
    # cartesian product
    inputs, outputs = cartesian_args(fixedargs, listargs, dictargs)
    for ip in inputs:
        ip.pop("run")  # run is not an argument of f, just an id for output
    # distributing different seeds
    if seeding is None:
        seeding = nruns > 1
    if seeding:
        seeds = distinct_seeds(len(inputs))
        f = seeder(f)
        for ip, op, seed in zip(inputs, outputs, seeds):
            ip["seed"] = seed
            op["seed"] = seed
    # the actual work happens here
    return distribute_work(f, inputs, outputs=outputs, nprocs=nprocs,
                           joblib_verbose=joblib_verbose)
