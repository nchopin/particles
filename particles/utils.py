# -*- coding: utf-8 -*-

""" 
This module gathers several non-numerical utilities. The only one of direct 
interest to the user is the `multiplexer` function, which we now describe
briefly.  

Say we have some function ``f``, which takes only keyword arguments::

    def f(x=0, y=0, z=0):
        return x + y + z**2

We wish to evaluate f repetitively for a range of x, y and/or z values.
To do so, we may use function multiplexer as follows::

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
but the output reports the corresponding keys, i.e. 

    [ {'z': 'good', 'out': 12},  # f(0, 0, 3)
      {'z': 'bad', 'out': 28}    # f(0, 0, 5)
    ]

This is useful when f takes as arguments complex objects that you would like to 
replace by more legible labels; e.g. option ` model` of class `SMC`. 

`multiplexer` also accepts three extra keyword arguments (whose name may not
therefore be used as keyword arguments for function f): 

* ``nruns`` (default=1): evaluate f *nruns* time for each combination of arguments;
  an entry 'run' (ranging from 0 to nruns-1) is added to the output dictionaries.
  This is mostly useful when the output of ``f`` is random.
* ``seeding``:  if True, generate random numbers that are all distinct, and use
  them as input for keyword argument `seed` of function. Again, this is useful if 
  f returns a random output, and if in addition, it does take as a kw argument 
  a seed for the random generator. 
* ``nprocs``: if >0, number of CPU cores to use in parallel; if <=0, number
  of cores *not* to use; in particular, ``nprocs=0`` means all CPU cores must 
  be used. 

.. warning :: 
    Option `nprocs` rely on the standard library `multiprocessing`, 
    whose performance and behaviour seems to be OS-dependent. In particular, 
    it may not work well on Windows. 

.. seealso :: `multiSMC`

"""

from __future__ import division, print_function

import copy
import functools
import itertools
import multiprocessing
from numpy import random
import time

def timer(method):
    @functools.wraps(method)
    def timed_method(self, **kwargs):
        starting_time = time.clock()
        out = method(self, **kwargs)
        self.cpu_time = time.clock() - starting_time
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
    return [{k: v for k, v in zip(d.keys(), args)}
            for args in itertools.product(*d.values())]


def cartesian_args(args, listargs, dictargs):
    """ Compute a list of inputs and outputs for a function
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
    ils = {k: [v, ] for k, v in args.items()}
    ils.update(listargs)
    ils.update({k: v.values() for k, v in dictargs.items()})
    ols = listargs.copy()
    ols.update({k: v.keys() for k, v in dictargs.items()})
    return cartesian_lists(ils), cartesian_lists(ols)


def add_to_dict(d, obj, key='output'):
    if isinstance(obj, dict):
        d.update(obj)
    else:
        d[key] = obj
    return d


def worker(qin, qout, f):
    """Worker for muliprocessing.
    
    A worker repeatedly picks a dict of arguments in the queue and computes 
    f for this set of arguments, until the input queue is empty.
    """
    while not qin.empty():
        i, args = qin.get()
        qout.put((i, f(**args)))


def distribute_work(f, inputs, outputs=None, nprocs=1, out_key='output'):
    """
    For each input i (a dict) in list **inputs**, evaluate f(**i)
    using multiprocessing if nprocs>1

    The result has the same format as the inputs: a list of dicts,
    taken from outputs, and updated with f(**i).
    If outputs is None, it is set to inputs.
    """
    if outputs is None:
        outputs = [ip.copy() for ip in inputs]
    if nprocs <= 0:
        nprocs += multiprocessing.cpu_count()

    # no multiprocessing
    if nprocs <= 1:
        return [add_to_dict(op, f(**ip), key=out_key)
                for ip, op in zip(inputs, outputs)]

    # multiprocessing
    queue_in = multiprocessing.Queue()
    queue_out = multiprocessing.Queue()
    procs = [multiprocessing.Process(target=worker,
                                     args=(queue_in, queue_out, f))
             for _ in range(nprocs)]
    sent = [queue_in.put((i, args)) for i, args in enumerate(inputs)]
    [p.start() for p in procs]
    results = [queue_out.get() for _ in sent]
    for i, r in results:
        add_to_dict(outputs[i], r)
    [p.join() for p in procs]

    return outputs


def distinct_seeds(k):
    """ returns k distinct seeds for random number generation
    """
    seeds = []
    for _ in range(k):
        while True:
            s = random.randint(2**32 - 1)
            if s not in seeds:
                break
        seeds.append(s)
    return seeds


def multiplexer(f=None, nruns=1, nprocs=1, seeding=None, **args):
    """Evaluate a function for different parameters, optionally in parallel.

    Parameters
    ----------
    f: function 
        function f to evaluate, must take only kw arguments as inputs
    nruns: int 
        number of evaluations of f for each set of arguments
    nprocs: int
        + if <=0, set to actual number of physical processors plus nprocs
        (i.e. -1 => number of cpus on your machine minus one)
        Default is 1, which means no multiprocessing
    seeding: bool (default: True if nruns > 1, False otherwise)
        whether we need to provide different seeds for RNGS
    **args:
        keyword arguments for function f.

    Note
    ----
    see documentation of `utils`

    """
    if not callable(f):
        raise ValueError('multiplexer: function f missing, or not callable')
    if seeding is None:
        seeding = (nruns > 1)
    # extra arguments (meant to be arguments for f)
    fixedargs, listargs, dictargs = {}, {}, {}
    listargs['run'] = list(range(nruns))
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
        ip.pop('run')  # run is not an argument of f, just an id for output
    # distributing different seeds
    if seeding:
        seeds = distinct_seeds(len(inputs))
        for ip, op, s in zip(inputs, outputs, seeds):
            ip['seed'] = s
            op['seed'] = s
    # the actual work happens here
    return distribute_work(f, inputs, outputs, nprocs=nprocs)
