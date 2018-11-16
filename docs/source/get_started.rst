Notations, naming conventions
=============================

We did our best to use the same "notations" as in the book, e.g.: 

* `N`: number of particles
* `W`: a vector of normalised weights (i.e. sum to one)
* `X`: an object representing the N current particles
* `t`: the time index (starts at 0)

In doing so, we took some liberty with the standard PEP8 naming conventions 
(see `here 
<https://www.python.org/dev/peps/pep-0008/#prescriptive-naming-conventions>`)
which states in particular that variable names should be in lower case. 

 * modules in lower case: `resampling`, `smoothing`, etc. 
 * classes in camel case: `StateSpaceModel`, `FeynmanKac`, etc. 
 * variables in lower case. 

 However, we break these conventions whenever 

 * W: the normalised weights (
 * X: the N particles 

adheres to python 

Time objects, particle objects
==============================

Objects that record information at every iteration (time) of the algorithms are
lists, or list-like objects (objects that behave like lists); e.g. a `data`
object should be such that `data[t]` represents data-point number t.  

Objects that represent the N particles gg


