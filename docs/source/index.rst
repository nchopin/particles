
Welcome to **particles**' documentation!
========================================

This package was developed to complement the book: 

`An introduction to Sequential Monte Carlo <https://www.springer.com/gp/book/9783030478445>`_

by Nicolas Chopin and Omiros Papaspiliopoulos.

The scripts used to perform the numerical experiments discussed in the book may
be found in folder ``book``.

The documentation refers sometimes to the book for some theoretical details,
but otherwise is meant to be self-sufficient. 

To get an overview of what **particles** can do, we strongly recommend that you
have a look at the **notebook tutorials** first (in the `overview` section).  

Finally, most things are self-documented in particles with docstrings; this
means you may access the documentation of any object using the Python ``help``
command::

    from particles import resampling as rs
    help(rs)  # help on module resampling 
    help(rs.multinomial)  # help on a specific function in that module 

.. toctree::
   :maxdepth: 2

   overview
   installation    
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

