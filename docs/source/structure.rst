General structure
=================

Folders
-------

The package contains the following noteworthy folders:

* ``particles/``: contains the modules of the package (in case you want to read
  the source, or edit it in editable mode, see `installation`).

* ``book/``: scripts to generate the plots in the book; each sub-folder corresponds
  to a different chapter (e.g. ``smoothing``). 

* ``papers/``: scripts to reproduce numerical experiments from a few relevant
  papers (i.e. papers which describe algorithms that are now implemented in
  ``particles``). Each sub-folder corresponds to a different paper, and contains a
  ``README.md`` file that describe briefly the experiments and give the full
  reference of the paper. At the moment, these papers are:

  + waste-free SMC (Dau & Chopin, 2022);
  + nested sampling SMC (Salomone et al, 2018);
  + SMC for large binary spaces (Schäfer & Chopin, 2014);
  + on backward smoothing algorithms (Dau & Chopin, 2023).

* ``docs/``: documentation (managed by ``sphinx``). The jupyter notebooks are in
  ``docs/source/notebooks``. 


API reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   particles
