.. _todo:

=====
To do
=====

This page lists what we would like to do when time permits.  In each section, items are ordered according to priority.

New features and long-term goals
================================

* Order-*N* mode with localized orbitals.
* Exact exchange.
* Calculation of stress-tensor.
* Hexagonal unit cells.
* Optimized effective potential.


Improved convergence (fewer iterations)
=======================================

* Use Kerker mixing.
* Alternative to RMM-DIIS.


Documetation
============

* Write more tutorials: bulk-modulus/lattice constant, MD, STM, Wannier functions, OpenDX, ...


Small improvements
==================

* When calculating matrix elements for maximally localized Wannier functions, add PAW correction.


Setups
======

* Find the optimal vbar.


Testing
=======

* Write a "long time MD run"-test.
* Run with ``-tt`` and ``-Qwarnall`` options.


Python code
===========

* Use the config command in ``setup.py`` (``check_func, check_lib, check_header``). 
* Python 3000:  Some divisions must be changed from '/' to '//' (Use ``finddiv.py`` to fix division problem).


Optimization
============

* Optimize interpolation and restriction: xyz -> Zxy -> YZx -> XYZ or xyz -> xyZ -> Zxy -> ZxY -> YZx -> YZX -> XYZ?
* Preconditioning:  Use a neares neighbor Laplacian on the fine grid.
* Try ``zheevr`` instead of ``zhveed``.
* Compare gcc and icc.


Parallelization
===============

* MPI_IN_PLACE??
* Use syevx from scalapack.


Stuff
=====

* Use ``cmdclass={'config': myconfig}`` in distutils.
* Multi grid acceleration for Poisson equation may not be necessary if we have a good initial guess.
* Don't create projectors if they are outside the sphere: They could be inside the cube, but outside the sphere!


Or not to do?
=============

Some questions:

* How do we treat two or more variations of a setup for a single type of atom?
* Does it pay off to make a good starting guess for the initial Hartree portential?  Or is it just as good to start from a constant potential?
* How many decimals should the arrays have in the XML format?  
* Use lower or upper packed storage?
* Should the evaluation of the pair potential be parallelized?
* What is the best convergence criteria? Change in energy, wave functions, density or eigenvalues?
