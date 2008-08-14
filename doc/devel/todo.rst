.. _todo:

=====
To do
=====

This page lists what we would like to do when time permits.  In each section, items are ordered according to priority.


Immediate documentation work
============================

To do:

* Write TOC-like documentation page (documentation/documentation) properly : Marco
* Write intro on tut-page : Marco
* Reorder dev-page : ask
* add more links from the exercises to the corresponding ASE-documentation.
* search for XXX and ASE imports, variable names loa : ask, marco
* put cross references from gpaw to ase where proper : marco
* siesta exercises on ase pages  : marco
* Fix TST-exercise : jj

* fix the global modules page and .. autoclass : not important right now
* Take care of orphan entries listed in the main TOC ("Talk about this later") : ?

Done:

* Fix frontpage : ask
* Sidebar: add svn, ase, trac : ask
* finish up ASE quick-overview and text on tutorials : marco
* STM-docs : marco
* use the ..class::QuasiNewton somewhere to get the links : marco
* fix background of gpaw logo - transparent vs green  :  ask
* fix math role in ase, copy to ase svn : ask
* latex-builder has trouble with image sizes : jj
* find out how it should work with document structure/TOC and so on
* povray-png's look black in pdf : JJ
* smaller abinit logo : Ask
* main page:  make TOC-like list in side bar : ask
* convert wiki:API: to :epydoc: : ask
* Also search for wiki:SVN: in rst-files : ask
* put contents and search on one line : ask
* top navigation bar should be moved to sidebar and combined with main page links to resemble TOC-like thing on the ase page : ask
* Add link to https://wiki.fysik.dtu.dk/gpaw/epydoc/ somewhere : Actually there's already one on the devel page, *and* on the documentation page


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


Documentation
=============

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
