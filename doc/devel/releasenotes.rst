.. _releasenotes:

=============
Release notes
=============


Version 0.8 (development version in trunk)
==========================================

:trac:`trunk <>`.

* Add important changes here.
* ...


Version 0.7.2
=============

13 August 2010: :trac:`tags/0.7.2 <../tags/0.7.2>`.

* For version 0.7, the default Poisson solver was changed to
  ``PoissonSolver(nn=3)``.  Now, also the Poisson solver's default
  value for ``nn`` has been changed from ``'M'`` to ``3``.


Version 0.7
===========

23 April 2010: :trac:`tags/0.7 <../tags/0.7>`.

* Better and much more efficient handling of non-orthorhombic unit
  cells.  It may actually work now!
* Much better use of ScaLAPACK and BLACS.  All large matrices can now
  be distributed.
* New test coverage pages for all files: :ref:`coverage`.
* New default value for Poisson solver stencil: ``PoissonSolver(nn=3)``.
* Much improved MPI module (:ref:`communicators`).
* Self-consistent Meta GGA.
* New :ref:`PAW setup tar-file <setups>` now contains revPBE setups and
  also dzp basis functions.
* New ``$HOME/.gpaw/rc.py`` configuration file.
* License is now GPLv3+.
* New HDF IO-format.
* :ref:`Advanced GPAW Test System <big-test>` Introduced.


Version 0.6
===========

9 October 2009: :trac:`tags/0.6 <../tags/0.6>`.

* Much improved default parameters.
* Using higher order finite-difference stencil for kinetic energy.
* Many many other improvements like: better parallelization, fewer bugs and
  smaller memory footprint.


Version 0.5
===========

1 April 2009: :trac:`tags/0.5 <../tags/0.5>`.

* `new setups added Bi, Br, I, In, Os, Sc, Te; changed Rb setup <https://trac.fysik.dtu.dk/projects/gpaw/changeset/3612>`_.
* `memory estimate feature is back <https://trac.fysik.dtu.dk/projects/gpaw/changeset/3575>`_

Version 0.4
===========

13 November 2008: :trac:`tags/0.4 <../tags/0.4>`.


* Now using ASE-3 and numpy.
* TPSS non self-consistent implementation.
* LCAO mode.
* VdW-functional now coded in C.
* Added atomic orbital basis generation scripts.
* Added an Overlap object, and moved apply_overlap and apply_hamiltonian
  from Kpoint to Overlap and Hamiltonian classes.

* Wannier code much improved.
* Experimental LDA+U code added.
* Now using libxc.
* Many more setups.
* Delta scf calculations.

* Using localized functions will now no longer use MPI group
  communicators and blocking calls to MPI_Reduce and MPI_Bcast.
  Instead non-blocking sends/receives/waits are used.  This will
  reduce syncronization time for large parallel calculations.

* More work on LB94.
* Using LCAO code forinitial guess for grid calculations.
* TDDFT.
* Moved documentation to Sphinx.
* Improved metric for Pulay mixing.
* Porting and optimization for BlueGene/P.
* Experimental Hartwigsen-Goedecker-Hutter pseudopotentials added.
* Transport calculations with LCAO.


Version 0.3
===========

19 December 2007: :trac:`tags/0.3 <../tags/0.3>`.
