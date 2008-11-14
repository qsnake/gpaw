.. _releasenotes:

=============
Release notes
=============

(The development version in trunk is now the version that will once become 0.5)


Version 0.4
===========

13 November 2008: :svn:`tags/0.4 <../tags/0.4>`.


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

19 December 2007: :svn:`tags/0.3 <../tags/0.3>`.
