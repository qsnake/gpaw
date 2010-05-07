.. _todo:

=====
To do
=====


Important setup problems to fix
===============================

* Improve Ru setup.  Problem with nitrogen adsorption energy on
  Ru(0001) surface: Improved f-scattering, do we need 4p semi-core
  states?
* Improve Mn setup.  Problem with strange states in the band-gap for
  AFM-II MnO.
* Fine-tune Fourier filtering of projector functions.  There is still
  room for improvements.


Ideas for new features
======================

* Implement Hybrid functionals for systems with **k**-point sampling.
* Add possibility to use a plane wave basis.
* Calculate stress tensor.
* Linear scaling for LCAO or FD mode.
* Implement simple tight-binding mode.


Documentation
=============

* Major overhaul of web-page.
* Improve documentation for developers.


Other stuff
===========

* Refactor IO code and restart code.
* Optimize IO for calculations running on 1000+ cores.
* Switch from SVN to some DVCS.  This should make it easier for
  everyone to work with the code.
* Linking to BLAS library from both GPAW and NumPy can cause problems
  on some architectures and for some BLAS libraries.


