.. _convergence:

.. default-role:: math


==================
Convergence Issues
==================

Here you find a list of suggestions that should be considered when
encountering convergence problems:

* Try to use default parameters for the calculator. Simple but
  sometimes useful!
* Remember that for Gamma Point calculations the :ref:`Fermi
  temperature <manual_width>` is set to zero by default, even if you
  use periodic boundary conditions. In the case of metallic systems
  you might want to specify a finite Fermi temperature.
* If you are specifying the :ref:`number of bands <manual_nbands>`
  manually, try to increase the number of empty states. You might also
  let gpaw choose the default number, which is in general large
  enough.
* Try to use a less aggressive :ref:`mixing <manual_mixer>`, i.e. a
  smaller mixing parameter. For example a typical mixer for a metallic
  system may be ``mixer=Mixer(0.1, 5, metric='new', weight=100.0)``.
  For spin polarised systems you can use either ``mixer=MixerSum()`` or
  ``mixer=MixerDif()`` using the same options as with ``mixer=Mixer()``.
* The initial guess for the electron density is always calculated
  using the LCAO scheme, with a default single-zeta basis, i.e. one
  orbital for each valence electron. You can try to make a better
  initial guess by enlarging the :ref:`manual_basis`. This can be done
  by setting ``basis='szp'`` if you want to use a
  single-zeta-polarized basis. Note that you first need to generate
  the basis file, as described in :ref:`LCAO mode <lcao>`.
* Try changing the :ref:`eigensolver <manual_eigensolver>`. The
  default, ``RMM-DIIS``, is good for speed, but can sometimes have
  poor convergence properties. The conjugate gradient, ``CG``, solver
  might be more stable.
