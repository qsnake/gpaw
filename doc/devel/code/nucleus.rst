.. _nucleus:

=======
Nucleus
=======

The ``Nucleus`` object basically consists of a setup, a scaled
position and some localized functions.  It takes care of adding
localized functions to functions on extended grids and calculating
integrals of functions on extended grids and localized functions.

 ============= ========================================================
 ``setup``     Setup object.
 ``spos_i``    Scaled position.
 ``a``         Index number for this nucleus.
 ``typecode``  Data type of wave functions (``Float`` or ``Complex``).
 ``neighbors`` List of overlapping neighbor nuclei.
 ``onohirose`` Number of grid points used for Ono-Hirose interpolation
               - currently 1, 2, 3, and 5 is implemented (1 means no
               interpolation).
 ============= ========================================================

Localized functions:
 ========== ===========================================================
 ``nct``    Pseudo core electron density.
 ``ghat_L`` Shape functions for compensation charges.
 ``vhat_L`` Correction potentials for overlapping compensation charges.
 ``pt_i``   Projector functions.
 ``vbar``   Arbitrary localized potential.
 ``phit_i`` Pseudo partial waves used for initial wave function guess.
 ========== ===========================================================

Arrays:
 ========= ===============================================================
 ``P_uni`` Integral of products of all wave functions and the projector
           functions of this atom (``P_{\sigma\vec{k}ni}^a``).
 ``D_sp``  Atomic density matrix (``D_{\sigma i_1i_2}^a``).
 ``dH_sp`` Atomic Hamiltonian correction (``\Delta H_{\sigma i_1i_2}^a``).
 ``Q_L``   Multipole moments  (``Q_{\ell m}^a``).
 ``F_i``   Force.
 ========= ===============================================================


Parallel stuff: ``comm``, ``rank`` and ``domain_overlap``
