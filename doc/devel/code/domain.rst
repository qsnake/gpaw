.. _domain:

======
Domain
======

A ``Domain`` object (in ``domain.py``) holds informaion on the unit cell and the boundary conditions:

 ============== ===============================================
 ``cell_i``     Array containing the lengths of the three axes.
 ``periodic_i`` Periodic boundary conditions or not?  A tuple
                of three ``bool``'s.
 ``angle``      Rotation angle applied to the unit cell after
                translation of one lattice vector in the
                *x*-direction (experimental feature).
 ============== ===============================================

Parallel stuff: ``displacement_idi``, ``comm``, ``neighbor_id``,
``parpos_i``, ``parsize_i`` and ``stride_i``.
 
