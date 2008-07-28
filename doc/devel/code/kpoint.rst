Defined in file ``kpoint.py``.

 ============ =================================================================
 ``gd``       Descriptor for wave-function grid.
 ``weight``   Weight of **k**-point.
 ``typecode`` Data type of wave functions (``Float`` or ``Complex``).
 ``timer``    Timer_ object.
 ``nbands``   Number of bands.
 ``s``        Spin index: up or down (0 or 1).
 ``k``        **k**-point index.
 ``u``        Combined spin/**k**-point index: ``u=????``.
 ============ =================================================================       

Arrays:
 ============= ================================================================
 ``phase_id``  Bloch phase-factors for translations - axis ``i=0,1,2``
               and direction ``d=0,1``.
 ``k_i``       **k**-point vector (coordinates scaled to [-0.5:0.5] interval).
 ``eps_n``     Eigenvalues.
 ``f_n``       Occupation numbers.
 ``H_nn``      Hamiltonian matrix.
 ``S_nn``      Overlap matrix.
 ``psit_nG``   Wave functions.
 ``Htpsit_nG`` Pseudo-part of the Hamiltonian applied to the wave functions.
 ============= ================================================================

Parallel stuff:
 ======== =================================================================
 ``comm`` MPI-communicator for parallelization over **k**-points.
 ``root`` Rank of the CPU that does the matrix diagonalization of ``H_nn``
          and the Cholesky decomposition of ``S_nn``.
 ======== =================================================================

.. _Timer: https://wiki.fysik.dtu.dk/stuff/html/public/gridpaw.utilities.timing.Timer-class.html
.. _LocFuncBroadcaster: https://wiki.fysik.dtu.dk/stuff/html/public/gridpaw.localized_functions.LocFuncBroadcaster-class.html
