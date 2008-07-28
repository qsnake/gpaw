The ``Paw`` object defined in ``paw.py`` is the central object for a
calculation.  Instantiating such an object by hand is not recommended.
Use the ``create_paw_object()`` helper-function instead (it will
supply many default values) - it is used py the ASE_ Calculator
interface.

.. _ASE: https://web2.fysik.dtu.dk/ase/

These are the most important attributes of a ``Paw`` object:
 =============== =================================================
 ``domain``      :ref:`domain`
 ``setups``      Dictionary mapping chemical symbols to Setups
 ``symmetry``    Symmetry object
 ``timer``       Timer
 ``wf``          WaveFunctions object
 ``xc``          XCOperator object
 ``xcfunc``      XCFunctional
 ``nuclei``      List of :ref:`nucleus` objects
 ``out``         Output stream for text
 ``pairpot``     PairPotential object
 ``poisson``     PoissonSolver
 ``gd``          :ref:`grid_descriptor` for coarse grid
 ``finegd``      :ref:`grid_descriptor` for fine grid
 ``restrict``    Function for restricting the effective potential
 ``interpolate`` Function for interpolating the electron density
 ``mixer``       DensityMixer
 =============== =================================================

Energy contributions and forces:
 =========== ================================
 ``Ekin``    Kinetic energy
 ``Epot``    Potential energy
 ``Etot``    Total energy
 ``Etotold`` Total energy from last iteration
 ``Exc``     Exchange-Correlation energy
 ``S``       Entropy
 ``Ebar``    Should be close to zero!
 ``F_ai``    Forces
 =========== ================================


The attributes ``tolerance``, ``fixdensity``, ``idiotproof`` and
``usesymm`` have the same meaning as the corresponding Calculator
keywords (see the :ref:`manual`).  Internal units are Hartree and Å and
``Ha`` and ``a0`` are the conversion factors to external ASE units.
``error`` is the error in the Kohn-Sham wave functions - should be
zero (or small) for a converged calculation.

Booleans describing the current state:
 ============= ======================================
 ``forces_ok`` Have the forces bee calculated yet?
 ``converged`` Do we have a self-consistent solution?
 ============= ======================================

Number of iterations for:
 ============ ===============================
 ``nfermi``   finding the Fermi-level
 ``niter``    solving the Kohn-Sham equations
 ``npoisson`` Solving the Poisson equation
 ============ ===============================

Soft and smooth pseudo functions on uniform 3D grids:
 ========== =========================================
 ``nt_sG``  Electron density on the coarse grid.
 ``nt_sg``  Electron density on the fine grid.
 ``rhot_g`` Charge density on the coarse grid.
 ``nct_G``  Core electron-density on the coarse grid.
 ``vHt_g``  Hartree potential on the fine grid.
 ``vt_sG``  Effective potential on the coarse grid.
 ``vt_sg``  Effective potential on the fine grid.
 ========== =========================================

Only attribute not mentioned now is ``nspins`` (number of spins) and
those used for parallelization:

 ================== =================================================== 
 ``my_nuclei``      List of :ref:`nucleus` objects that have their
                    center in this domain.
 ``p_nuclei``       List of :ref:`nucleus` objects with projector functions
                    overlapping this domain.
 ``g_nuclei``       List of :ref:`nucleus` objects with compensation charges
                    overlapping this domain.
 ``locfuncbcaster`` LocFuncBroadcaster object for parallelizing 
                    evaluation of localized functions (used when
                    parallelizing over **k**-points).
 ================== ===================================================
