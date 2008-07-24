.. _releasenotes:

=============
Release notes
=============

* Current-SVN

  - Fixed bug for ``lmax > 0``!
  - Much improved setups.
  - Ono-Hirose filtering replaced by Fourier filtering.
  - Writing of state to netCDF and tar-file formats.
  - Perturbative exact exchange calculation are now possible for molecules.
  - Introduced a ``GRIDPAWSETUPPATH`` environment variable to lacate setup-files.
  - New Gauss-Seidel solver for the Poisson equation.
  - The real spherical harmonics are now computer generated.
  - We can now have f-type projector functions.
  - Better initial guess for transition metals.

* Version 0.9.0

  - Dropped numarray!  This gives a slight `speedup <NumarrayToNumeric>`_.
  - Renamed ``gridpaw-slave`` to the more correct name ``gridpaw-mpi``.
  - ``paw.py`` has been split into four files: ``startup.py``, ``paw.py``,
    ``wf.py`` and ``netcdf.py``.
  - New keywords added: ``softgauss``, ``onohirose`` and ``order``.
  - Initial preparations for rotation-translation symmetry.
  - Lots of cleaning up!
  - Code has move to a SVN server at berlios.de.

* Version 0.8.3

  - Changed default Fermi temperature to 0.1 eV.
  - Compiles on SunOS.
  - moved python code from ``python`` directory to ``gridpaw`` directory.

* Version 0.8.2

  - Fixed some bugs in the serial version.
  - Fixed some Python 2.2 bugs.
  - Rank and size of a communicator are now attributes - ``get_rank``
    and ``get_size`` methods are gone.

* Version 0.8.1

  - Parallellization over spin and **k**-points added.
  - Interpolation of the density and restriction of the potential and
    the radial functions done to :math:`O(h^6)` - gives much
    improved convergence with respect to *h*.
  - Imporoved test-suite (no failing tests!).
  - Parallellized calculation of localized functions - was not
    parallel when parallellization over **k**-points was in use.

* Version 0.8.0

  - Rewrote setup-generator.  It will now find ghost-states and plot
    logarithmic derivatives.
  - Projectors can now be listed in the setups in any order.
  - Got rid of ``Spin.py``.
  - Major structural changes. Moved code to ``gridpaw``.
  - New basic multi-grid subroutine (BMGS) library added.
  - Changed C++ code to C99 code.

* Version 0.7.2

  - Many small fixes!

* Version 0.7.1

  - Using a polynomial for the localized compensation charges.
  - Fixed a bug in the generation of scalar-realtivistic setups.

* Version 0.7.0

  - The output now has a little ascii-art plot of the atoms.
  - Added two ``Calculator``-methods:
    
    + ``SetNumberOfProjectors`` and
    + ``SetSetupName``

  - Generation of scalar-relativistiv paw-setups is now possible.
  - Output from a parallel calculation is now flushed all the way!
  - The default value for the Fermi temperature is now zero Kelvin.
  - Added a ``Center`` function.  Useful for calculations without
    periodic boundary conditions, where the molecule should be in the
    center of the unit cell.
  - The PAW-setups include scalar-relativistic effects and are read
    from the new XML-format.

* Version 0.6.0

  - Added PAW-setup for Na and Mg.
  - Switched to numarray.
  - Calculates localization matrix for Wannier functions.
  - Non-periodic boundary conditions implemented.

* Version 0.5.2

  - Added a test script for the tutorial scripts.
  - Free energies are now calculated.
  - Forces are now correct for **k**-point calculations.
  - Added more tests - there are now 24 tests!

* Version 0.5.1

  - Changed Wavefunction to WaveFunction!

* Version 0.5

  - Added ``GetDensityArray`` and ``GetWavefunctionArray`` methods.
  - Timing of parallel runs fixed.
  - Restarting from a file much faster.
  
* Version 0.4

  - Fermi-level added to the ``.nc`` file.
  - ASE units used for ``.nc`` file.
  - Flushing the output file after each iteration.
  - Parallelized the code.
  - The interpolation of radial functions onto 3D grids is now
    three times faster.
  - Parallel jobs can be started from a script running in serial.
  - Preparing for switch to numarray.
  - Added a tutorial on how to calculate atomization energies.
  - The Python script that writes a netCDF file was removed from the
    netCDF file.
  - There is now one and only one ``libgridpaw.so`` extension module, that
    works for both serial and parallel calculations - at least for
    lam-mpi! 
 
* Version 0.3

  - Fixed bug #27: Wrong variable name ``PseudoElectronDensitiy`` in
    NetCDF file.
  - All real-space arrays are now three dimensional.
  - Moved the atomdata directory to ``~/.gridpaw``.
  - Moved ``49.pickle`` to ``~/.gridpaw/atomdata``.
  - NetCDF variable ``PseudoWavefunctions`` -> ``PseudoWaveFunctions``.
  - Forces have been fixed (#25).
  - Dipole moment is calculated.
  - The python script that writes a netCDF file is put in the netCDF
    file (#49).
  - Eigenvalues are now also in the ``.nc`` file.
  - Added a grid spacing keyword: ``'h'``.
  - Forces should now be correctly calculated for gamma-point
    calculations.

* Version 0.2 
