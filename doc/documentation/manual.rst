.. _manual:

======
Manual
======

.. default-role:: math


GPAW calculations are controlled through scripts written in the
programming language Python_.  GPAW relies on the :ase:`Atomic
Simulation Environment <>` (ASE), which is a Python package that helps
us describe our atoms.  The ASE package also
handles molecular dynamics, analysis, visualization, geometry
optimization and more.  If you don't know anything about ASE, then it
might be a good idea to familiarize yourself with it before continuing
(at least read the :ase:`ASE introduction <intro.html>`).

Below, there will be Python code examples starting with ``>>>`` (and
``...`` for continuation lines).  It is a good idea to start the
Python interpreter and try some of the examples below.

.. _Python: http://www.python.org

The units used by the GPAW calculator correspond to the :ase:`ASE
conventions <ase/units.html>`, most importantly electron volts and
angstroms.

-----------------------
Doing a PAW calculation
-----------------------

To do a PAW calculation with the GPAW code, you need an ASE
:ase:`Atoms <ase/atoms.html>` object and a GPAW
:class:`~gpaw.aseinterface.Calculator`::

   _____________          ____________
  |             |        |            |
  | Atoms       |------->| GPAW       |
  |             |        |            |
  |_____________|        |____________|
       atoms                  calc

In Python code, it looks like this:

.. literalinclude:: h2.py

If the above code was executed, a calculation for a single `\rm{H}_2`
molecule would be started.  The calculation would be done using a
supercell of size :math:`6.0 \times 6.0 \times 6.0` Å with cluster
boundary conditions.  The parameters for the PAW calculation are:

* 2 electronic bands.
* Local density approximation (LDA)\ [#LDA]_ for the
  exchange-correlation functional.
* Spin-paired calculation.
* :math:`32 \times 32 \times 32` grid points.

The values of these parameters can be found in the text output file:
`h2.txt <../h2.txt>`_.

The calculator will try to make sensible choices for all parameters
that the user does not specify.  Specifying parameters can be done
like this:

>>> calc = GPAW(nbands=1,
...             xc='PBE',
...             gpts=(24, 24, 24))

Here, we want to use one electronic band, the Perdew, Burke, Ernzerhof
(PBE)\ [#PBE]_ exchange-correlation functional and 24 grid points in
each direction.


----------
Parameters
----------

The complete list of all possible parameters and their defaults is
shown below. A detailed description of the individual parameters is
given in the following sections.

=================  =========  ===================  ============================
keyword            type       default value        description
=================  =========  ===================  ============================
``mode``           ``str``    ``'fd'``             :ref:`manual_mode`
``nbands``         ``int``                         :ref:`manual_nbands`
``xc``             ``str``    ``'LDA'``            :ref:`manual_xc`
``kpts``           *seq*      `\Gamma`-point       :ref:`manual_kpts`
``spinpol``        ``bool``                        :ref:`manual_spinpol`
``gpts``           *seq*                           :ref:`manual_gpts`
``h``              ``float``  ``0.2``              :ref:`manual_h`
``usesymm``        ``bool``   ``True``             :ref:`manual_usesymm`
``random``         ``bool``   ``False``            Use random numbers for
                                                   :ref:`manual_random`
``width``          ``float``  ``0`` or ``0.1`` eV  Width of :ref:`manual_width`
``lmax``           ``int``    ``2``                Maximum angular momentum
                                                   for expansion of
                                                   :ref:`manual_lmax`
``charge``         ``float``  ``0``                Total :ref:`manual_charge`
                                                   of the system
``convergence``    ``dict``                        :ref:`manual_convergence`
``maxiter``        ``int``    ``120``              :ref:`manual_maxiter`
``txt``            ``str``,   ``sys.stdout``       :ref:`manual_h`
                   None, or
                   file obj.
``parsize``        *seq*                           Parallel
                                                   :ref:`manual_parsize`
``stencils``       tuple      ``(2, 3)``           Number of neighbors for
                                                   :ref:`manual_stencils`
``mixer``          Object                          Pulay :ref:`manual_mixer`
                                                   scheme
``fixdensity``     ``bool``   ``False``            Use :ref:`manual_fixdensity`
``fixmom``         ``bool``   ``False``            Use :ref:`manual_fixmom`
``setups``         ``str``    ``'paw'``            :ref:`manual_setups`
                   or
                   ``dict``
``basis``          ``str``    ``{}``               Specification of
                   or                              :ref:`manual_basis`
                   ``dict``
``eigensolver``    ``str``    ``'rmm-diis'``       :ref:`manual_eigensolver`
``hund``           ``bool``   ``False``            :ref:`Use Hund's rule
                                                   <manual_hund>`
``parsize_bands``  ``int``    ``1``                XXX Missing doc
``external``       Object                          XXX Missing doc
``verbose``        ``int``    ``0``                XXX Missing doc
``poissonsolver``  Object                          XXX Missing doc
``communicator``   Object                          XXX Missing doc
``idiotproof``     ``bool``   ``True``             XXX Missing doc
``notify``         Object                          XXX Missing doc
=================  =========  ===================  ============================

*seq*: A sequence of three ``int``'s.


.. note:: 
   
   Parameters can be changed after the calculator has been constructed
   by using the :meth:`~gpaw.paw.set` method:

   >>> calc.set(txt='H2.txt', charge=1)

   This would send all output to a file named :file:`'H2.txt'`, and the
   calculation will be done with one electron removed.


.. _manual_mode:

Finite Difference or LCAO mode
------------------------------

The default mode (``mode='fd'``) is Finite Differece. This means that
the wave functions will be expanded on a real space grid. The
alternative is to expand the wave functions on a basis-set constructed
as linear combination as atomic-like orbitals, in short LCAO. This is
done by setting (``mode='lcao'``).

See also the page on :ref:`lcao`.

.. _manual_nbands:

Number of electronic bands
--------------------------

The default number of electronic bands (``nbands``) is equal to the
number of atomic orbitals present in the atomic setups.  For systems
with the occupied states well separated from the unoccupied states,
one could use just the number of bands needed to hold the occupied
states.  For metals, more bands are needed.  Sometimes, adding more
unoccupied bands will improve convergence.

.. tip::
   ``nbands=0`` will give zero empty bands, and ``nbands=-n`` will
   give ``n`` empty bands.


.. _manual_xc:

Exchange-Correlation functional
-------------------------------

The exchange-correlation functional can be one of (only the most
common are listed here, for the complete list see
:trac:`~gpaw/libxc_functionals.py`):

============  =================== ===========================  ==========
``xc``        libxc_ keyword      description                  reference 
============  =================== ===========================  ==========
``'LDA'``     ``'X-C_PW'``        Local density approximation  [#LDA]_
``'PBE'``     ``'X_PBE-C_PBE'``   Perdew, Burke, Ernzerhof     [#PBE]_
``'revPBE'``  ``'X_PBE_R-C_PBE'`` revised PBE                  [#revPBE]_
``'RPBE'``    ``'X_RPBE-C_PBE'``  revised revPBE               [#RPBE]_
============  =================== ===========================  ==========

``'LDA'`` is the default value.  The three last ones are of
generalized gradient approximation (GGA) type.

The functionals from libxc_ are used by default - keywords are based
on the :file:`gpaw/libxc_functionals.py` file.  Custom combinations of
exchange and correlation functionals are allowed, the exchange and
correlation strings from the :file:`gpaw/libxc_functionals.py` file need
to be stripped off the ``'XC_LDA'`` or ``'XC_GGA'`` prefix and
combined using the dash (-); e.g. to use "the" LDA approximation (most
common) in chemistry specify ``'X-C_VWN'``.

**For developers only**: It is still possible to use the "old" functionals
by prefixing the keyword with ``'old'``, e.g. ``'oldrevPBEx'``.
It this case the ``'oldrevPBEx'`` setup will be used.

See details of implementation on the :ref:`xc_functionals` page.

.. _libxc: http://www.tddft.org/programs/octopus/wiki/index.php/Libxc


.. _manual_kpts:

Brillouin-zone sampling
-----------------------

The default sampling of the Brillouin-zone is with only the
`\Gamma`-point.  This allows us to choose the wave functions to be real.
Monkhorst-Pack sampling can be used if required: ``kpts=(n1, n2,
n3)``, where ``n1``, ``n2`` and ``n3`` are positive integers.  This
will sample the Brillouin-zone with a regular grid of ``n1`` `\times`
``n2`` `\times` ``n3`` **k**-points.


.. _manual_spinpol:

Spinpolarized calculation
-------------------------

If any of the atoms have magnetic moments, then the calculation will
be spin-polarized - otherwise, a spin-paired calculation is carried
out.  This behavior can be overruled with the ``spinpol`` keyword
(``spinpol=True``).


.. _manual_gpts:

Number of grid points
---------------------

The number of grid points to use for the grid representation of the
wave functions determines the quality of the calculation.  More
gridpoints (smaller grid spacing, *h*), gives better convergence of
the total energy.  For most elements, *h* should be 0.2 Å for
reasonable convergence of total energies.  If a ``n1`` `\times` ``n2``
`\times` ``n3`` grid is desired, use ``gpts=(n1, n2, n3)``, where
``n1``, ``n2`` and ``n3`` are positive ``int``s all divisible by four.
Alternatively, one can use something like ``h=0.25``, and the program
will try to choose a number of grid points that gives approximately
the desired grid spacing.  For more details, see :ref:`grids`.

If you are more used to think in terms of plane waves; a conversion
formula between plane wave energy cutoffs and realspace grid spacings
have been provided by Briggs *et. al* PRB **54**, 14362 (1996).  The
conversion can be done like this::

  >>> from gpaw.utilities.tools import cutoff2gridspacing, gridspacing2cutoff
  >>> from ase import *
  >>> h = cutoff2gridspacing(50 * Rydberg)


.. _manual_h:

Grid spacing
------------

XXX Missing doc


.. _manual_usesymm:

Use of symmetry
---------------

With ``usesymm=True`` (default) the **k**-points are reduced to only
those in the irreducible part of the Brillouin-zone.  Moving the atoms
so that a symmetry is broken will cause an error.  This can be avoided
by using ``usesymm=False`` which will reduce the number of applied
symmetries to just the time-reversal symmetry (implying that the
Hamiltonian is invariant under **k** -> -**k**). For some purposes you
might want to have no symmetry reduction of the **k**-points at all
(debugging, transport, wannier functions). This can be achieved be
specifying ``usesymm=None``.


.. _manual_random:

Wave function initialization
----------------------------

By default, a linear combination of atomic orbitals is used as initial
guess for the wave functions. If the user wants to calculate more bands
than there are precalculated atomic orbitals, random numbers will be
used for the remaining bands.


.. _manual_width:

Fermi-distribution
------------------

The width (`k_B T`) of the Fermi-distribution used for
occupation numbers:

.. math::  f(E) = \frac{1}{1 + \exp[E / (k_B T)]}

is given by the ``width`` keyword.  For calculations with
**k**-points, the default value is 0.1 eV and the total energies are
extrapolated to *T* = 0 Kelvin.  For a `\Gamma`-point calculation (no
**k**-points) the default value is ``width=0``, which gives integer
occupation numbers.


.. _manual_lmax:

Compensation charges
--------------------

The compensation charges are expanded with correct multipoles up to
and including `\ell=\ell_{max}`.  Default value: ``lmax=2``.


.. _manual_charge:

Charge
------

The default is charge neutral.  The systems total charge may be set in
units of the negative electron charge (i.e. ``charge=-1`` means one
electron more than the neutral).


.. _manual_convergence:

Accuracy of the self-consistency cycle
--------------------------------------

The ``convergence`` keyword is used to set the convergence criteria.
The default value is this Python dictionary::

  {'energy': 0.001, # eV
   'density': 1.0e-4,
   'eigenstates': 1.0e-9,
   'bands': 'occupied'}

In words:

* The energy change (last 3 iterations) should be less than 1 meV.

* The change in density (integrated absolute value of density change) 
  should be less than 0.001 electrons per valence electron.

* The integrated value of the square of the residuals of the Kohn-Sham
  equations should be less than :math:`1.0 \times 10^{-9}` per state
  (FD mode only).

The individual criteria can be changed by giving only the specific
entry of dictionary e.g. ``convergence={'energy': 0.0001}`` would set
the convergence criteria of energy to 0.1 meV while other criteria
remain in their default values.

As the total energy and charge density depend only on the occupied
states, unoccupied states do not contribute to the convergence
criteria.  However, with the ``bands`` set to ``'all'``, it is
possible to force convergence also for the unoccupied states.  One can
also use ``{'bands': 200}`` to converge the lowest 200 bands. One can
also write ``{'bands': -10}`` to converge all bands except the last
10. It is often hard to converge the last few bands in a calculation.

The calculation will stop with an error if convergence is not reached
in ``maxiter`` self-consistent iterations (defaults to 120).


.. _manual_maxiter:

Maximum number of SCF-iterations
--------------------------------

XXX Missing doc


.. _manual_txt:

Where to send text output
-------------------------

The ``txt`` keyword defaults to the string ``'-'``, which means
standard output.  One can also give a ``file`` object (anything with a
``write`` method will do).  If a string (different from ``'-'``) is
passed to the ``txt`` keyword, a file with that name will be opened
and used for output.  Use ``txt=None`` to disable all text output.


.. _manual_stencils:

Finite-difference stencils
--------------------------

XXX Missing doc


.. _manual_mixer:

Density mixing
--------------

The default is to use Pulay mixing using the three last densities, a
linear mixing coefficient of 0.25 and no special metric for estimating
the magnitude of the change from input density to output density -
this is equivalent to ``mixer=Mixer(0.25, 3)``.  In some cases
(metals) it can be an advantage to use something like
``mixer=Mixer(0.1, 5, metric='new', weight=100.0)``.  Here, long
wavelength changes are weighted 100 times higher than short wavelength
changes. In spin-polarized calculations using Fermi-distribution
occupations one has to use :class:`~gpaw.mixer.MixerSum` instead of
:class:`~gpaw.mixer.Mixer`.

See also the documentation on :ref:`density mixing <densitymix>`.


.. _manual_fixdensity:

Fixed density
-------------

XXX Missing doc


.. _manual_fixmom:

Fixed total magnetic moment
---------------------------

XXX Missing doc


.. _manual_setups:

Type of setup to use
--------------------

The ``setups`` keyword can be a dictionary mapping chemical symbols or
atom numbers to types of setups (strings).  The default type is
``'paw'``.  Another type is ``'ae'`` for all-electron calculations.
In the future there might be a ``'hgh'`` type for
Hartwigsen-Goedecker-Hutter pseudopotential calculations.  An
example::

  setups={'Li': 'mine', 'H': 'ae'}

For an LDA calculation, GPAW will look for :file:`Li.mine.LDA` (or
:file:`Li.mine.LDA.gz`) in your :envvar:`GPAW_SETUP_PATH` environment
variable and use an all-electron potential for hydrogen atoms.


.. _manual_basis:

Atomic basis set
----------------

The ``basis`` keyword can be used to specify the basis set which
should be used in LCAO mode, which also affects the LCAO
initialization in FD mode.

In FD mode, the initial guess for the density / wave functions is
determined by solving the Kohn-Sham equations in the LCAO basis.

Default is to use the pseudo partial waves from the setup as a
basis. This basis is always available; choosing anything else requires
the existence of the corresponding basis set file in the
:envvar:`GPAW_SETUP_PATH`.

For details on the LCAO mode and generation of basis set files; see
the :ref:`LCAO <lcao>` documentation.


.. _manual_eigensolver:

Eigensolver
-----------

The default solver for iterative diagonalization of the Kohn-Sham
Hamiltonian is RMM-DIIS (Residual minimization method - direct
inversion in iterative subspace) which seems to perform well in most
cases. However, some times more efficient/stable convergence can be
obtained with a different eigensolver. Especially, when calculating
many unoccupied states RMM-DIIS might not be optimal. Further
available options in FD mode are conjugate gradient method
(``eigensolver='cg'``) and a simple Davidson method
(``eigensolver='dav'``). From the alternatives, conjugate gradient
seems to perform better in general.

LCAO mode has its own eigensolver, which directly diagonalizes the
Hamiltonian matrix instead of using an iterative method.


.. _manual_hund:

Using Hund's rule for guessing initial magnetic moments
-------------------------------------------------------

The ``hund`` keyword can be used for single atoms only. If set to
``True``, the calculation will become spinpolarized, and the initial
ocupations, and magnetic moment of the atom will be fixed to the value
required by Hund's rule. Any user specified magnetic moment is
ignored. Default is False.

.. _manual_parallel_calculations:

---------------------
Parallel calculations
---------------------

Information about running parallel calculations can be found on the
:ref:`parallel_runs` page.


.. _zero_energy:

--------------
Total Energies
--------------

The GPAW code calculates energies relative to the energy of separated
reference atoms, where each atom is in a spin-paired, neutral, and
spherically symmetric state - the state that was used to generate the
setup.  For a calculation of a molecule, the energy will be minus the
atomization energy and for a solid, the resulting energy is minus the
cohesive energy.  So, if you ever get positive energies from your
calculations, your system is in an unstable state!

.. note::
   You don't get the true atomization/cohesive energy.  The true
   number is always lower, because most atoms have a spin-polarized
   and non-spherical symmetric ground state, with an energy that is
   lower than that of the spin-paired, and spherically symmetric
   reference atom.


------------------------
Restarting a calculation
------------------------

The state of a calculation can be saved to a file like this:

>>> calc.write('H2.gpw')

The file :file:`H2.gpw` is a binary file containing
wave functions, densities, positions and everything else (also the
parameters characterizing the PAW calculator used for the
calculation).

If you want to restart the `\rm{H}_2` calculation in another Python session
at a later time, this can be done as follows:

>>> from gpaw import *
>>> atoms, calc = restart('H2.gpw')
>>> print atoms.get_potential_energy()

Everything will be just as before we wrote the :file:`H2.gpw` file.
Often, one wants to restart the calculation with one or two parameters
changed slightly.  This is very simple to do.  Suppose you want to
change the number of grid points:

>>> atoms, calc = restart('H2.gpw', gpts=(20, 20, 20))
>>> print atoms.get_potential_energy()

.. tip::
   There is an alternative way to do this, that can be handy sometimes:

   >>> atoms, calc = restart('H2.gpw')
   >>> calc.set(gpts=(20, 20, 20))
   >>> print atoms.get_potential_energy()


More details can be found on the :ref:`restart_files` page.

----------------------
Command line arguments
----------------------

The behaviour of GPAW can be controlled with some command line
arguments. The arguments for GPAW should be specified after the
python-script, i.e.::

    python script.py [options]

The possible command line arguments are:

===============================  ============================================
argument                         description
===============================  ============================================
``--trace``
``--debug``                      Run in debug-mode, e.g. check
                                 consistency of arrays passed to c-extensions
``--setups=path``                Use setups from the colon-separated
                                 list of directories in ``path``
``--dry-run[=nprocs]``           Print out the computational
                                 parameters and estimate memory usage, 
                                 do not perform actual calculation. 
                                 If ``nprocs`` is specified, print also how 
                                 parallelization would be done.
``--domain-decomposition=comp``  Specify the domain decomposition with
				 the tuple ``comp``, e.g. ``(2,2,2)``
===============================  ============================================


----------
Extensions
----------

Currently available extensions:

 1. :ref:`Linear response time-dependent DFT <lrtddft>`
 2. :ref:`Time propagation time-dependent DFT <timepropagation>`


:ref:`lrtddft`
--------------

Optical photoabsorption spectrum can be simulated using :ref:`lrtddft`


:ref:`timepropagation`
----------------------

Optical photoabsorption spectrum as well as nonlinear effects can be
studied using :ref:`timepropagation`. This approach
scales better than linear response, but the prefactor is so large that
for small and moderate systems linear response is significantly
faster.




.. [#LDA]    J. P. Perdew and Y. Wang,
             Accurate and simple analytic representation of the
             electron-gas correlation energy
             *Phys. Rev. B* **45**, 13244-13249 (1992)
.. [#PBE]    J. P. Perdew, K. Burke, and M. Ernzerhof,
             Generalized Gradient Approximation Made Simple,
             *Phys. Rev. Lett.* **77**, 3865 (1996)
.. [#revPBE] Y. Zhang and W. Yang,
             Comment on "Generalized Gradient Approximation Made Simple",
             *Phys. Rev. Lett.* **80**, 890 (1998)
.. [#RPBE]   B. Hammer, L. B. Hansen and J. K. Nørskov,
             Improved adsorption energetics within density-functional
             theory using revised Perdew-Burke-Ernzerhof functionals,
             *Phys. Rev. B* **59**, 7413 (1999)

.. default-role::
