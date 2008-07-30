==============
Bulk aluminium
==============

.. default-role:: math

Now we are ready to run the first GPAW calculation. We will look at
bulk fcc aluminum and make a single energy calculation at the
experimental lattice constant `a_0` = 4.05 Å. For the first example,
we choose 0.2 Å as grid spacing and 6 x
6 x 6 **k**-points.  Copy the script:

    Al_fcc.py_

    .. _Al_fcc.py : wiki:SVN:examples/aluminium/Al_fcc.py


to a place in your file area. Read the script and try to get an idea
of what it will do. Run the script by typing::

  $ python Al_fcc.py


The program will pop up a window showing the bulk structure.  Verify
that the structure indeed is fcc. Try to identify the closepacked
(111) planes.

Notice that the program has generated two output files::

  Al-fcc.gpw
  Al-fcc.txt

In general, when you execute a GPAW electronic structure
calculation, you get two files:

* A tar-file (conventional suffix ``.gpw``) containing binary data
  input/output files and an XML file. The contents may be viewed with
  the command::

    $ tar -tf Al-fcc.gpw

  The file info.xml has information about array sizes, types,
  endianness, parameters, and more.  Try::

    $ tar -f Al-fcc.gpw -xO info.xml

  or use ``tar --help`` for more options.

* An ASCII formatted log file (conventional suffix ``.txt``) that
  monitors the progress of the calculation.

Try to take a look at the file ``Al-fcc.txt``.  You can conveniently
monitor some variables by using the ``grep`` utility.  By typing::

  $ grep iter Al-fcc.txt

you see the progress of the iteration cycles including convergence of
wavefunctions, density and total energy.

The binary file contains all information about the calculation. Try
typing the following from the Python interpreter::

  >>> from gpaw import Calculator
  >>> calc = Calculator('Al-fcc.gpw')
  >>> bulk = calc.get_atoms()
  >>> print bulk.get_potential_energy()
  >>> density = calc.get_pseudo_valence_density()
  >>> from ase import *
  >>> write('Al.cube', bulk, data=density)
  >>> [hit CTRL-d]
  $ vmd Al.cube

Try to make VMD show an isosurface of the electron density.


Convergence in **k**-points and grid spacing
--------------------------------------------

Now we will investigate the necessary **k**-point sampling
and grid spacing needed for bulk fcc Aluminum at the
experimental lattice constant `a_0` = 4.05 Å; this is a standard
first step in all DFT calculations.

* Copy the script Al_fcc_convergence.py_  to a place in your file
  area.  Read the script and get an idea of what it will do. Then run
  the script by typing::

    $ python Al_fcc_convergence.py

* Estimate the necessary values of grid spacing and **k**-point sampling.

* Do you expect that the **k**-point / grid spacing test is universal
  for all other Aluminum structures than fcc? What about other
  chemical elements ?

* Are both **k**-point / grid spacing energy convergence covered by the
  variational principle, i.e. are all your calculated energies upper
  bounds to *true* total energy?

* Why do you think Al was chosen for this exercise?

..
  We use h = 0.2 Å
  and kpts = (8,8,8) for fcc and  kpts = (10,10,10) for bcc


.. _Al_fcc_convergence.py: wiki:SVN:examples/aluminium/Al_fcc_convergence.py


Equilibrium lattice properties
==============================

Having determined the necessary values of grid spacing and
**k**-point sampling, we now proceed to calculate some equilibrium
lattice properties of bulk Aluminum.

* First map out the cohesive curve `E(a)` for Al(fcc), i.e.  the
  total energy as function of lattice constant a, around the
  experimental equilibrium value of `a_0` = 4.05 Å.  Notice that the
  vacuum energy level `E(\infty)` is not zero.  Get four or more
  energy points, so that you can make a fit.

* Fit the data you have obtained to get `a_0` and the energy curve
  minimum `E_0=E(a_0)`.  From your fit, calculate the bulk
  modulus

  .. math:: B = \frac{M}{9a_0}\frac{d^2 E}{da^2}

  for *a* = `a_0`, where *M* is the number of atoms per cubic unit
  cell.  Make the fit using your favorite math package
  (Mathematica/MatLab/Maple/Python/...) or use `ag` like this::

    $ ag bulk-*.txt

  Then choose `Tools` -> `Bulk Modulus`.

* Compare your results to the experimental values `a_0` = 4.05 Å and `B`
  = 76 GPa.  Mind the units when you calculate the bulk modulus.
  What are the possible error sources, and what quantity is more
  sensitive, the lattice constant or the bulk modulus?





Equilibrium lattice properties for bcc
======================================

* Set up a similar calculation for bcc, in the minimal unit cell. Note that 
  the cubic unit cell for a bcc lattice only contains two atoms.
  
* Make a qualified starting guess on *a*\ :sub:`bcc` from the lattice
  constant for fcc, that you have determined above. One can either
  assume that the primitive unit cell volumes of the fcc and bcc
  structure are the same or that the nearest neighbor distances are
  the same. Find a guess for *a*\ :sub:`bcc` for both
  assumptions. Later, you can comment on which assumption gives the
  guess closer to the right lattice constant.

* Check that your structure is right by repeating the unit cell. In `ag` this
  is done by choosing `View` -> `Repeat...`.

* Map out the cohesive curve *E*\ (*a*) for Al(bcc) and determine *a*\
  :sub:`bcc`, using a few points.  Is it a good idea to use the same
  **k**-point setup parameters as for the fcc calculations?  Calculate the
  bulk modulus, as it was done for fcc, and compare the result to the
  fcc bulk modulus. What would you expect?

* Using the lattice constants determined above for fcc and bcc,
  calculate the fcc/bcc total energies at different grid spacings:
  0.25 Å and 0.2 Å, i.e. four calculations.  Compare the
  structure energy differences for the two cutoffs.  Generally,
  energy differences converge much faster
  with grid spacing than total energies themselves.  Further
  notice that the energy zero does not
  have physical significance. This exercise is sensitive to the number
  of **k**-points, make sure that your **k**-point sampling is dense enough.

* GPAW requires an orthorhombic unit cell and therefore one cannot choose a
  primitive unit cell with one atom for bcc and fcc calculations. Show that it 
  is  possible to choose an orthorhombic (but not cubic) unit cell for fcc 
  which contains two atoms. Would this minimal choice affect the choice of 
  **k**-point sampling?

.. default-role::
