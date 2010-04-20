.. _lattice_constants:

=========================
Finding lattice constants
=========================

.. seealso::

   * `ASE equation of state module
     <https://wiki.fysik.dtu.dk/ase/ase/utils.html#equation-of-state>`_
   * `ASE EOS tutorial
     <https://wiki.fysik.dtu.dk/ase/tutorials/eos/eos.html>`_


BCC iron
========

The easiest way to calculate the lattice constant of bulk iron is to
use the command-line tool simply called :program:`gpaw`:

.. highlight:: bash

::

  $ gpaw Fe -x bcc --cubic -a 2.84 -M 2.3 --xc=PBE --kpts=8,8,8 --h=0.18 --fit

This will start a GPAW calculation with:

* 2 Fe atoms
* in the BCC crystal structure
* in an cubic unit cell with two atoms
* a lattice constant of 2.84 Å
* a magnetic moment of 2.3 per atom
* with the PBE XC-functional
* 8x8x8 **k**-points
* and a grid-spacing of approximately 0.18 Å.

The final option to the :program:`gpaw` command is the :option:`--fit`
option which will make the program do five calculations for five
different lattice constants distributed from -2% to +2%  arround the
given value of 2.84 Å.

Try::

  $ gpaw --help

for the full explanation.

The result of the calculation is::

  Fit using 5 points:
  Volume per atom: 11.493 Ang^3
  Lattice constant: 2.843 Ang
  Bulk modulus: 179.1 GPa
  Total energy: -18.153 eV (2 atoms)

.. highlight:: python


Careful test for convergence
----------------------------

In addition to the grid-spacing, one should also make sure that the
result is converged with respect to number of **k**-points and with
respect to the smearing width used for occupations numbers.

From the figure below (`h=0.18\text{Å}`), one can see that with a
smearing width of 0.2 eV, one will get quick convergence of the
lattice constant with respect to the number of **k**-points, but
convergence to a value that is a bit too large.

.. image:: Fe_conv_k.png

Convergence with respect to grid-spacing looks like this (6x6x6
**k**-points and width=0.1 eV):

.. image:: Fe_conv_h.png

For iron, one can get a reasonable value for the lattice constant
using `h=0.18\text{Å}`, 8x8x8 **k**-points and
``occupations=FermiDirac(0.1)``.

These test calculations were performed with the Python interface behind
the :program:`gpaw` program:

.. literalinclude:: iron.py


Convergence of lattice constant with respect to grid spacing
------------------------------------------------------------

When we vary the lattice constant, we also vary the grid-spacing
becauce the number of grid points is fixed.  If we look at :ref:`the
setups page for Fe <Iron>`, then we can see that the energy converges
monotonically as a function of grid-spacing `h` towards the `h=0`
value.  If we linearize this variation arround `h_0 = 0.18 \text{Å}`,
then we get `\Delta E = A(h-h_0)` with `A\simeq 1 \text{eV/Å}`.  Let
`a_0` be the equilibrium lattice constant and `B` the bulk modulus for
iron.  Then, the energy as function of lattice constant `a` will be:

.. math::

  E(a) = \frac{9 a_0}{2} B (a - a_0)^2 + A (h - h_0)

To find the minimum, we solve `dE(a)/da=0` using `a/h=a_0/h_0` and
get:

.. math::

  a - a_0 = -\frac{Ah_0}{9a_0^2B} = -0.002 \text{Å}.

Here we used `a_0=2.84 \text{Å}` and `B=180\text{GPa}=1.12 \text{eV/Å}^3`.
It is seen that the minimum is shifted a bit towards smaller `a`.
If an error of 0.002 Å is not good enough, then one will have to reduce `A`
by reducing the grid-spacing or by using higher order stencils (see
:ref:`manual_stencils`).

.. note::

  For calculations where the unit cell is fixed, the `A(h-h_0)` term
  will cancel out if one is looking at energy differences.
