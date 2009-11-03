.. _lattice_constants:

=========================
Finding lattice constants
=========================

.. note:: WORK IN PROGRESS!


BCC iron
========

The easiest way to calculate the lattice constant of bulk iron is to
use the command-line tool simply called :program:`gpaw`::

  $ gpaw Fe -x bcc --cubic -a 2.84 -M 2.3 --xc=PBE --kpts=8,8,8 --h=0.18 --fit

The will start a GPAW calculation with:

* for Fe
* in the BCC crystal structure
* in an cubic unit cell with two atmos
* a lattice constant of 2.84 Å
* a magnetic moment of 2.3 per atom
* with the PBE XC functional
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

  $ gpaw Fe -x
  Calculating Fe ...
  Fit using 5 points:
  Volume per atom: 11.417 Ang^3
  Lattice constant: 2.837 Ang
  Bulk modulus: 226.9 GPa
  Total energy: -18.207 eV (2 atoms)


Convergence of lattice constant with respect to grid spacing
------------------------------------------------------------

When we vary the lattice constant, we also vary the grid-spacing
becauce the number of grid points is fixed.  If we look at :ref:`the
setups page for Fe <Iron>`_, then we can see that the energy converges
monotonically as a function of grid-spacing `h` towards the `h=0`
value.  If we linearize this variation arround `h_0 = 0.2` Å, then we
get `\Delta E = A(h-h_0)` with `A\simeq 1` eV/Å.  Let `a_0` be the
equilibrium lattice constant and `B` the bulk modulus for iron.  Then,
the energy as function of lattice constant `a` will be:

.. math::

  E(a) = \frac{9a_0}{2} B (a - a_0)^2 + A (h - h_0)

To find the minimum, we solve `dE(a)/da=0` using `a/h=a_0/h_0` and
get:

.. math::

  a - a_0 = -\frac{Ah_0}{9a_0^2B} = -0.002 \text{Å}.

Here we used `a_0=2.84` Å and `B=200\text{GPa}=1.25 \text{eV/Å}^3`.
It is seen that the minimum is shifted a bit towards smaller `a`.
If an error of 0.002 Å is not good enough, then one will have to reduce `A`
by reducing the grid-spacing or by using higher order stencils (see
:ref:`manual_stencils`).


Careful test for convergence
----------------------------

In addition to the grid-spacing, one should also make sure that the
result is converged with respect to number of **k**-points and with
respect to the smearing width used for occupations numbers.  Here is
an example showing how to test this using the Python interface behind
the :program:`gpaw` program:

.. literalinclude:: convergence.py

