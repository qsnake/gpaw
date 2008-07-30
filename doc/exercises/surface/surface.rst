=================
Aluminum surfaces
=================

In this exercise, we make a toolbox for building an Al(100) surface. For this
surface, we calculate the surface energy and other properties.


Making Python Tool Boxes
========================

A science project (like the one you are going to make), will often
contain some repeated and similar sub tasks like loops over different
kind of atoms, structures, parameters etc.  As an alternative to a
plethora of similar Python scripts, made by *copy+paste*, it is
advantageous to put the repeated code into tool boxes.

Python supports such tool boxes (in Python called modules): put any
Python code into a file ``stuff.py`` then it may be used as a tool box
in other scripts, using the Python command: ``from stuff import
thing``, where ``thing`` can be almost anything.  When Python sees
this line, it runs the file ``stuff.py`` (only the first time) and
makes ``thing`` available.  Lets try an example:

* In file ``stuff.py``, put::

    constant = 17


    def function(x):
        return x - 5

* and in file ``program.py``, put::

    from stuff import constant, function
    print 'result =', function(constant)

* Now run the script ``program.py`` and watch the output.

You can think of ASE and GPAW as big collections of modules, that
we use in our scripts.



Fcc Surface Builders
--------------------

As a non-trivial example of a Python module, we'll try to make a tool
box for making fcc surface slabs with arbitrary number of layers.  A
real fcc surface has a large number of atomic layers, but most surface
properties are well reproduced by a slab with just 2 - 20 layers,
depending on the material and what properties you are looking for.

The most important cubic surfaces are (100), (110), and (111).  For
face centered cubic, (111) has the most compact atomic arrangement,
whereas (110) is most open. Here we'll focus on (100).

* What is the coordination number *Z* (number of nearest neighbors) of an
  fcc(100) surface atom?  What is it for a bulk atom?

* Now that we know the surface geometry, we can setup a toolbox
  for making surface structures with arbitrary number of layers.  Copy
  the script build_bcc.py_ to your area.  Browse the script and try
  to understand it. The central part is the function starting with
  ``def bcc100(...)``  that creates a body centered cubic (100)
  surface.

* Create a ``build_fcc.py`` script containing a function called ``fcc100``.
  This function should build an fcc(100) surface slab.  Run the script
  until you are sure that your new function works (by running the
  script, you activate the self test in the last few lines of the
  script, starting at ``if __name__ == '__main__':`` - the
  self test is not done, when you import from your module, though).


.. hint::

   Square roots are calculated like this: ``2**0.5`` or
   ``sqrt(2)`` (the ``sqrt`` function must first be imported: ``from
   math import sqrt`` or ``from ase import *``).

.. note::

   In python, ``/`` is used for both integer- and float
   divisions. Integer division is only performed if both sides of the
   operator are integers (you can always force an integer division by
   using ``//``)::

     >>> 1 / 3
     0
     >>> 1 / 3.0
     0.33333333333333331

.. _build_bcc.py: wiki:SVN:examples/surface/build_bcc.py

Aluminum fcc(100) Surface Slabs
===============================

In this section, we'll study how surface properties converge, as
the slab becomes thicker and thicker.


Surface Energetics
------------------

One surface property is the surface tension
`\sigma` defined implicitly via:

.. math:: E_N = 2A\sigma + NE_B

where `E_N` is the total energy of a slab with `N` layers,
`A` the area of the surface unit cell (the factor 2 because the slab
has two surfaces), and finally `E_B` is the total energy per bulk
atom.  The limit `N \rightarrow \infty` corresponds to the macroscopic
crystal termination.

Estimate the surface tension using an expression from the simplest
Effective Medium Theory (EMT) description:

.. math:: A\sigma \simeq [1 - (Z/Z_0)^{1/2}] E_{coh}

where `Z` and `Z_0` are the coordination numbers (number of nearest
neighbors) of a surface and a bulk atom, respectively, and `A` is the
surface area per surface atom, and `E_{coh}` = `E_{atom}-E_B` > 0 is
the cohesive energy per bulk atom. For Aluminium we have `E_{coh}` = 3.34 eV.

* Derive the following equation:

  .. math:: \sigma = \frac{NE_{N-1} - (N-1)E_N}{2A}

* Take a look at the script `Al100.py`_.  Calculate `\sigma` for `N` =
  2, 3, 4, 5 and 6.  Use a two-dimensional Monkhorst-Pack **k**-point
  sampling (``kpts=(k, k, 1)``) that matches the size of your unit
  cell.  The experimental value of `\sigma` is 54 meV/Å\ :sup:`2`.  How
  well is the EMT estimate satisfied?

  .. hint::

    A rule of thumb for choosing the initial **k**-point sampling is,
    that the product, *ka*, between the number of **k**-points, *k*,
    in any direction, and the length of the basis vector in this
    direction, *a*, should be:

    * *ka* ~ 30 Å, for *d* band metals
    * *ka* ~ 25 Å, for simple metals
    * *ka* ~ 20 Å, for semiconductors
    * *ka* ~ 15 Å, for insulators

    Remember that convergence in this parameter should always be checked.

.. _Al100.py : wiki:SVN:examples/surface/Al100.py



Work function
-------------

Run the work_function.py_ script and estimate the work function for a
Al(100) surface. A typical experimental value for the work function of 
the Al(100) surface is 4.20 eV.
Try to do the slab calculation with periodic
boundary conditions in all three directions, and run the script again.
How does this affect the Fermi level and the average potential?


.. _work_function.py : wiki:SVN:examples/surface/work_function.py
