.. _tst:

===================================================
Analytical problem on transition state theory (TST)
===================================================


.. |angst|  unicode:: U+0212B .. ANGSTROM SIGN
.. |infin|  unicode:: U+0221E .. INFINITY
.. |simeq|  unicode:: U+02243 .. ASYMPTOTICALLY EQUAL TO
.. |sigma|  unicode:: U+003C3 .. GREEK SMALL LETTER SIGMA
.. |Delta|  unicode:: U+00394 .. GREEK CAPITAL LETTER DELTA
.. |mu|     unicode:: U+003BC .. GREEK SMALL LETTER MU
.. |beta|   unicode:: U+003B2 .. GREEK SMALL LETTER BETA
.. |pi|     unicode:: U+003C0 .. GREEK SMALL LETTER PI
.. |alpha|  unicode:: U+003B1 .. GREEK SMALL LETTER ALPHA
.. |nu|     unicode:: U+003BD .. GREEK SMALL LETTER NU
.. |deg|    unicode:: U+000B0 .. DEGREE SIGN


This problem is mostly analytical and was developed by 
Hannes Jonsson. It deals with the potential energy surface (PES) 
of a hydrogen atom adsorbed on a (100) crystal surface of an FCC metal.

What are the four basic assumptions of TST?  What should the shape of
the potential energy surface be like in order for TST to give a good
approximation to the rate constant of a transition?

A hydrogen atom adsorbed on the surface of a metal crystal can diffuse
by hopping from one binding site to another. The atom can be
considered to be a particle moving on a periodic potential surface
(PES) while the metal atoms can be taken to be stationary. This is a
good first approximation because the metal atoms are so much heavier
than the hydrogen atom.  For a hydrogen atom on the (100) surface of
an FCC metal, the potential energy of the hydrogen atom can be
approximated by the function

  *V*\ (*x, y, z*) = *V*\ :sub:`s`\ [exp(-cos(2\ |pi|\ *x*\ /*b*)
  - cos(2\ |pi|\ *y*\ /*b*) - 2\ |alpha|\ *z*) - 2exp(-|alpha|\ *z*)]

For the parameters of the potential, we take *V*\ :sub:`s` = 0.2 eV,
*b* = 3 |angst|, |alpha| = 2 |angst|\ :sup:`-1`.

Identify the minima, maxima and saddle points of the PES. You may do this
by either looking at the plot or taking the partial derivatives. You will need
the derivatives anyway at a later point.

Plot the function with the
plot program of your choice. In the following, it is described, how
you can plot the potential with Mathematica. Type ``mathematica`` to
start the program.  Then, type::

  f[x_, y_, z_] := Vs * (Exp[-Cos[2 Pi x/b]-Cos[2 Pi y/b] -2 alpha z]
                         -2 Exp[-alpha z])

To make mathematica read and evaluate this statement, press
SHIFT + ENTER. You have to do this after each statement to make mathematica
evaluate your statement and give you output. Then you can set the parameters
by the statements::

  Vs = 0.2
  b = 3
  alpha = 2

Then you would like a 3D contour
plot of the function and type::

  Plot3D[f[x, y, -1], {x, -3, 3} ,{y, -3, 3}]

or::

  Plot3D[f[x, 0, z], {x, -3, 3} ,{z, -1.5, 3}]

Then again, press SHIFT+ENTER. Now you should see a pretty plot of the
potential.

Evaluate the activation energy for diffusion hops. Then Taylor expand
the potential energy about the minimum and saddle point to find the
frequency of the vibrational modes and evaluate the prefactor.  Add a
correction to the activation energy due to zero point energy.

What is the average length of time in between diffusion hops at room
temperature and at 400 K?
