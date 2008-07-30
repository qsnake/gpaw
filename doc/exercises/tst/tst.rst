.. _tst:

===================================================
Analytical problem on transition state theory (TST)
===================================================

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

.. math::

   V(x, y, z) = V_s [\exp(-\cos(2\pi x/b)
	-\cos(2\pi y/b) - 2\alpha z) - 2\exp(-\alpha z)]

For the parameters of the potential, we take :math:`V_s` = 0.2 eV,
:math:`b` = 3 Å, :math:`\alpha` = 2 Å\ :sup:`-1`.

Identify the minima, maxima and saddle points of the PES. You may do this
by either looking at the plot or taking the partial derivatives. You will need
the derivatives anyway at a later point.

Plot the function with the plot program of your choice. In the
following, it is described, how you can plot the potential with
:program:`Mathematica`. Type ``mathematica`` to start the
program.  Then, type::

  f[x_, y_, z_] := Vs * (Exp[-Cos[2 Pi x/b]-Cos[2 Pi y/b] -2 alpha z]
                         -2 Exp[-alpha z])

To make :program:`Mathematica` read and evaluate this statement, press
:kbd:`Shift-Enter`. You have to do this after each statement to make
:program:`Mathematica` evaluate your statement and give you
output. Then you can set the parameters by the statements::

  Vs = 0.2
  b = 3
  alpha = 2

Then you would like a 3D contour
plot of the function and type::

  Plot3D[f[x, y, -1], {x, -3, 3} ,{y, -3, 3}]

or::

  Plot3D[f[x, 0, z], {x, -3, 3} ,{z, -1.5, 3}]

Then again, press :kbd:`Shift-Enter`. Now you should see a pretty plot
of the potential.

Evaluate the activation energy for diffusion hops. Then Taylor expand
the potential energy about the minimum and saddle point to find the
frequency of the vibrational modes and evaluate the prefactor.  Add a
correction to the activation energy due to zero point energy.

What is the average length of time in between diffusion hops at room
temperature and at 400 K?
