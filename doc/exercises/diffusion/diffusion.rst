.. _diffusion_exercise:

=========================================
Diffusion of gold atom on Al(100) surface
=========================================

In these two tutorials:

* :ase:`NEB <tutorials/neb/diffusion.html>`
* :ase:`Constraint <tutorials/constraints/diffusion.html>`

the energy barrier for diffusion of a gold atom on an Al(100) surface
was calculated using a semi-emperical EMT potential.  In this
exercise, we will try to use DFT and GPAW.

* Run the scripts form one of the two tutorials above to get good
  initial guesses for the height of the gold atom in the initial and
  transition states (hollow and bridge sites).

The PAW setups for both Al and Au are quite smooth (see
:ref:`Aluminium` and :ref:`Gold`), so we can try with a coarse
real-space grid with grid-spacing 0.25 Å.  For a quick'n'dirty
calculation we can do with just a :math:`2 \times 2` sampling of the
surface Brillouin zone.  Use these parameters for the DFT
calculations::

  from gpaw import *
  calc = GPAW(h=0.25, kpts=(2, 2, 1), xc='PBE')

In order to speed up the calculation, use just a single frozen Al(100) layer.

* What is the PBE energy barrier?

* Do we need to apply any constraint to the gold atom?

* Can both initial and transition state calculations be done with only
  one **k**-point in the irreducible part of the Brillouin zone?

* Try to repeat the EMT calculations with a single frozen Al(100) layer.

