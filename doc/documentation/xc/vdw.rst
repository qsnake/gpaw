.. _vdw:

=============
van der Waals
=============

There is one method of including van der Waals forces in GPAW
[#vdW-DF]_, [#vdW-DF2]_, [#vdW-DF3]_. This method is implemented in
two different versions. The differences are related to how a six
dimentional integral is calculated where one method (real space
method) solves this integral in real space, while the other method
solves it by approximating the integral as a convolution which can be
Fourier Transformed (FFT method) [#soler]_.



Doing a van der Waals calculation 
==================================

The FFT method is up to 1000 times faster for big systems which
probably makes it the weapon of choice for every GPAW vdW enthusiast

The vdW-DF is implemented self consitently in GPAW. In many cases the
density is hardly affected by van der Waals forces and hence it can
make sense to add vdW forces to a self consistent GGA calculation.


Perturbative vdW-DF calculation (Non self consistent) 
-----------------------------------------------------
  
>>> from gpaw import GPAW
>>> from gpaw.vdw import FFTVDWFunctional
>>> vdw = FFTVDWFunctional(nspins=1,
...                        Nalpha=20, lambd=1.2, 
...                        rcut=125.0, Nr=2048, 
...                        Verbose=True,
...                        size=None) 
>>> calc = GPAW('input.gpw') 
>>> GGA_energy = calc.get_potential_energy()
>>> vdW_dif = calc.get_xc_difference(vdw)
>>> vdW_energy = GGA_energy + vdW_dif 

For self consistent vdW-DF calculations one should instead simply
change XC functional to the vdW-DF. This can either be done in two
ways.


Self Consistent vdW-DF Calculation
----------------------------------

>>> from ase.all import *
>>> from gpaw import GPAW
>>> from gpaw.vdw import FFTVDWFunctional
>>> vdw = FFTVDWFunctional(nspins=1, verbose=True)
>>> atoms = ...
>>> calc = GPAW(xc=vdw, ...)
>>> atoms.set_calculator(calc)
>>> e = calc.get_potential_energy()

This is not quite the conventional method since we are now using an
object and not a string. If one wants to use all the default settings
it is also possible to use ``xc='vdW-DF'`` withouth having to import any
vdW object. Another function is that it is now possible to extend the
cell with empty space when doing a non self consistent vdW-DF
calculation. This is practical since vdW interactions are long range
which would other wise have resulted in bigger cells and more
demanding GGA calculations. This is referred to as zero padding. This
is controlled by the 'size' argument and it only works for non
periodic systems. The zero padded cell should be a number dividable
with 4. The calculations then adds zeros to the points not included in
the GGA calculation.
 

Real space method vdW-DF Calculation
------------------------------------

It is also possible to use the slower real space method, which could
make sense for smaller systems. This method is not self consistent and
can only be used in the perturbative method described above. To use
the real space method one changes the following lines from above:

>>> from gpaw.vdw import RealSpaceVDWFunctional
>>> vdw = RealSpaceVDWFunctional(nspins=1, ncut=0.0005)





.. [#vdW-DF] M. Dion, H. Rydberg, E. Schroder, D.C. Langreth, and
   B. I. Lundqvist.  Van der Waals density functional for
   general geometries.  Physical Review Letters, 92, 246401 (2004)

.. [#vdW-DF2] M. Dion, H. Rydberg, E. Schroder, D.C. Langreth, and
   B. I. Lundqvist.  Erratum: Van der Waals density functional for
   general geometries.  Physical Review Letters, 95, 109902 (2005)

.. [#vdW-DF3] T. Thonhauser, V.R. Cooper, S. Li, A. Puzder,
   P. Hyldgaard, and D.C. Langreth. Van der Waals density functional:
   Self-consistent potential and the nature of the van der Waals bond.
   Phys. Rev. B 76, 125112 (2007)

.. [#soler] Guillermo Román-Pérez and José M. Soler.
   Efficient Implementation of a van der Waals Density Functional: Application
   to Double-Wall Carbon Nanotubes.
   Phys. Rev. Lett. 103, 096102 (2009)
