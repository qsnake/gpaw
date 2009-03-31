.. _vdw:

Introduction 
------------ 

There is one method of including van der Waals forces in GPAW
[#vdW-DF]_. This method is implemented in two different versions. The
differences are related to how a six dimentional integral is
calculated where one method (real space method) solves this integral
in real space [#vdW-DF]_, while the other method solves it by
approximating the integral as a convolution which can be Fourier
Transformed (FFT method) [#soler]_. Both methods are using tables in
order to speed up calculations. In the FFT method the function will
look for an existing table with desired setups and generate one if it
can not find a table. This requires scipy. A table with some default
setups can be downloaded here: phi-0.500-1.000-20.000-21-201.pckl_.

.. _phi-0.500-1.000-20.000-21-201.pckl: http://wiki.fysik.dtu.dk/gpaw-files/phi-0.500-1.000-20.000-21-201.pckl

The real space methods table can be downloaded here phi_. Download this an unpack it in a folder
name e.g VDW and refer to it as setenv VDW /home/user/Phi/. The
provided interaction kernal phi has been calculated using the scripts
which can be downloaded here: makephi_


---------------------------------
Doing a van der Waals calculation 
---------------------------------
The FFT method is up to 1000 times faster for big systems which probably makes it the weapon of choice for every GPAW vdW enthusiast

The vdW-DF is implemented self consitently in GPAW. In many cases the density is hardly affected by van der Waals forces and hence it can make sense to add vdW forces to a self consistent GGA calculation . This method is possible to run parallell, with spins. This is parallellized over Nalpha, in the default case this is 20. For systems when the vdW calculation is small it can still make sense to use more than 20 CPUs since revPBE exchange and LDA correlation still can be parallellized on more CPUs.

Perturbative vdW-DF calculation (Non self consistent) 
-----------------------------------------------------
  
>>> from ase import *
>>> from gpaw import *
>>> from gpaw.vdw import FFTVDWFunctional
>>> vdw = FFTVDWFunctional(nspins=1,
                           Nalpha=20, lambd=1.2, 
                           rcut=125.0, Nr=2048, 
                           Verbose=True,
                           size=None) 
>>> calc = GPAW(input.gpw) 
>>> GGA_energy = calc.get_potential_energy()
>>> vdW_dif = calc.get_xc_difference(vdw)
>>> vdW_energy = GGA_energy + vdW_dif 

For self consistent vdW-DF calculations one should instead simply change XC functional to the vdW-DF. This can either be done in two ways.


Self Consistent vdW-DF Calculation
----------------------------------

>>> from ase import *
>>> from gpaw import *
>>> from gpaw.vdw import FFTVDWFunctional
>>> vdw = FFTVDWFunctional(nspins=1,Verbose=True)
>>> atoms=Atoms('H'[(,0,0,0)],cell=(1,1,1),pbc=True)
>>> calc=GPAW(xc=vdw,h=0.2,txt='H.txt')
>>> atoms.set_calculator(calc)
>>> e=calc.get_potential_energy()

This is not quite the conventional method since we are now using an object and not a string. If one wants to use all the default settings it is also possible to use xc='vdW-DF' withouth having to import any vdW object. Another function is that it is now possible to extend the cell with empty space when doing a non self consistent vdW-DF calculation. This is practical since vdW interactions are long range which  would other wise have resulted in bigger cells and more demanding GGA calculations. This is referred to as zero padding. This is controlled by the 'size' argument and it only works for non periodic systems. The zero padded  cell should be a number dividable with 4. The calculations then adds zeros to the points not included in the GGA calculation.  
 
It is also possible to use the slower real space method, which could make sense for smaller systems. This method is not self consistent and can only be used in the perturbative method described above. To use the real space method one changes the following lines from above:


Real space method vdW-DF Calculation
------------------------------------

>>> from gpaw.vdw import RealSpaceVDWFunctional
>>> vdw=RealSpaceVDWFunctional(nspins=1, repeat=None, ncut=0.0005)





.. [#vdW-DF] M. Dion, H. Rydberg, E. Schroder, D.C. Langreth, and
                B. I. Lundqvist.  Van der Waals density functional for
                general geometries.  Physical Review Letters, 92
                (24):246401. 2004

.. [#soler] Guillermo Román-Pérez and José M. Soler 

.. _phi: ../../_static/phi.dat

.. _makephi: ../../_static/makephi.tar.gz
