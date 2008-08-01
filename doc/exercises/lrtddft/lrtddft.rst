.. _exercise_lrtddft:

==============================
Calculation of optical spectra
==============================

In this exercise we calculate optical spectrum of Na2 molecule using
linear response time-dependent density functional theory. We start
with a normal ground state calculation:

Linear response TDDFT needs unoccupied states, so we calculate 19 of
them in addition to the one occupied state. Note that in realistic
calculation there should be more vacuum around the molecule. One might
also want to first :ref:`optimize the geometry
<structure_optimization>`, save the structure::

  calc.write('na2_gs.gpw')

and start a then a new calculation with more vacuum and unoccupied states::

  atoms, calc = restart('na2_gs.gpw')
  atoms.center(vacuum=6.0)
  calc.set(nbands=20, ...)
  ...

Once the ground state calculation with unoccupied states is finished, a linear response TDDFT calculation is performed:


As the construction of the Omega matrix is computationally the most intensive part it is sometimes convenient to
perform diagonalisation and construction of spectrum in separate calculations::

  lr = LrTDDFT(filename='Omega_Na2.gz')
  lr.diagonalize()
  lr.write('excitations_Na2.gz')

and::
  
  lr = LrTDDFT(filename='excitations_Na2.gz')
  photoabsorption_spectrum(lr, 'Na2_spectrum.dat', e_min=0.0, e_max=10)

The number of electron-hole pairs used in the calculation can be controlled with 
``istart`` and ``jend`` options of LrTDDFT::

  LrTDDFT(calc, istart=0, jend=10)

By default only singlet-singlet transitions are calculated, singlet-triplet transitions can be calculated by giving the ``nspins`` parameter::

  LrTDDFT(calc, istart=0, jend=10, nspins=2)
  
Exercises
--------- 

1. Check how the results vary with the number of unoccupied states in calculation (``jend`` parameter).

2. Calculate also singlet-triplet transitions. Why they do not show up in spectrum?

3. Check how the results vary with the empty space around the molecule.

4. Try to calculate optical spectrum also with the :ref:`timepropagation` approach and see how the results compare to linear response calculation.

