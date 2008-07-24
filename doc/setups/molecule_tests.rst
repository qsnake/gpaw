.. _molecule_tests:

==============
Molecule tests
==============

Atomization energies (*E*\ `a`:sub:) and bond lengths (*d*) for 20
small molecules calculated with the PBE functional.  All calculations
are done in a box of size 12.6 x 12.0 x 11.4 Å with a grid spacing of
*h*\ =0.16 Å and zero-boundary conditions.  Compensation charges are
expanded with correct multipole moments up to *l*\ `max`:sub:\ =2.
Open-shell atoms are treated as non-spherical with integer occupation
numbers, and zero-point energy is not included in the atomization
energies. The numbers are compared to very accurate, state-of-the-art,
PBE calculations (*ref* subscripts).

.. figure:: molecules.png
   

Bond lengths and atomization energies at relaxed geometries
===========================================================

(*rlx* subscript)

.. list-table::
   :widths: 2 3 8 5 6 8

   * -
     - *d* [Å]
     - *d*-*d*\ `ref`:sub: [Å]
     - *E*\ `a,rlx`:sub: [eV]
     - *E*\ `a,rlx`:sub:-*E*\ `a`:sub: [eV]
     - *E*\ `a,rlx`:sub:-*E*\ `a,rlx,ref`:sub: [eV] [1]_
   * - H\ `2`:sub:\ 
     - 0.751
     - +0.000 [4]_
     -  4.539
     -  0.002
     -
   * - LiH
     - 1.619
     - +0.015 [1]_
     -  2.321
     -  0.002
     -  0.001
   * - HF
     - 0.935
     - 
     -  6.145
     -  0.010
     -  0.009
   * - Li\ `2`:sub:\ 
     - 2.728
     - +0.000 [1]_
     -  0.858
     -  0.002
     - -0.005
   * - LiF
     - 1.638
     - +0.055 [1]_
     -  5.866
     -  0.064
     - -0.135
   * - Be\ `2`:sub:\ 
     - 2.435
     - 
     -  0.421
     -  0.000
     -
   * - CO
     - 1.136
     - -0.000 [1]_
     - 11.713
     -  0.003
     -  0.065
   * - N\ `2`:sub:\ 
     - 1.102
     - -0.001 [1]_, -0.001 [4]_
     - 10.619
     -  0.001
     -  0.052
   * - NO
     - 1.158
     - -0.000 [4]_
     -  7.482
     -  0.003
     -  0.023
   * - O\ `2`:sub:\ 
     - 1.221
     - +0.003 [1]_, +0.001 [4]_
     -  6.230
     -  0.007
     -  0.016
   * - F\ `2`:sub:\ 
     - 1.415
     - +0.001 [1]_, +0.001 [4]_
     -  2.306
     -  0.000
     -  0.025
   * - P\ `2`:sub:\ 
     - 1.912
     - 
     -  5.194
     -  0.006
     - -0.074
   * - Cl\ `2`:sub:\ 
     - 2.008
     - +0.009 [1]_
     -  2.837
     -  0.004
     - -0.017

Atomization energies at experimental geometries
===============================================

.. list-table::
   :widths: 6 6 12

   * -
     - *E*\ `a`:sub: [eV]
     - *E*\ `a`:sub:-*E*\ `a,ref`:sub: [eV]
   * - H\ `2`:sub:\ 
     -  4.537
     - +0.002 [2]_, +0.006 [3]_
   * - LiH
     -  2.319
     - -0.001 [2]_, -0.001 [3]_
   * - CH\ `4`:sub:\ 
     - 18.257
     - +0.053 [2]_, +0.079 [3]_
   * - NH\ `3`:sub:\ 
     - 13.119
     - +0.036 [2]_, +0.066 [3]_
   * - OH
     -
     -
   * - H\ `2`:sub:\ O
     - 10.150
     - -0.006 [2]_, +0.012 [3]_
   * - HF
     -  6.135
     - -0.023 [2]_, -0.010 [3]_
   * - Li\ `2`:sub:\ 
     -  0.856
     - -0.007 [2]_, +0.002 [3]_
   * - LiF
     -  5.803
     - -0.208 [2]_, -0.247 [3]_
   * - Be\ `2`:sub:\ 
     -  0.421
     - -0.004 [2]_, +0.009 [3]_
   * - C\ `2`:sub:\ H\ `2`:sub:\ 
     - 18.113
     - +0.122 [2]_, +0.208 [3]_
   * - C\ `2`:sub:\ H\ `4`:sub:\ 
     - 24.896
     - +0.113 [2]_, +0.170 [3]_
   * - HCN
     - 14.250
     - +0.109 [2]_, +0.178 [3]_
   * - CO
     - 11.710
     - +0.053 [2]_, +0.105 [3]_
   * - N\ `2`:sub:\ 
     - 10.618
     - +0.072 [2]_, +0.159 [3]_
   * - NO
     -  7.479
     - +0.025 [2]_, +0.120 [3]_
   * - O\ `2`:sub:\ 
     -  6.222
     - -0.009 [2]_, +0.078 [3]_
   * - F\ `2`:sub:\ 
     -  2.306
     - -0.010 [2]_, +0.055 [3]_
   * - P\ `2`:sub:\ 
     -  5.188
     - -0.063 [2]_, +0.106 [3]_
   * - Cl\ `2`:sub:\ 
     -  2.833
     - +0.010 [2]_, +0.096 [3]_


References
==========

.. [1] "The Perdew-Burke-Ernzerhof exchange-correlation functional
       applied to the G2-1 test set using a plane-wave basis set",
       J. Paier, R. Hirschl, M. Marsman and G. Kresse,
       J. Chem. Phys. 122, 234102 (2005)

.. [2] "Molecular and Solid State Tests of Density Functional
       Approximations: LSD, GGAs, and Meta-GGAs", S. Kurth,
       J. P. Perdew and P. Blaha, Int. J. Quant. Chem. 75, 889-909
       (1999)

.. [3] "Comment on 'Generalized Gradient Approximation Made Simple'",
       Y. Zhang and W. Yang, Phys. Rev. Lett.

.. [4] Reply to [3]_, J. P. Perdew, K. Burke and M. Ernzerhof


