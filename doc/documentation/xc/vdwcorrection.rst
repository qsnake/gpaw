.. _vdwcorrection:

========================
van der Waals correction
========================

A correction on top of the PBE functional has been proposed
by Tkachenko and Scheffler [#TS09]_. While nearly all parameters
are obtained from ab-initio calculations, the method requires
nearly no additional computational cost and performs very well:

============== ===  ===== ====== ====
?              PBE  TPSS  vdW-DF TS09
============== ===  ===== ====== ====
Mean deviation 116  155   83     16
RMS deviation  111  131   64     18
============== ===  ===== ====== ====

Error in energies compared to CCSD results of the S26 test set.
All values in meV.
GPAW calculations were done with h=0.18 and at least 4 A vacuum.

Calculating the S26 test set 
============================

As an example of the usage, here the S26 test set is calculated:

>>> import sys
>>> from ase import *
>>> from ase.parallel import paropen
>>> from ase.data.s22 import data, s22
>>> from ase.calculators.vdwcorrection import vdWTkatchenko09prl
>>> from gpaw import *
>>> from gpaw.cluster import Cluster
>>> from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning
>>> from gpaw.analyse.vdwradii import vdWradii
>>> h = 0.18
>>> box = 4.
>>> xc = 'TS09'
>>> f = paropen('energies_' + xc +'.dat', 'w')
>>> print  >> f, '# h=', h
>>> print  >> f, '# box=', box
>>> print >> f, '# molecule E[1]  E[2]  E[1+2]  E[1]+E[2]-E[1+2]'
>>> for molecule in data:
>>>     print >> f, molecule,
>>>     ss = Cluster(Atoms(data[molecule]['symbols'], 
...                       data[molecule]['positions']))
>>>     # split the structures
>>>     s1 = ss.find_connected(0)
>>>     s2 = ss.find_connected(-1)
>>>     assert(len(ss) == len(s1) + len(s2))
>>>     if xc == 'TS09' or xc == 'TPSS' or xc == 'M06L':
>>>         c = GPAW(xc='PBE', h=h, nbands=-6, occupations=FermiDirac(width=0.1))
>>>     else:
>>>         c = GPAW(xc=xc, h=h, nbands=-6, occupations=FermiDirac(width=0.1))
>>>     E = []
>>>     for s in [s1, s2, ss]:
>>>         s.set_calculator(c)
>>>         s.minimal_box(box, h=h)
>>>         if xc == 'TS09':
>>>             s.get_potential_energy()
>>>             cc = vdWTkatchenko09prl(HirshfeldPartitioning(c),
...                                     vdWradii(s.get_chemical_symbols(), 'PBE'))
>>>             s.set_calculator(cc)
>>>         if xc == 'TPSS' or xc == 'M06L':
>>>             ene = s.get_potential_energy()
>>>             ene += c.get_xc_difference(xc)
>>>             E.append(ene)
>>>         else:
>>>             E.append(s.get_potential_energy())
>>>     print >> f, E[0], E[1], E[2],
>>>     print >> f, E[0] + E[1] - E[2]
>>>     f.flush()
>>> f.close()

.. [#TS09] Tkachenko and Scheffler Phys. Rev. Lett. 102 (2009) 073005

