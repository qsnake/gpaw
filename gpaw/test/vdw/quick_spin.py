import os
from ase import Atoms
from gpaw import GPAW, FermiDirac
from ase.parallel import rank, barrier
from gpaw.xc.vdw import FFTVDWFunctional
from gpaw.test import gen

# Generate setup
gen('H', xcname='revPBE')

L = 2.5
a = Atoms('H', cell=(L, L, L), pbc=True)
calc = GPAW(xc='vdW-DF',
            occupations=FermiDirac(width=0.001),
            txt='H.vdw-DF.txt')
a.set_calculator(calc)
e1 = a.get_potential_energy()

calc.set(txt='H.vdw-DF.spinpol.txt',
         spinpol=True,
         occupations=FermiDirac(width=0.001, fixmagmom=True))
e2 = a.get_potential_energy()

assert abs(calc.get_eigenvalues(spin=0)[0] -
           calc.get_eigenvalues(spin=1)[0]) < 1e-10

print e1-e2
assert abs(e1 - e2) < 1e-10

vdw = FFTVDWFunctional('vdW-DF')
calc = GPAW(xc=vdw, width=0.001,
            txt='H.vdw-DFb.txt')
a.set_calculator(calc)
e3 = a.get_potential_energy()
assert abs(e1 - e3) < 1e-12
