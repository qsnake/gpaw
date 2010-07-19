import numpy as np
from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from gpaw.test import equal

a = 5.475
calc = GPAW(h=0.24,
            kpts=(4, 4, 4),
            occupations=FermiDirac(width=0.0),
            nbands=5)
atoms = bulk('Si', 'diamond', a=a)
atoms.set_calculator(calc)
E = atoms.get_potential_energy()
equal(E, -11.8092858, 0.0001)
niter = calc.get_number_of_iterations()
assert 25 <= niter <= 27, niter

equal(atoms.calc.get_fermi_level(), 5.17751284, 0.005)
homo, lumo = calc.get_homo_lumo()
equal(lumo - homo, 1.11445025, 0.001)
