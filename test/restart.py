import os
from gpaw import *
from ase import *
from gpaw.utilities import equal

d = 3.0
atoms = Atoms('Na3', positions=[( 0, 0, 0),
                              ( 0, 0, d),
                              ( 0, d*sqrt(3./4.), d/2.)],
                   magmoms=[1.0, 1.0, 1.0],
                   cell=(3.5, 3.5, 3.5),
                   pbc=True)

# Only a short, non-converged calcuation
conv = {'eigenstates': 1e-2, 'energy':2e-1, 'density':1e-1}
calc = Calculator(h=0.35, nbands=3, convergence=conv)
atoms.set_calculator(calc)
e0 = atoms.get_potential_energy()
f0 = atoms.get_forces()
m0 = atoms.get_magnetic_moments()
eig00 = calc.get_eigenvalues(spin=0)
eig01 = calc.get_eigenvalues(spin=1)
calc.write('tmp.gpw')
del atoms, calc
atoms, calc = restart('tmp.gpw')
e1 = atoms.get_potential_energy()
f1 = atoms.get_forces()
m1 = atoms.get_magnetic_moments()
eig10 = calc.get_eigenvalues(spin=0)
eig11 = calc.get_eigenvalues(spin=1)
# print e0, e1
equal(e0, e1, 1e-10)
# print f0, f1
for ff0, ff1 in zip(f0, f1):
    err = npy.linalg.norm(ff0-ff1)
    assert err <= 1e-10
# print m0, m1
for mm0, mm1 in zip(m0, m1):
    equal(mm0, mm1, 1e-10)
for eig0, eig1 in zip(eig00, eig10):
    equal(eig0, eig1, 1e-10)
for eig0, eig1 in zip(eig01, eig11):
    equal(eig0, eig1, 1e-10)

os.remove('tmp.gpw')
