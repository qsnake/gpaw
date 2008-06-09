from ase import *
from gpaw import *
from gpaw.utilities import equal


atoms = Atoms([Atom('H', magmom=1.0)], cell=(4, 4, 4), pbc=1)
conv = {'eigenstates': 1e-2, 'energy':2e-1, 'density':1e-1}
calc = Calculator(nbands=1, gpts=(20, 20, 20), convergence=conv)# out=None)
atoms.set_calculator(calc)
e = atoms.get_potential_energy()
calc.set(gpts=(16,16,16))
e = atoms.get_potential_energy()
calc.set(xc='PBE')
e = atoms.get_potential_energy()
calc.set(nbands=4)
e = atoms.get_potential_energy()

