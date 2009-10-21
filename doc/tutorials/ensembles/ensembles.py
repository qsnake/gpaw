import numpy as np
from ase.utils.bee import get_ensemble_energies
from gpaw import GPAW 

atom = GPAW('H.gpw', txt=None).get_atoms()
molecule = GPAW('H2.gpw', txt=None).get_atoms()
e1 = atom.get_potential_energy()
e2 = molecule.get_potential_energy()
ea = 2 * e1 - e2
print 'PBE:', ea, 'eV'

e1i = get_ensemble_energies(atom)
e2i = get_ensemble_energies(molecule)
eai = 2 * e1i - e2i

n = len(eai)
ea0 = np.sum(eai) / n
sigma = (np.sum((eai - ea0)**2) / n)**0.5
print 'Best fit:', ea0, '+-', sigma, 'eV'


f = open('ensemble.dat', 'w')
for e in eai:
    print >> f, e
