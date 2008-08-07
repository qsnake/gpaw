import Numeric as na
from ASE.Utilities.BEE import GetEnsembleEnergies
from gpaw import GPAW 

atom = GPAW('H.gpw', txt=None).get_atoms()
molecule = GPAW('H2.gpw', txt=None).get_atoms()
e1 = atom.GetPotentialEnergy()
e2 = molecule.GetPotentialEnergy()
ea = 2 * e1 - e2
print 'PBE:', ea, 'eV'

e1i = GetEnsembleEnergies(atom)
e2i = GetEnsembleEnergies(molecule)
eai = 2 * e1i - e2i

n = len(eai)
ea0 = na.sum(eai) / n
sigma = (na.sum((eai - ea0)**2) / n)**0.5
print 'Best fit:', ea0, '+-', sigma, 'eV'


f = open('ensemble.dat', 'w')
for e in eai:
    print >> f, e
