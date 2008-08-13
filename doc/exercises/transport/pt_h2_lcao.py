# creates: pt_h2.png

from ase import *
from gpaw import GPAW, Mixer
from gpaw.lcao.gpawtransport import GPAWTransport 
import pickle


a = 2.41
b = 0.9
c = 1.7
L = 7
l = L * 0.5
N = 5

#Setup the Atoms for the scattering region
#The left lead principal layer atoms must be
#the first in the Atoms list. The right lead principal
#layer atoms must be the last.
atoms1 = Atoms('Pt',[(0.0, l, l)],pbc=True,cell=(a, L, L))
atoms1 *= (N, 1, 1)
atoms2 = atoms1.copy()
atoms2.translate(((N - 1) * a + b + 2 * c, 0, 0))
h2 = Atoms('H2',[((N - 1) * a + c, l, l),(( N - 1) * a + c + b, l, l)],pbc=True)
atoms = atoms1 + h2 + atoms2
atoms.set_cell([(2 * N - 1)* a + b + 2 * c, L, L])

mixer = Mixer(0.1, 5, metric='new', weight=100.0)
calc = GPAW(h=0.3, xc='PBE', width=0.1, basis='szp', eigensolver='lcao',
            txt='pt_h2_lcao.txt', mixer=mixer)

atoms.set_calculator(calc)
atoms.get_potential_energy()

#Setup the GPAWTransport calculator
pl_atoms1 = range(4) #The left principal layer atoms indices
pl_atoms2 = range(-4,0) #The right principal layer atoms indices
pl_cell1 = (len(pl_atoms1)*a,L,L) #Cell for the left principal layer
pl_cell2 = pl_cell1 #Cell for the right principal layer

gpawtran = GPAWTransport(atoms=atoms,
                         pl_atoms=(pl_atoms1,pl_atoms2),
                         pl_cells=(pl_cell1,pl_cell2),
                         d=0) #transport direction (d=0->x)

#Will the pickle files: lead1_hs.pickle, lead2_hs.pickle and scat_hs.pickle
gpawtran.write('hs.pickle')
#The files can be read using
