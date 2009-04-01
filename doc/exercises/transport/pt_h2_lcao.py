from ase import *
from gpaw import *
from gpaw.lcao.gpawtransport import GPAWTransport 

a = 2.41 # Pt binding lenght
b = 0.90 # H2 binding lenght
c = 1.70 # Pt-H binding lenght
L = 7.00 # width of unit cell

# Setup the Atoms for the scattering region.
atoms = Atoms('Pt5H2Pt5', pbc=(1, 0, 0), cell=[9 * a + b + 2 * c, L, L])
atoms.positions[:5, 0] = [i * a for i in range(5)]
atoms.positions[-5:, 0] = [i * a + b + 2 * c for i in range(4, 9)]
atoms.positions[5:7, 0] = [4 * a + c, 4 * a + c + b]
atoms.positions[:, 1:] = L / 2.

# Attach a GPAW calculator
atoms.set_calculator(GPAW(h=0.3, xc='PBE', basis='szp', width=0.1, kpts=(1,1,1),
                          mode='lcao', txt='pt_h2_lcao.txt',
                          mixer=Mixer(0.1, 5, metric='new', weight=100.0)))

# Setup the GPAWTransport calculator
pl_atoms1 = range(4)     # Atomic indices of the left principal layer
pl_atoms2 = range(-4, 0) # Atomic indices of the right principal layer
pl_cell1 = (4 * a, L, L) # Cell for the left principal layer
pl_cell2 = pl_cell1      # Cell for the right principal layer

gpawtran = GPAWTransport(atoms=atoms,
                         pl_atoms=(pl_atoms1, pl_atoms2),
                         pl_cells=(pl_cell1, pl_cell2),
                         d=0) #transport direction (0 := x)

# Dump the Hamiltonian matrices to the files:
# lead1_hs.pickle, lead2_hs.pickle and scat_hs.pickle
gpawtran.write('hs.pickle')
