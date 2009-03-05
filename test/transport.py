from ase import *
from gpaw import *
from gpaw.lcao.gpawtransport import GPAWTransport 
from gpaw.atom.basis import BasisMaker
import pickle

a = 3.6
L = 7.00 

#basis = BasisMaker('Na').generate(1, 1)

atoms = Atoms('Na4', pbc=(1, 1, 1), cell=[L, L, 4 * a])
atoms.positions[:4, 2] = [i * a for i in range(4)]
atoms.positions[:, :2] = L / 2.
atoms.center()
atoms.set_calculator(GPAW(h=0.3,
                          xc='PBE',
                          #basis={'Na': basis},
                          basis='szp',
                          kpts=(1,1,3),
                          width=0.01,
                          mode='lcao',
                          txt='Na_lcao.txt',
                          usesymm=None,
                          mixer=Mixer(0.1, 5, metric='new', weight=100.0)))
pl_atoms1 = range(4)     
pl_atoms2 = range(-4, 0) 
pl_cell1 = (L, L, 4 * a) 
pl_cell2 = pl_cell1      

gpawtran = GPAWTransport(atoms=atoms,
                         pl_atoms=(pl_atoms1, pl_atoms2),
                         pl_cells=(pl_cell1, pl_cell2),
                         d=2) 
gpawtran.negf_prepare(scat_restart=False, lead_restart=True)
gpawtran.get_selfconsistent_hamiltonian(bias=0, gate=0,verbose=1)
filename = 'Na4_eq'
gpawtran.output(filename)
