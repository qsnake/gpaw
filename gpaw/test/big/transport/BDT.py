from gpaw import *
from ase import *
from ase.lattice.surface import fcc111
from gpaw.transport.calculator import Transport
from gpaw.mpi import world
import pickle
from mytools import *

system = read('BDT.traj', -1)
pl_atoms1 = range(36)     
pl_atoms2 = range(72,108) 
pl_cell1 = np.diag(system.cell)
pl_cell1[2] = 7.067
pl_cell2 = pl_cell1      

t = Transport( h=0.2,     
               xc='PBE',
               basis={'Au': 'sz', 'H': 'dzp', 'C': 'dzp', 'S':'dzp'},
               kpts=(4, 4, 1),
               parsize=(1,1,2),
               occupations=FermiDirac(0.1),
               mode='lcao',
               txt='ABA.txt',
               poissonsolver=PoissonSolver(nn=2),
               mixer=Mixer(0.1, 5, weight=100.0),
               pl_atoms=[pl_atoms1, pl_atoms2],
               pl_cells=[pl_cell1, pl_cell2],
               pl_kpts=[4, 4, 15],
               extra_density=True,
               analysis_data_list=['tc'],
               edge_atoms=[[ 0, 35],[0 , 107]],
               mol_atoms=range(36, 72),
               nleadlayers=[1,1])
system.set_calculator(t)
t.calculate_iv(3., 16)

