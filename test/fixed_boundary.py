from ase import *
from gpaw import *
from gpaw.atom.basis import BasisMaker
from gpaw.transport.calculator import Transport 

a = 3.6
L = 7.0

basis = BasisMaker('Na').generate(1, 1, energysplit=0.3)

atoms = Atoms('Na4', pbc=(0, 0, 0), cell=[L, L, 4*a])
atoms.positions[:4, 2] = [i * a for i in range(4)]
atoms.positions[:, :2] = L / 2.
atoms.center()

pl_atoms1 = range(4)     
pl_atoms2 = range(4) 
pl_cell1 = (L, L, 4*a) 
pl_cell2 = pl_cell1      

t = Transport( h=0.2,     
               xc='LDA',
               basis={'Na': basis},
               kpts=(1,1,1),
               width=0.1,
               mode='lcao',
               txt='Na_lcao.txt',
               usesymm=False,
               spinpol=False,
               identical_leads=False,
               poissonsolver=PoissonSolver(nn=2),
               mixer=Mixer(0.1, 5, weight=100.0),
               pl_atoms=[pl_atoms1, pl_atoms2],
               pl_cells=[pl_cell1, pl_cell2],
               pl_kpts=[1,1,5],
               fixed_boundary=True,
               save_file=False,
               bias=[0, 0],
               gate=0,
               edge_atoms=[[0,3],[0,3]]) 

atoms.set_calculator(t)
t.calculate_iv(0, 1)


