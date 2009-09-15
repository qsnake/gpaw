from ase import *
from gpaw import *
from gpaw.transport.calculator import Transport

a = 2.41 # Pt binding lenght
b = 0.90 # H2 binding lenght
c = 1.70 # Pt-H binding lenght
L = 7.00 # width of unit cell

# Setup the Atoms for the scattering region.
atoms = Atoms('Pt5H2Pt5', pbc=(1, 1, 1), cell=[L, L, 9 * a + b + 2 * c])
atoms.positions[:5, 0] = [i * a for i in range(5)]
atoms.positions[-5:, 0] = [i * a + b + 2 * c for i in range(4, 9)]
atoms.positions[5:7, 0] = [4 * a + c, 4 * a + c + b]
atoms.positions[:, 1:] = L / 2.
pl_atoms1 = range(4)     # Atomic indices of the left principal layer
pl_atoms2 = range(8, 12) # Atomic indices of the right principal layer
pl_cell1 = (L, L, 4*a) # Cell for the left principal layer
pl_cell2 = pl_cell1      # Cell for the right principal layer

atoms.rotate('x', 'z')
# Attach a GPAW calculator
atoms.set_calculator(
                 Transport(h=0.3, 
                           xc='PBE', 
                           basis='szp', 
                           width=0.1, 
                           kpts=(1,1,1),
                           mode='lcao', 
                           save_file=False,
                           txt='pt_h2_transport.txt',
                           mixer=Mixer(0.1, 5, weight=100.0), 
                           pl_atoms=[pl_atoms1, pl_atoms2],
                           pl_cells=[pl_cell1, pl_cell2], 
                           pl_kpts=(1,1,13), 
                           mol_atoms=range(4, 8),
                           lead_restart=False,
                           scat_restart=False,
                           edge_atoms=[[0, 3],[0,11]], 
                           non_sc=True))

atoms.calc.negf_prepare()
atoms.calc.non_sc_analysis()
