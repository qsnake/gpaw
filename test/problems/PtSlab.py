from ase import *
from ase.lattice.surface import fcc111
from gpaw import GPAW
from gpaw.mixer import Mixer

atoms = fcc111('Pt', orthogonal = True, size = (2, 2, 8), a = 4.00, vacuum = 12.)
atoms.center(axis = 2)
atoms.set_pbc((True, True, False))

mask = [at.tag == 8 for at in atoms]
atoms.set_constraint(FixAtoms(mask = mask))

view(atoms)
    
calc = GPAW(nbands = -35, h = 0.15, kpts = (4, 4, 1), txt = 'PtSlab.txt', xc = 'RPBE', stencils = (3, 3))
atoms.set_calculator(calc)

dyn = QuasiNewton(atoms, logfile = 'PtSlab.log', restart = 'PtSlab.pickle', trajectory = 'PtSlab.traj')
dyn.attach(calc.write, 1, 'PtSlab.gpw')
dyn.run(fmax = 0.05)

calc.write('PtSlab.gpw')

