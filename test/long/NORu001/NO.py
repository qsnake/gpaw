from gpaw import *
from ase import *

for name in ['N2', 'O2', 'NO']:
    mol = molecule(name)
    mol.center(vacuum=5.0)
    calc = GPAW(xc='PBE',
                h=0.2,
                stencils=(3, 3),
                txt=name + '.txt')
    mol.set_calculator(calc)
  
    opt = HessLBFGS(mol, logfile=name + '.log', trajectory=name + '.traj')
    opt.run(fmax=0.05)
