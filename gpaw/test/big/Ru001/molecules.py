from gpaw import *
from ase import *

for name in ['H2', 'N2', 'O2', 'NO']:
    mol = molecule(name)
    mol.center(vacuum=5.0)
    if name == 'NO':
        mol.translate((0, 0.1, 0))
    calc = GPAW(xc='PBE',
                h=0.2,
                stencils=(3, 3),
                txt=name + '.txt')
    mol.set_calculator(calc)
  
    opt = QuasiNewton(mol, logfile=name + '.log', trajectory=name + '.traj')
    opt.run(fmax=0.05)
    calc.write(name)
