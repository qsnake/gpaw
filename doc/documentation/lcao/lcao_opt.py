from ase import *
from gpaw import GPAW
from gpaw.poisson import PoissonSolver

a = 6
b = a / 2
mol = Atoms([Atom('O',(b, b, 0.1219 + b)),
             Atom('H',(b, 0.7633 + b, -0.4876 + b)),
             Atom('H',(b, -0.7633 + b, -0.4876 + b))],
            pbc=False, cell=[a, a, a])

calc = GPAW(nbands=4, mode='lcao', basis='dzp',
            gpts=(32, 32, 32),
            poissonsolver=PoissonSolver(relax='GS', eps=1e-7))

mol.set_calculator(calc)
dyn = QuasiNewton(mol, trajectory='lcao2_h2o.traj')
dyn.run(fmax=0.05)
