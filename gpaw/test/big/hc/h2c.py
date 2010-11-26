"""Hydrogen chain test.

See:

    S. Suhai: *Electron correlation in extended systems: Fourth-order
    many-body perturbation theory and density-functional methods
    applied to an infinite chain of hydrogen atoms*, Phys. Rev. B 50,
    14791-14801 (1994)
    
"""

import sys

import numpy as np
from ase import Atoms

from gpaw.xc.hybridk import HybridXC, HybridRMMDIIS
from gpaw import GPAW, FermiDirac

k = int(sys.argv[1])
h = float(sys.argv[2])
alpha = float(sys.argv[3])

a = 6.0
d0 = 1.0

h2 = Atoms('H2', cell=(2 * d0, a, a), pbc=True,
           positions=[(0, 0, 0), (d0 + 0.1, 0, 0)])
h2.center(axis=1)
h2.center(axis=2)

calc = GPAW(kpts=(k, 1, 1),
            usesymm=None,
            nbands=1,
            h=h,
            setups='hgh',
            occupations=FermiDirac(0.0),
            txt='h2c-%02d-%.3f-%.1f.txt' % (k, h, alpha))
h2.calc = calc

e = h2.get_potential_energy()

calc.set(xc=HybridXC('EXX', alpha=alpha),
         eigensolver=HybridRMMDIIS())

for d in np.linspace(0.1, 0.4, 20):
    h2[1].x = d0 + d
    e = h2.get_potential_energy()
    v0 = calc.hamiltonian.vHt_g[:, 0, 0].mean()
    calc.text('EIGS:', [kpt.eps_n[0] - v0 for kpt in calc.wfs.kpt_u])
    calc.text('GAMMA:', calc.hamiltonian.xc.gamma)
    calc.text('ALPHA:', calc.hamiltonian.xc.alpha)
