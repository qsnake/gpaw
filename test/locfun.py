import numpy as np
#np.seterr(all='raise')
#np.set_printoptions(precision=3, suppress=True)

from ase import *
from gpaw import *
from gpaw.coulomb import get_vxc
from gpaw.wannier import LocFun

if 1:
    calc = GPAW(nbands=9)
    atoms = molecule('H2O', calculator=calc)
    atoms.center(vacuum=2.4)
    atoms.get_potential_energy()
    calc.write('H2O.gpw', mode='all')

atoms, calc = restart('H2O.gpw', txt=None)

locfun = LocFun()
locfun.localize(calc, ortho=True)
H = locfun.get_hamiltonian(calc)
U = locfun.U_nn
xc = get_vxc(calc, spin=0, U=U)
