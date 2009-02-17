import numpy as np
#np.seterr(all='raise')
#np.set_printoptions(precision=3, suppress=True)

from ase import *
from gpaw import *
from gpaw.coulomb import get_vxc, HF
from gpaw.wannier import LocFun

if 1:
    calc = GPAW(nbands=9)
    atoms = molecule('H2O', calculator=calc)
    atoms.center(vacuum=2.4)
    atoms.get_potential_energy()
    calc.write('H2O.gpw', mode='all')

atoms, calc = restart('H2O.gpw', txt=None)
# calc.set_positions() # XXX this will ruin calc.wfs.kpt_u[u].P_ani
calc.density.ghat.set_positions(atoms.get_scaled_positions() % 1.)
calc.hamiltonian.poisson.initialize(calc.finegd)

locfun = LocFun()
locfun.localize(calc, ortho=True)
H = locfun.get_hamiltonian(calc)
U = locfun.U_nn
xc = get_vxc(calc, spin=0, U=U)

hf = HF(calc)
F = hf.apply(calc, 0)
Fcore = np.zeros((calc.wfs.nbands, calc.wfs.nbands), float)
hf.atomic_val_core(calc, Fcore, u=0)
Fcore *= Hartree
