import numpy as np

from time import time
from os.path import isfile

from ase import Atoms
from gpaw import GPAW, Mixer
from gpaw.lcao.projected_wannier import ProjectedWannierFunctions, get_phs

if not isfile('al8.gpw'):
    atoms = Atoms('Al', cell=(2.42, 7, 7), pbc=True)
    atoms*=(8, 1, 1)
    calc = GPAW(h=0.2, basis='szp', kpts=(1, 1, 1), 
                convergence={'bands':4*8}, width=0.1,
                maxiter=200, mixer=Mixer(0.1, 7, weight=100.))
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('al8.gpw', 'all')
calc = GPAW('al8.gpw', txt=None, basis='sz')

ibzk_kc = calc.wfs.ibzk_kc
nk = len(ibzk_kc)
Ef = calc.get_fermi_level()
eps_kn = np.asarray([calc.get_eigenvalues(kpt=k) for k in range(nk)])
eps_kn -= Ef

V_knM, H_kMM, S_kMM, P_aqMi = get_phs(calc, s=0)
H_kMM -= S_kMM * Ef

pwf = ProjectedWannierFunctions(V_knM, 
                                h_lcao=H_kMM, 
                                s_lcao=S_kMM, 
                                eigenvalues=eps_kn, 
                                fixedenergy=0.0,
                                kpoints=ibzk_kc)

t1 = time()
h_kMM, s_kMM = pwf.get_hamiltonian_and_overlap_matrix(useibl=True)
t2 = time()
print "\nTime to construct PWF: %.3f seconds "  % (t2 - t1)


