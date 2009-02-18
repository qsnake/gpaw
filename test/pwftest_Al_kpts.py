import numpy as np

from time import time
from os.path import isfile

from ase import Atoms, molecule
from gpaw import GPAW, Mixer
from gpaw.lcao.tools import get_realspace_hs
from gpaw.lcao.projected_wannier import ProjectedWannierFunctions, get_phs

if not isfile('al.gpw'):
    atoms = Atoms('Al', cell=(2.42, 7, 7), pbc=True)
    calc = GPAW(h=0.2, basis='dzp', kpts=(12, 1, 1), 
                convergence={'bands':9},
                maxiter=200)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('al.gpw', 'all')
else:
    calc = GPAW('al.gpw', txt=None, basis='sz')

ibzk_kc = calc.wfs.ibzk_kc
nk = len(ibzk_kc)
Ef = calc.get_fermi_level()
eps_kn = np.asarray([calc.get_eigenvalues(kpt=k) for k in range(nk)])
eps_kn -= Ef

V_knM, H_kMM, S_kMM, P_aqMi = get_phs(calc, s=0)
H_kMM -= S_kMM*Ef

pwf = ProjectedWannierFunctions(V_knM, 
                                h_lcao=H_kMM, 
                                s_lcao=S_kMM, 
                                eigenvalues=eps_kn, 
                                fixedenergy=1.0,
                                kpoints=ibzk_kc)

t1 = time()
h_kMM, s_kMM = pwf.get_hamiltonian_and_overlap_matrix(useibl=False)
t2 = time()

print "\nTime to construct PWF: %.3f seconds "  % (t2 - t1)
print "max condition number:", pwf.get_condition_number().max()
eigs_kn = pwf.get_eigenvalues()
fd2 = open('bands_al_sz.dat','w')
fd1 = open('bands_al_exact.dat', 'w')
for eps1_n, eps2_n, k in zip(eps_kn, eigs_kn, ibzk_kc[:,0]):
    for e1 in eps1_n:
        print >> fd1, k, e1
    for e2 in eps2_n:
        print >> fd2, k, e2
fd1.close()            
fd2.close()
h_skMM = h_kMM[None]
n = 2
w_k = calc.wfs.weight_k
h_n, s_n = get_realspace_hs(h_skMM, s_kMM, ibzk_kc, w_k, (n, 0, 0),
                            usesymm=False)
