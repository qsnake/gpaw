import os

import numpy as np

from time import time
from ase import molecule
from gpaw import GPAW
from gpaw.lcao.projected_wannier import ProjectedWannierFunctions, get_phs

if not os.path.isfile('C6H6.gpw'):
    atoms = molecule('C6H6')
    atoms.center(vacuum=2.5)
    calc = GPAW(h=0.2, basis='szp', width=0.05, convergence={'bands':17})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('C6H6.gpw', 'all')
calc = GPAW('C6H6.gpw', txt=None, basis='sz')

ibzk_kc = calc.wfs.ibzk_kc
nk = len(ibzk_kc)
Ef = calc.get_fermi_level()
eps_kn = np.asarray([calc.get_eigenvalues(k) for k in range(nk)])
eps_kn -= Ef

V_knM, H_kMM, S_kMM, P_aqMi = get_phs(calc, s=0)
H_kMM -= Ef * S_kMM 

pwf = ProjectedWannierFunctions(V_knM, 
                                h_lcao=H_kMM, 
                                s_lcao=S_kMM, 
                                eigenvalues=eps_kn,
                                kpoints=ibzk_kc,
                                fixedenergy=5.0)
t1 = time()
h, s = pwf.get_hamiltonian_and_overlap_matrix(useibl=False)
t2 = time()

print "\nTime to construct PWF: %.3f seconds "  % (t2 - t1)
norm_kn = pwf.get_norm_of_projection()
eps1_kn = pwf.get_eigenvalues()
print "band | deps/eV |  norm"
print "-------------------------"
for n in range(norm_kn.shape[1]):
    norm = norm_kn[0, n]
    if n >= eps1_kn.shape[1]:
        print "%4i |    -    | %.1e " % (n, norm)
    else:
        deps = np.around(abs(eps1_kn[0,n] - eps_kn[0, n]), 13)
        print "%4i | %.1e | %.1e " % (n, deps, norm)

for M, norm_n in zip(pwf.M_k, norm_kn):
    assert np.all(abs(norm_n[:M]-1.0) < 1.0e-15)

print pwf.get_condition_number()

#os.remove('C6H6.gpw')
