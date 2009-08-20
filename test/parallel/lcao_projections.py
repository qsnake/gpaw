from ase import *
from gpaw import *
from gpaw.lcao.projected_wannier import get_lcao_projections_HSP


atoms = molecule('C2H2')
atoms.center(vacuum=3.0)
calc = GPAW()
atoms.set_calculator(calc)
atoms.get_potential_energy()

V_qnM, H_qMM, S_qMM, P_aqMi = get_lcao_projections_HSP(
    calc, bfs=None, spin=0, projectionsonly=False)
eig = np.linalg.eigvals(np.linalg.solve(S_qMM[0], H_qMM[0])).real
eig.sort()
print eig


eig_ref = np.array([-17.81199134, -13.20508588, -11.37846045,  -7.07651757,
                    -7.07651757,   0.6523491 ,   0.6523491 ,   3.96182172,
                    7.49739606,  26.85012745])
assert np.allclose(eig, eig_ref)
