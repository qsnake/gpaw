from ase import *
from gpaw import *

a = 2.7
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=3, basis='szp')
bulk = Atoms([Atom('Li')], pbc=True, cell=[a, a, a], calculator=calc)
print bulk.get_potential_energy()


from gpaw.lcao.projected_wannier import get_lfc, get_bfs
from gpaw.lcao.overlap import TwoCenterIntegrals

dtype = calc.wfs.dtype
nq = len(calc.wfs.ibzk_qc)
nao = calc.wfs.setups.nao
nbands = calc.wfs.nbands
kpt_q = calc.wfs.kpt_u
setups = calc.wfs.setups
spos_ac = bulk.get_scaled_positions()

tci = TwoCenterIntegrals(calc.gd, setups, calc.wfs.gamma, calc.wfs.ibzk_qc)
tci.set_positions(spos_ac)

# Calculate projector overlaps, and (lower triangle of-) S and T matrices
S_qMM = np.zeros((nq, nao, nao), dtype)
T_qMM = np.zeros((nq, nao, nao), dtype)
P_aqMi = {}
for a in range(len(spos_ac)):
    ni = calc.wfs.setups[a].ni
    P_aqMi[a] = np.zeros((nq, nao, ni), dtype)
tci.calculate(spos_ac, S_qMM, T_qMM, P_aqMi)


# Calculate projections using BFS
bfs = get_bfs(calc)
V_qnM = np.zeros((nq, nbands, nao), dtype)
for q, V_nM in enumerate(V_qnM):
    bfs.integrate2(kpt_q[q].psit_nG[:], V_nM, q)
    for a, P_ni in kpt_q[q].P_ani.items():
        P_Mi = P_aqMi[a][q]
        V_nM += np.dot(np.dot(P_ni, setups[a].O_ii), P_Mi.T.conj())

bfs_qnM = V_qnM.copy()


# Calculate projections using LFC
lfc = get_lfc(calc)
V_qnM = np.zeros((nq, nbands, nao), dtype)
V_qAni = [lfc.dict(nbands) for q in range(nq)]
for q, V_Ani in enumerate(V_qAni):
    lfc.integrate(kpt_q[q].psit_nG[:], V_Ani, q)
    M1 = 0
    for A in range(len(V_Ani)):
        V_ni = V_Ani[A]
        M2 = M1 + V_ni.shape[1]
        V_qnM[q, :, M1:M2] += V_ni
        M1 = M2
    for a, P_ni in calc.wfs.kpt_u[q].P_ani.items():
        P_Mi = P_aqMi[a][q]
        V_qnM[q] += np.dot(P_ni, np.inner(setups[a].O_ii, P_Mi).conj())

lfc_qnM = V_qnM

print 'Difference', abs(lfc_qnM - bfs_qnM).ptp()
