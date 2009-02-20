import numpy as np
la = np.linalg

from ase import Hartree
from gpaw.aseinterface import GPAW
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.lfc import BasisFunctions
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
from gpaw.coulomb import get_vxc as get_ks_xc

from gpaw.lcao.projected_wannier import dots, condition_number, eigvals,\
     get_bfs, get_lfc


def get_rot(F_MM, V_oM, L):
    eps_M, U_MM = la.eigh(F_MM)
    indices = eps_M.real.argsort()[-L:] 
    U_Ml = U_MM[:, indices]
    U_Ml /= np.sqrt(dots(U_Ml.T.conj(), F_MM, U_Ml).diagonal())

    U_ow = V_oM.copy()
    U_lw = np.dot(U_Ml.T.conj(), F_MM)
    for col1, col2 in zip(U_ow.T, U_lw.T):
         norm = np.sqrt(np.vdot(col1, col1) + np.vdot(col2, col2))
         col1 /= norm
         col2 /= norm
    return U_ow, U_lw, U_Ml
    

def get_lcao_projections_HSP(calc, bfs=None, spin=0, projectionsonly=True):
    """Some title.

    if projectionsonly is True, return the projections::

      V_qnM = <psi_qn | Phi_qM>

    else, also return the Hamiltonian, overlap, and projector overlaps::

      H_qMM  = <Phi_qM| H |Phi_qM'>
      S_qMM  = <Phi_qM|Phi_qM'>
      P_aqMi = <pt^a_qi|Phi_qM>
    """
    spos_ac = calc.atoms.get_scaled_positions()
    nq = len(calc.wfs.ibzk_qc)
    nao = calc.wfs.setups.nao
    dtype = calc.wfs.dtype
    if bfs is None:
        bfs = get_bfs(calc)
    tci = TwoCenterIntegrals(calc.domain, calc.wfs.setups,
                             calc.wfs.gamma, calc.wfs.ibzk_qc)
    tci.set_positions(spos_ac)

    # Calculate projector overlaps, and (lower triangle of-) S and T matrices
    S_qMM = np.zeros((nq, nao, nao), dtype)
    T_qMM = np.zeros((nq, nao, nao), dtype)
    P_aqMi = {}
    for a in range(len(spos_ac)):
        ni = calc.wfs.setups[a].ni
        P_aqMi[a] = np.zeros((nq, nao, ni), dtype)
    tci.calculate(spos_ac, S_qMM, T_qMM, P_aqMi, dtype)

    # Calculate projections
    V_qnM = np.zeros((nq, calc.wfs.nbands, nao), dtype)
    for q, V_nM in enumerate(V_qnM):
        bfs.integrate2(calc.wfs.kpt_u[q].psit_nG[:], V_nM)
        for a, P_ni in calc.wfs.kpt_u[q].P_ani.items():
            dS_ii = calc.wfs.setups[a].O_ii
            P_Mi = P_aqMi[a][q]
            V_nM += np.dot(P_ni, np.inner(dS_ii, P_Mi).conj())
    if projectionsonly:
        return V_qnM

    # Determine potential matrix
    vt_G = calc.hamiltonian.vt_sG[spin]
    V_qMM = np.zeros((nq, nao, nao), dtype)
    for q, V_MM in enumerate(V_qMM):
        bfs.calculate_potential_matrix(vt_G, V_MM, q)

    # Make Hamiltonian as sum of kinetic (T) and potential (V) matrices
    # and add atomic corrections
    H_qMM = T_qMM + V_qMM
    for a, P_qMi in P_aqMi.items():
        dH_ii = unpack(calc.hamiltonian.dH_asp[a][spin])
        for P_Mi, H_MM in zip(P_qMi, H_qMM):
            H_MM +=  np.dot(P_Mi, np.inner(dH_ii, P_Mi).conj())
    
    # Fill in the upper triangles of H and S
    for H_MM, S_MM in zip(H_qMM, S_qMM):
        tri2full(H_MM)
        tri2full(S_MM)
    H_qMM *= Hartree

    return V_qnM, H_qMM, S_qMM, P_aqMi


def get_lcao_xc(calc, P_aqMi, bfs=None, spin=0):
    nq = len(calc.wfs.ibzk_qc)
    nao = calc.wfs.setups.nao
    dtype = calc.wfs.dtype
    if bfs is None:
        bfs = get_bfs(calc)
    
    if calc.density.nt_sg is None:
        calc.density.interpolate()
    nt_g = calc.density.nt_sg[spin]
    vxct_g = calc.finegd.zeros()
    calc.hamiltonian.xc.get_energy_and_potential(nt_g, vxct_g)
    vxct_G = calc.gd.empty()
    calc.hamiltonian.restrict(vxct_g, vxct_G)
    Vxc_qMM = np.zeros((nq, nao, nao), dtype)
    for q, Vxc_MM in enumerate(Vxc_qMM):
        bfs.calculate_potential_matrix(vxct_G, Vxc_MM, q)
        tri2full(Vxc_MM)

    # Add atomic PAW corrections
    for a, P_qMi in P_aqMi.items():
        D_sp = calc.density.D_asp[a][:]
        H_sp = np.zeros_like(D_sp)
        calc.wfs.setups[a].xc_correction.calculate_energy_and_derivatives(
            D_sp, H_sp)
        H_ii = unpack(H_sp[spin])
        for Vxc_MM, P_Mi in zip(Vxc_qMM, P_qMi):
            Vxc_MM += np.dot(P_Mi, np.dot(H_ii, P_Mi.T).conj())
    return Vxc_qMM * Hartree


class ProjectedWannierFunctionsFBL:
    """PWF in the finite band limit.

    ::
    
                --N              
        |w_w> = >    |psi_n> U_nw
                --n=1            
    """
    def __init__(self, V_nM, No):
        Nw = V_nM.shape[1]
        V_oM, V_uM = V_nM[:No], V_nM[No:]
        F_MM = np.dot(V_uM.T.conj(), V_uM)
        U_ow, U_lw, U_Ml = get_rot(F_MM, V_oM, Nw - No)
        self.U_nw = np.vstack((U_ow, dots(V_uM, U_Ml, U_lw)))
        self.S_ww = np.dot(self.U_nw.T.conj(), self.U_nw)
        self.norms_n = np.dot(self.U_nw, la.solve(
            self.S_ww, self.U_nw.T.conj())).diagonal()

    def rotate_matrix(self, A_nn):
        if A_nn.ndim == 1:
            return np.dot(self.U_nw.T.conj() * A_nn, self.U_nw)
        else:
            return dots(self.U_nw.T.conj(), A_nn, self.U_nw)

    def rotate_projections(self, P_ani):
        P_awi = {}
        for a, P_ni in P_ani.items():
            P_awi[a] = np.tensordot(self.U_nw, P_ni, axes=[[0], [0]])
        return P_awi

    def function(self, psit_nG):
        return np.tensordot(psit_nG, self.U_nw, axes=[[0], [0]])


class ProjectedWannierFunctionsIBL:
    """PWF in the infinite band limit.

    ::
    
                --No               --Nw
        |w_w> = >   |psi_o> U_ow + >   |f_M> U_Mw
                --o=1              --M=1
    """
    def __init__(self, V_nM, S_MM, No):
        Nw = V_nM.shape[1]
        self.V_oM, V_uM = V_nM[:No], V_nM[No:]
        F_MM = S_MM - np.dot(self.V_oM.T.conj(), self.V_oM)
        U_ow, U_lw, U_Ml = get_rot(F_MM, self.V_oM, Nw - No)
        self.U_Mw = np.dot(U_Ml, U_lw)
        self.U_ow = U_ow - np.dot(self.V_oM, self.U_Mw)
        self.S_ww = np.dot(U_ow.T.conj(), U_ow) + dots(self.U_Mw.T.conj(),
                                                       F_MM, self.U_Mw)
        P_uw = np.dot(V_uM, self.U_Mw)
        self.norms_n = np.hstack((
            np.dot(U_ow, la.solve(self.S_ww, U_ow.T.conj())).diagonal(),
            np.dot(P_uw, la.solve(self.S_ww, P_uw.T.conj())).diagonal()))

    def rotate_matrix(self, A_oo, A_MM):
        if A_oo.ndim == 1:
            A_ww = dots(self.U_ow.T.conj() * A_oo, self.V_oM, self.U_Mw)
            A_ww += np.conj(A_ww.T)
            A_ww += np.dot(self.U_ow.T.conj() * A_oo, self.U_ow)
        else:
            A_ww = dots(self.U_ow.T.conj(), A_oo, self.V_oM, self.U_Mw)
            A_ww += np.conj(A_ww.T)
            A_ww += dots(self.U_ow.T.conj(), A_oo, self.U_ow)
        A_ww += dots(self.U_Mw.T.conj(), A_MM, self.U_Mw)
        return A_ww

    def rotate_projections(self, P_aoi, P_aMi):
        P_awi = {}
        for a, P_oi in P_aoi.items():
            P_awi[a] = np.tensordot(self.U_ow, P_oi, axes=[[0], [0]]) + \
                       np.tensordot(self.U_Mw, P_aMi[a], axes=[[0], [0]])
        return P_awi

    def function(self, psit_oG, bfs, q=-1):
        w_wG = np.tensordot(self.U_ow, psit_oG, axes=[[0], [0]])
        bfs.lcao_to_grid(self.U_Mw.T.copy(), w_wG, q)


class PWF2:
    def __init__(self, gpwfilename, fixedenergy, spin=0, ibl=True):
        calc = GPAW(gpwfilename, txt=None, basis='sz')
        Ef = calc.get_fermi_level()

        nq = 1 # Temporary hack XXX
        q = 0 # Temporary hack XXX
        eps_n = calc.get_eigenvalues(q) - Ef
        M = sum(eps_n <= fixedenergy)
        kpt = calc.wfs.kpt_u[spin * nq + q]
        
        if ibl:
            bfs = get_bfs(calc)
            V_qnM, H_qMM, S_qMM, P_aqMi = get_lcao_projections_HSP(
                calc, bfs=bfs, spin=spin, projectionsonly=False)
            H_qMM -= Ef * S_qMM
            pwf = ProjectedWannierFunctionsIBL(V_qnM[q], S_qMM[q], M)

            H_ww = pwf.rotate_matrix(eps_n[:M], H_qMM[q])
            xc_ww = pwf.rotate_matrix(get_ks_xc(calc, spin=spin)[:M, :M],
                                      get_lcao_xc(calc, P_aqMi, bfs, spin)[q])
            w_wG = pwf.function(kpt.psit_nG[:][:M], bfs)
            P_awi = pwf.rotate_projections(
                dict([(a, P_ni[:M]) for a, P_ni in kpt.P_ani.items()]),
                dict([(a, P_qMi[q]) for a, P_qMi in P_aqMi.items()]))
        else:
            V_qnM = get_lcao_projections_HSP(calc, spin=spin)
            pwf = ProjectedWannierFunctionsFBL(V_qnM[q], M)
            H_ww = pwf.rotate_matrix(eps_n)
            xc_ww = pwf.rotate_matrix(get_ks_xc(calc, spin=spin))
            w_wG = pwf.function(kpt.psit_nG[:])
            P_awi = pwf.rotate_projections(kpt.P_ani)

        S_ww = pwf.S_ww
        norms_n = pwf.norms_n

        # Store all the relevant stuff
        self.HSxc = (H_ww, S_ww, xc_ww)
        self.orbitals = (w_wG, P_awi)
        self.norms = norms_n
        self.eigs = eigvals(H_ww, S_ww)
        print 'Condition number:', condition_number(S_ww)
