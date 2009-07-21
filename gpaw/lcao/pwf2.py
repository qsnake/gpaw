import numpy as np

from ase import Hartree
from gpaw.aseinterface import GPAW
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.lfc import BasisFunctions
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full, lowdin
from gpaw.coulomb import get_vxc as get_ks_xc
from gpaw.utilities.blas import r2k, gemm

from gpaw.lcao.projected_wannier import dots, condition_number, eigvals, \
     get_bfs, get_lcao_projections_HSP


def get_rot(F_MM, V_oM, L):
    eps_M, U_MM = np.linalg.eigh(F_MM)
    indices = eps_M.real.argsort()[-L:] 
    U_Ml = U_MM[:, indices]
    U_Ml /= np.sqrt(dots(U_Ml.T.conj(), F_MM, U_Ml).diagonal())

    U_ow = V_oM.copy()
    U_lw = np.dot(U_Ml.T.conj(), F_MM)
    for col1, col2 in zip(U_ow.T, U_lw.T):
        norm = np.linalg.norm(np.hstack((col1, col2)))
        col1 /= norm
        col2 /= norm
    return U_ow, U_lw, U_Ml
    

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
    vxct_G = calc.gd.zeros()
    calc.hamiltonian.restrict(vxct_g, vxct_G)
    Vxc_qMM = np.zeros((nq, nao, nao), dtype)
    for q, Vxc_MM in enumerate(Vxc_qMM):
        bfs.calculate_potential_matrix(vxct_G, Vxc_MM, q)
        tri2full(Vxc_MM, 'L')

    # Add atomic PAW corrections
    for a, P_qMi in P_aqMi.items():
        D_sp = calc.density.D_asp[a][:]
        H_sp = np.zeros_like(D_sp)
        calc.wfs.setups[a].xc_correction.calculate_energy_and_derivatives(
            D_sp, H_sp)
        H_ii = unpack(H_sp[spin])
        for Vxc_MM, P_Mi in zip(Vxc_qMM, P_qMi):
            Vxc_MM += dots(P_Mi, H_ii, P_Mi.T.conj())
    return Vxc_qMM * Hartree


def get_xc2(calc, w_wG, P_awi, spin=0):
    if calc.density.nt_sg is None:
        calc.density.interpolate()
    nt_g = calc.density.nt_sg[spin]
    vxct_g = calc.finegd.zeros()
    calc.hamiltonian.xc.get_energy_and_potential(nt_g, vxct_g)
    vxct_G = calc.gd.empty()
    calc.hamiltonian.restrict(vxct_g, vxct_G)

    # Integrate pseudo part
    Nw = len(w_wG)
    xc_ww = np.empty((Nw, Nw))
    r2k(.5 * calc.gd.dv, w_wG, vxct_G * w_wG, .0, xc_ww)
    tri2full(xc_ww, 'L')
    
    # Add atomic PAW corrections
    for a, P_wi in P_awi.items():
        D_sp = calc.density.D_asp[a][:]
        H_sp = np.zeros_like(D_sp)
        calc.wfs.setups[a].xc_correction.calculate_energy_and_derivatives(
            D_sp, H_sp)
        H_ii = unpack(H_sp[spin])
        xc_ww += dots(P_wi, H_ii, P_wi.T.conj())
    return xc_ww * Hartree


class ProjectedWannierFunctionsFBL:
    """PWF in the finite band limit.

    ::
    
                --N              
        |w_w> = >    |psi_n> U_nw
                --n=1            
    """
    def __init__(self, V_nM, No, ortho=False):
        Nw = V_nM.shape[1]
        assert No <= Nw
        V_oM, V_uM = V_nM[:No], V_nM[No:]
        F_MM = np.dot(V_uM.T.conj(), V_uM)
        U_ow, U_lw, U_Ml = get_rot(F_MM, V_oM, Nw - No)
        self.U_nw = np.vstack((U_ow, dots(V_uM, U_Ml, U_lw)))

        # stop here ?? XXX
        self.S_ww = self.rotate_matrix(np.ones(1))
        if ortho:
            lowdin(self.U_nw, self.S_ww)
            self.S_ww = np.identity(Nw)
        self.norms_n = np.dot(self.U_nw, np.linalg.solve(
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

    def rotate_function(self, psit_nG):
        return np.tensordot(self.U_nw, psit_nG, axes=[[0], [0]])


class ProjectedWannierFunctionsIBL:
    """PWF in the infinite band limit.

    ::
    
                --No               --Nw
        |w_w> = >   |psi_o> U_ow + >   |f_M> U_Mw
                --o=1              --M=1
    """
    def __init__(self, V_nM, S_MM, No):
        Nw = V_nM.shape[1]
        assert No <= Nw
        self.V_oM, V_uM = V_nM[:No], V_nM[No:]
        F_MM = S_MM - np.dot(self.V_oM.T.conj(), self.V_oM)
        U_ow, U_lw, U_Ml = get_rot(F_MM, self.V_oM, Nw - No)
        self.U_Mw = np.dot(U_Ml, U_lw)
        self.U_ow = U_ow - np.dot(self.V_oM, self.U_Mw)

        # stop here ?? XXX
        self.S_ww = self.rotate_matrix(np.ones(1), S_MM)
        P_uw = np.dot(V_uM, self.U_Mw)
        self.norms_n = np.hstack((
           np.dot(U_ow, np.linalg.solve(self.S_ww, U_ow.T.conj())).diagonal(),
           np.dot(P_uw, np.linalg.solve(self.S_ww, P_uw.T.conj())).diagonal()))

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

    def rotate_projections(self, P_aoi, P_aMi, indices=None):
        if indices is None:
            U_ow = self.U_ow
            U_Mw = self.U_Mw
        else:
            U_ow = self.U_ow[:, indices]
            U_Mw = self.U_Mw[:, indices]
        P_awi = {}
        for a, P_oi in P_aoi.items():
            P_awi[a] = np.tensordot(U_Mw, P_aMi[a], axes=[[0], [0]])
            if len(U_ow) > 0:
                P_awi[a] += np.tensordot(U_ow, P_oi, axes=[[0], [0]])
        return P_awi

    def rotate_function(self, psit_oG, bfs, q=-1, indices=None):
        if indices is None:
            U_ow = self.U_ow
            U_Mw = self.U_Mw
        else:
            U_ow = self.U_ow[:, indices]
            U_Mw = self.U_Mw[:, indices]
        w_wG = np.zeros((U_ow.shape[1],) + psit_oG.shape[1:])
        if len(U_ow) > 0:
            gemm(1., psit_oG, U_ow.T.copy(), 0., w_wG)
        bfs.lcao_to_grid(U_Mw.T.copy(), w_wG, q)
        return w_wG


class PWF2:
    def __init__(self, gpwfilename, fixedenergy=0.,
                 spin=0, ibl=True, basis='sz', zero_fermi=False):
        calc = GPAW(gpwfilename, txt=None, basis=basis)
        calc.wfs.initialize_wave_functions_from_restart_file()
        calc.density.ghat.set_positions(calc.atoms.get_scaled_positions() % 1.)
        calc.hamiltonian.poisson.initialize()
        if zero_fermi:
            try:
                Ef = calc.get_fermi_level()
            except NotImplementedError:
                Ef = calc.get_homo_lumo().mean()
        else:
            Ef = 0.0

        self.ibzk_kc = calc.get_ibz_k_points()
        self.nk = len(self.ibzk_kc)
        self.eps_kn = [calc.get_eigenvalues(kpt=q, spin=spin) - Ef
                       for q in range(self.nk)]
        self.M_k = [sum(eps_n <= fixedenergy) for eps_n in self.eps_kn]
        print 'Fixed states:', self.M_k 
        self.calc = calc
        self.dtype = self.calc.wfs.dtype
        self.spin = spin
        self.ibl = ibl
        self.pwf_q = []
        self.norms_qn = []
        self.S_qww = []
        self.H_qww = []

        if ibl:
            self.bfs = get_bfs(calc)
            V_qnM, H_qMM, S_qMM, self.P_aqMi = get_lcao_projections_HSP(
                calc, bfs=self.bfs, spin=spin, projectionsonly=False)
            H_qMM -= Ef * S_qMM
            for q, M in enumerate(self.M_k):
                pwf = ProjectedWannierFunctionsIBL(V_qnM[q], S_qMM[q], M)
                self.pwf_q.append(pwf)
                self.norms_qn.append(pwf.norms_n)
                self.S_qww.append(pwf.S_ww)
                self.H_qww.append(pwf.rotate_matrix(self.eps_kn[q][:M],
                                                    H_qMM[q]))
        else:
            V_qnM = get_lcao_projections_HSP(calc, spin=spin)
            for q, M in enumerate(self.M_k):
                pwf = ProjectedWannierFunctionsFBL(V_qnM[q], M, ortho=False)
                self.pwf_q.append(pwf)
                self.norms_qn.append(pwf.norms_n)
                self.S_qww.append(pwf.S_ww)
                self.H_qww.append(pwf.rotate_matrix(self.eps_kn[q]))

        for S in self.S_qww:
            print 'Condition number:', condition_number(S)

    def get_hamiltonian(self, q=0, indices=None):
        if indices is None:
            return self.H_qww[q]
        else:
            return self.H_qww[q].take(indices, 0).take(indices, 1)

    def get_overlap(self, q=0, indices=None):
        if indices is None:
            return self.S_qww[q]
        else:
            return self.S_qww[q].take(indices, 0).take(indices, 1)

    def get_projections(self, q=0, indices=None):
        kpt = self.calc.wfs.kpt_u[self.spin * self.nk + q]
        if not hasattr(self, 'P_awi'):
            if self.ibl:
                M = self.M_k[q]
                self.P_awi = self.pwf_q[q].rotate_projections(
                    dict([(a, P_ni[:M]) for a, P_ni in kpt.P_ani.items()]),
                    dict([(a, P_qMi[q]) for a, P_qMi in self.P_aqMi.items()]),
                    indices)
            else:
                self.P_awi = pwf.rotate_projections(kpt.P_ani, indices)
        return self.P_awi

    def get_orbitals(self, q=0, indices=None):
        kpt = self.calc.wfs.kpt_u[self.spin * self.nk + q]
        if not hasattr(self, 'w_wG'):
            if self.ibl:
                self.w_wG = self.pwf_q[q].rotate_function(
                    kpt.psit_nG[:][:self.M_k[q]], self.bfs, q, indices)
            else:
                self.w_wG = self.pwf_q[q].rotate_function(
                    kpt.psit_nG[:], indices)
        return self.w_wG

    def get_Fcore(self, q=0, indices=None):
        if indices is None:
            Fcore_ww = np.zeros_like(self.H_qww[q])
        else:
            Fcore_ww = np.zeros((len(indices), len(indices)))
        for a, P_wi in self.get_projections(q, indices).items():
            X_ii = unpack(self.calc.wfs.setups[a].X_p)
            Fcore_ww -= dots(P_wi, X_ii, P_wi.T.conj())
        return Fcore_ww * Hartree

    def get_eigs(self, q=0):
        return eigvals(self.H_qww[q], self.S_ww[q])

    def get_condition_number(self, q=0):
        return condition_number(self.S_qww[q])

    def get_xc(self, q=0, indices=None):
        if self.ibl:
            return get_xc2(self.calc, self.get_orbitals(q, indices),
                           self.get_projections(q, indices), self.spin)
        else:
            return self.pwf_q[q].rotate_matrix(get_ks_xc(self.calc,
                                                         spin=self.spin))
