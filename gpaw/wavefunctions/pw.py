import numpy as np
from numpy.fft import fftn, ifftn
import ase.units as units

from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator


class PWDescriptor:
    def __init__(self, ecut, gd, ibzk_qc):
        assert gd.pbc_c.all() and gd.comm.size == 1

        self.ecut = ecut

        assert 0.5 * np.pi**2 / (gd.h_cv**2).sum(1).max() > ecut
        
        # Calculate reciprocal lattice vectors:
        N_c = gd.N_c
        i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
        i_Qc += N_c // 2
        i_Qc %= N_c
        i_Qc -= N_c // 2
        B_cv = 2.0 * np.pi * gd.icell_cv
        G_Qv = np.dot(i_Qc, B_cv).reshape((-1, 3))
        G2_Q = (G_Qv**2).sum(axis=1)
        self.Q_q = np.arange(len(G2_Q))[G2_Q <= 2 * ecut]
        K_qv = np.dot(ibzk_qc, B_cv)
        G_qv = G_Qv[self.Q_q]
        self.kin_qq = np.zeros((len(ibzk_qc), len(self.Q_q)))
        for q, K_v in enumerate(K_qv):
            self.kin_qq[q] = 0.5 * ((G_qv + K_v)**2).sum(1)
        
        self.gd = gd
        self.dv = gd.dv / N_c.prod()
        self.comm = gd.comm
        
    def zeros(self, n=(), dtype=float):
        assert dtype == complex
        if isinstance(n, int):
            n = (n,)
        shape = n + self.Q_q.shape
        return np.zeros(shape, complex)
    
    def fft(self, a_xG):
        a_xQ = fftn(a_xG, axes=(-3, -2, -1))
        return a_xQ.reshape(a_xG.shape[:-3] + (-1,))[..., self.Q_q].copy()

    def ifft(self, a_xq):
        xshape = a_xq.shape[:-1]
        a_xQ = self.gd.zeros(xshape, complex)
        a_xQ.reshape(xshape + (-1,))[..., self.Q_q] = a_xq
        return ifftn(a_xQ, axes=(-3, -2, -1)).copy()


class Preconditioner:
    def __init__(self, pd):
        self.pd = pd
        self.allocated = True

    def __call__(self, R_q, kpt):
        return R_q / (1.0 + self.pd.kin_qq[kpt.q])


class PWWaveFunctions(FDPWWaveFunctions):
    def __init__(self, ecut, diagksl, orthoksl, initksl,
                 gd, nspins, nvalence, setups, bd,
                 world, kpt_comm,
                 bzk_kc, ibzk_kc, weight_k,
                 symmetry, timer):
        self.ecut =  ecut / units.Hartree
        # Set dtype=complex and gamma=False:
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nspins, nvalence, setups, bd, complex,
                                   world, kpt_comm,
                                   False, bzk_kc, ibzk_kc, weight_k,
                                   symmetry, timer)
        
        self.matrixoperator = MatrixOperator(self.bd, self.pd, orthoksl)
        self.wd = self.pd        

    def set_setups(self, setups):
        self.pd = PWDescriptor(self.ecut, self.gd, self.ibzk_qc)
        pt = LFC(self.gd, [setup.pt_j for setup in setups],
                 self.kpt_comm, dtype=self.dtype, forces=True)
        self.pt = PWLFC(pt, self.pd)
        FDPWWaveFunctions.set_setups(self, setups)

    def summary(self, fd):
        fd.write('Mode: Plane waves (%d, ecut=%.3f eV)\n' %
                 (len(self.pd.Q_q), self.pd.ecut * units.Hartree))
        
    def make_preconditioner(self):
        return Preconditioner(self.pd)

    def apply_hamiltonian(self, hamiltonian, kpt, psit_xq, Htpsit_xq):
        Htpsit_xq[:] = psit_xq * self.pd.kin_qq[kpt.q]
        for psit_q, Htpsit_q in zip(psit_xq, Htpsit_xq):
            psit_G = self.pd.ifft(psit_q)
            Htpsit_q += self.pd.fft(psit_G * hamiltonian.vt_sG[kpt.s])

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        nt_G = nt_sG[kpt.s]
        for f, psit_q in zip(f_n, kpt.psit_nG):
            nt_G += f * abs(self.pd.ifft(psit_q))**2

    def initialize_wave_functions_from_basis_functions(self, basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        FDPWWaveFunctions.initialize_wave_functions_from_basis_functions(
            self, basis_functions, density, hamiltonian, spos_ac)

        for kpt in self.kpt_u:
            kpt.psit_nG = self.pd.fft(kpt.psit_nG)


class PWLFC:
    def __init__(self, lfc, pd):
        self.lfc = lfc
        self.pd = pd

    def set_positions(self, spos_ac):
        self.lfc.set_positions(spos_ac)
        self.my_atom_indices = self.lfc.my_atom_indices
        
    def set_k_points(self, ibzk_qc):
        self.lfc.set_k_points(ibzk_qc)
        N_c = self.pd.gd.N_c
        self.expikr_qG = np.exp(2j * np.pi * np.dot(np.indices(N_c).T,
                                                    (ibzk_qc / N_c).T).T)

    def add(self, a_xq, C_axi, q):
        a_xG = self.pd.gd.zeros(a_xq.shape[:-1], complex)
        self.lfc.add(a_xG, C_axi, q)
        a_xq[:] += self.pd.fft(a_xG / self.expikr_qG[q])

    def integrate(self, a_xq, C_axi, q):
        a_xG = self.pd.ifft(a_xq) * self.expikr_qG[q]
        C_axi[0][:]=0
        self.lfc.integrate(a_xG, C_axi, q)


class PW: ####### use mode='pw'?  ecut=???
    def __init__(self, ecut=340):
        self.ecut = ecut

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, diagksl, orthoksl, initksl, *args)
        return wfs
