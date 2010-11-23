import numpy as np
from numpy.fft import fftn, ifftn
import ase.units as units

from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator


class PWDescriptor:
    def __init__(self, ecut, gd, ibzk_qc=[(0, 0, 0)]):
        assert gd.pbc_c.all() and gd.comm.size == 1

        self.ecut = ecut

        assert 0.5 * np.pi**2 / (gd.h_cv**2).sum(1).max() >= ecut
        
        # Calculate reciprocal lattice vectors:
        N_c = gd.N_c
        i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
        i_Qc += N_c // 2
        i_Qc %= N_c
        i_Qc -= N_c // 2
        B_cv = 2.0 * np.pi * gd.icell_cv
        G_Qv = np.dot(i_Qc, B_cv).reshape((-1, 3))
        G2_Q = (G_Qv**2).sum(axis=1)
        self.Q_G = np.arange(len(G2_Q))[G2_Q <= 2 * ecut]
        K_qv = np.dot(ibzk_qc, B_cv)
        G_Gv = G_Qv[self.Q_G]
        self.G2_qG = np.zeros((len(ibzk_qc), len(self.Q_G)))
        for q, K_v in enumerate(K_qv):
            self.G2_qG[q] = ((G_Gv + K_v)**2).sum(1)
        
        self.gd = gd
        self.dv = gd.dv / N_c.prod()
        self.comm = gd.comm

    def bytecount(self, dtype=float):
        return len(self.Q_G) * np.array(1, dtype).itemsize
    
    def zeros(self, n=(), dtype=float):
        assert dtype == complex
        if isinstance(n, int):
            n = (n,)
        shape = n + self.Q_G.shape
        return np.zeros(shape, complex)
    
    def empty(self, n=(), dtype=float):
        assert dtype == complex
        if isinstance(n, int):
            n = (n,)
        shape = n + self.Q_G.shape
        return np.empty(shape, complex)
    
    def fft(self, a_xR):
        a_xQ = fftn(a_xR, axes=(-3, -2, -1))
        return a_xQ.reshape(a_xR.shape[:-3] + (-1,))[..., self.Q_G].copy()

    def ifft(self, a_xG):
        xshape = a_xG.shape[:-1]
        a_xQ = self.gd.zeros(xshape, complex)
        a_xQ.reshape(xshape + (-1,))[..., self.Q_G] = a_xG
        return ifftn(a_xQ, axes=(-3, -2, -1)).copy()


class Preconditioner:
    def __init__(self, pd):
        self.pd = pd
        self.allocated = True

    def __call__(self, R_G, kpt):
        return R_G / (1.0 + self.pd.G2_qG[kpt.q])


class PWWaveFunctions(FDPWWaveFunctions):
    def __init__(self, ecut, diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd,
                 world, kd, timer):
        self.ecut =  ecut / units.Hartree
        # Set dtype=complex and gamma=False:
        kd.gamma = False
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd, complex,
                                   world, kd, timer)
        
        self.matrixoperator = MatrixOperator(self.bd, self.pd, orthoksl)
        self.wd = self.pd        

    def set_setups(self, setups):

        self.pd = PWDescriptor(self.ecut, self.gd, self.kd.ibzk_qc)
        pt = LFC(self.gd, [setup.pt_j for setup in setups],
                 self.kpt_comm, dtype=self.dtype, forces=True)
        self.pt = PWLFC(pt, self.pd)
        FDPWWaveFunctions.set_setups(self, setups)

    def summary(self, fd):
        fd.write('Mode: Plane waves (%d, ecut=%.3f eV)\n' %
                 (len(self.pd.Q_G), self.pd.ecut * units.Hartree))
        
    def make_preconditioner(self, block=1):
        return Preconditioner(self.pd)

    def apply_pseudo_hamiltonian(self, kpt, hamiltonian, psit_xG, Htpsit_xG):
        """Apply the non-pseudo Hamiltonian i.e. without PAW corrections."""
        Htpsit_xG[:] = 0.5 * self.pd.G2_qG[kpt.q] * psit_xG
        for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
            psit_R = self.pd.ifft(psit_G)
            Htpsit_G += self.pd.fft(psit_R * hamiltonian.vt_sG[kpt.s])

    def add_to_density_from_k_point_with_occupation(self, nt_sR, kpt, f_n):
        nt_R = nt_sR[kpt.s]
        for f, psit_G in zip(f_n, kpt.psit_nG):
            nt_R += f * abs(self.pd.ifft(psit_G))**2

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

    def dict(self, shape=(), derivative=False, zero=False):
        return self.lfc.dict(shape, derivative, zero)

    def set_positions(self, spos_ac):
        self.lfc.set_positions(spos_ac)
        self.my_atom_indices = self.lfc.my_atom_indices
        
    def set_k_points(self, ibzk_qc):
        self.lfc.set_k_points(ibzk_qc)
        N_c = self.pd.gd.N_c
        self.expikr_qR = np.exp(2j * np.pi * np.dot(np.indices(N_c).T,
                                                    (ibzk_qc / N_c).T).T)

    def add(self, a_xG, C_axi, q):
        a_xR = self.pd.gd.zeros(a_xG.shape[:-1], complex)
        self.lfc.add(a_xR, C_axi, q)
        a_xG[:] += self.pd.fft(a_xR / self.expikr_qR[q])

    def integrate(self, a_xG, C_axi, q):
        a_xR = self.pd.ifft(a_xG) * self.expikr_qR[q]
        C_axi[0][:] = 0.0  # XXXXX
        self.lfc.integrate(a_xR, C_axi, q)


class PW: ####### use mode='pw'?  ecut=???
    def __init__(self, ecut=340):
        self.ecut = ecut

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, diagksl, orthoksl, initksl, *args)
        return wfs
