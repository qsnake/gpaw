import numpy as np
from numpy.fft import fftn, ifftn
import ase.units as units

from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.fd import GridWaveFunctions
from gpaw.hs_operators import MatrixOperator


class PWDescriptor:
    def __init__(self, ecut, gd):
        assert gd.pbc_c.all() and gd.comm.size == 1
        
        self.ecut = ecut

        # Calculate reciprocal lattice vectors:
        N_c = gd.N_c
        i_Qc = np.indices(N_c).reshape((3, -1)).T
        i_Qc += N_c // 2
        i_Qc %= N_c
        i_Qc -= N_c // 2
        B_cv = 2.0 * np.pi * gd.icell_cv
        G_Qv = np.dot(i_Qc, B_cv)
        G2_Q = (G_Qv**2).sum(axis=1).ravel()
        self.mask_Q = G2_Q <= 2 * self.ecut
        self.G2_q = G2_Q[self.mask_Q]

        self.gd = gd
        self.dv = gd.dv / N_c.prod()
        self.comm = gd.comm
        
    def zeros(self, n=(), dtype=float):
        assert dtype == complex
        if isinstance(n, int):
            n = (n,)
        shape = n + self.G2_q.shape
        return np.zeros(shape, complex)
    
    def fft(self, a_xG):
        a_xq = fftn(a_xG, axes=(-3, -2, -1))
        return a_xq.reshape(a_xG.shape[:-3] + (-1,))[..., self.mask_Q].copy()

    def ifft(self, a_xq):
        xshape = a_xq.shape[:-1]
        a_xQ = self.gd.zeros(xshape, complex)
        a_xQ.reshape(xshape + (-1,))[..., self.mask_Q] = a_xq
        return ifftn(a_xQ, axes=(-3, -2, -1)).copy()


class Preconditioner:
    def __init__(self, pd):
        self.pd = pd
        self.allocated = True

    def __call__(self, R_q, phase_cd, psit_q):
        return 0.01 * R_q


class PWWaveFunctions(GridWaveFunctions):
    def __init__(self, ecut, diagksl, orthoksl, initksl, gd, *args):
        self.pd = PWDescriptor(ecut / units.Hartree, gd)
        GridWaveFunctions.__init__(self, 1, diagksl, orthoksl, initksl,
                                   gd, *args)
        del self.kin
        self.matrixoperator = MatrixOperator(self.bd, self.pd, orthoksl)
        self.wd = self.pd

    def set_setups(self, setups):
        GridWaveFunctions.set_setups(self, setups)
        self.pt = PWLFC(self.pt, self.pd)
        
    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.lfc.my_atom_indices)
        self.positions_set = True

    def make_preconditioner(self):
        return Preconditioner(self.pd)

    def apply_hamiltonian(self, hamiltonian, kpt, psit_xq, Htpsit_xq):
        Htpsit_xq[:] = 0.5 * psit_xq * self.pd.G2_q
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
        GridWaveFunctions.initialize_wave_functions_from_basis_functions(
            self, basis_functions, density, hamiltonian, spos_ac)

        for kpt in self.kpt_u:
            kpt.psit_nG = self.pd.fft(kpt.psit_nG)

    def estimate_memory(self, mem):
        gridbytes = self.gd.bytecount(self.dtype)
        mem.subnode('Arrays psit_nG', 
                    len(self.kpt_u) * self.mynbands * gridbytes)
        #self.eigensolver.estimate_memory(mem.subnode('Eigensolver'), self.gd,
        #                                 self.dtype, self.mynbands,
        #                                 self.nbands)
        self.pt.estimate_memory(mem.subnode('Projectors'))
        self.overlap.estimate_memory(mem.subnode('Overlap op'), self.dtype)


class PWLFC:
    def __init__(self, lfc, pd):
        self.lfc = lfc
        self.pd = pd

    def set_positions(self, spos_ac):
        self.lfc.set_positions(spos_ac)

    def add(self, a_xq, C_axi, q):
        a_xG = self.pd.gd.zeros(a_xq.shape[:-1], complex)
        self.lfc.add(a_xG, C_axi, q)
        a_xq[:] += self.pd.fft(a_xG) 

    def integrate(self, a_xq, C_axi, q):
        a_xG = self.pd.ifft(a_xq)
        self.lfc.integrate(a_xG, C_axi, q)


class PW:
    def __init__(self, ecut=340):
        self.ecut = ecut

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, diagksl, orthoksl, initksl, *args)
        return wfs
