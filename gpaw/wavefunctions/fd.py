import numpy as np

from gpaw.overlap import Overlap
from gpaw.fd_operators import Laplace
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import unpack
from gpaw.io.tar import TarFileReference
from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.transformers import Transformer
from gpaw.fd_operators import Gradient
from gpaw.band_descriptor import BandDescriptor
from gpaw import extra_parameters
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator
from gpaw.preconditioner import Preconditioner


class FDWaveFunctions(FDPWWaveFunctions):
    def __init__(self, stencil, diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd,
                 dtype, world, kd, timer=None):
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd,
                                   dtype, world, kd, timer)

        self.wd = self.gd  # wave function descriptor
        
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype, allocate=False)

        self.matrixoperator = MatrixOperator(self.bd, self.gd, orthoksl)

    def set_setups(self, setups):
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        FDPWWaveFunctions.set_setups(self, setups)

    def set_positions(self, spos_ac):
        if not self.kin.is_allocated():
            self.kin.allocate()
        FDPWWaveFunctions.set_positions(self, spos_ac)

    def summary(self, fd):
        fd.write('Mode: Finite-difference\n')
        
    def make_preconditioner(self, block=1):
        return Preconditioner(self.gd, self.kin, self.dtype, block)
    
    def apply_hamiltonian(self, hamiltonian, kpt, psit_xG, Htpsit_xG):
        """Apply the non-pseudo Hamiltonian i.e. without PAW corrections."""
        self.kin.apply(psit_xG, Htpsit_xG, kpt.phase_cd)
        hamiltonian.apply_local_potential(psit_xG, Htpsit_xG, kpt.s)
        hamiltonian.xc.add_non_local_terms(psit_xG, Htpsit_xG, kpt)

    def add_orbital_density(self, nt_G, kpt, n):
        if self.dtype == float:
            axpy(1.0, kpt.psit_nG[n]**2, nt_G)
        else:
            axpy(1.0, kpt.psit_nG[n].real**2, nt_G)
            axpy(1.0, kpt.psit_nG[n].imag**2, nt_G)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        # Used in calculation of response part of GLLB-potential
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G.real**2, nt_G)
                axpy(f, psit_G.imag**2, nt_G)

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                            dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            for d_n, psi0_G in zip(d_nn, kpt.psit_nG):
                for d, psi_G in zip(d_n, kpt.psit_nG):
                    if abs(d) > 1.e-12:
                        nt_G += (psi0_G.conj() * d * psi_G).real

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        """Add contribution to pseudo kinetic energy density."""

        if isinstance(kpt.psit_nG, TarFileReference):
            raise RuntimeError('Wavefunctions have not been initialized.')

        d_c = [Gradient(self.gd, c, dtype=self.dtype).apply for c in range(3)]
        dpsit_G = self.gd.empty(dtype=self.dtype)
        if self.dtype == float:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for c in range(3):
                    d_c[c](psit_G, dpsit_G)
                    axpy(0.5*f, dpsit_G**2, taut_G) #taut_G += 0.5*f*dpsit_G**2
        else:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for c in range(3):
                    d_c[c](psit_G, dpsit_G, kpt.phase_cd)
                    taut_G += 0.5 * f * (dpsit_G.conj() * dpsit_G).real

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                            dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            dwork_G = self.gd.empty(dtype=self.dtype)
            if self.dtype == float:
                for d_n, psit0_G in zip(d_nn, kpt.psit_nG):
                    for c in range(3):
                        d_c[c](psit0_G, dpsit_G)
                        for d, psit_G in zip(d_n, kpt.psit_nG):
                            if abs(d) > 1.e-12:
                                d_c[c](psit_G, dwork_G)
                                # taut_G += 0.5*f*dpsit_G*dwork_G
                                axpy(0.5*d, dpsit_G * dwork_G, taut_G)
            else:
                for d_n, psit0_G in zip(d_nn, kpt.psit_nG):
                    for c in range(3):
                        d_c[c](psit0_G, dpsit_G, kpt.phase_cd)
                        for d, psit_G in zip(d_n, kpt.psit_nG):
                            if abs(d) > 1.e-12:
                                d_c[c](psit_G, dwork_G, kpt.phase_cd)
                                taut_G += 0.5 * (dpsit_G.conj() * d *
                                                 dwork_G).real

    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_av.fill(0.0)
        F_aniv = self.pt.dict(self.bd.mynbands, derivative=True)
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = hamiltonian.setups[a].dO_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q) #XXX again
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for a, F_niv in F_aniv.items():
                    F_niv = F_niv.conj()
                    dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                    Q_ni = np.dot(d_nn, kpt.P_ani[a])
                    F_vii = np.dot(np.dot(F_niv.transpose(), Q_ni), dH_ii)
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    dO_ii = hamiltonian.setups[a].dO_ii
                    F_vii -= np.dot(np.dot(F_niv.transpose(), Q_ni), dO_ii)
                    F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

        self.bd.comm.sum(F_av, 0)

        if self.bd.comm.rank == 0:
            self.kpt_comm.sum(F_av, 0)

    def estimate_memory(self, mem):
        FDPWWaveFunctions.estimate_memory(self, mem)
        self.kin.estimate_memory(mem.subnode('Kinetic operator'))
