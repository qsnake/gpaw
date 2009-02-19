import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.utilities import pack, unpack2
from gpaw.kpoint import KPoint
from gpaw.transformers import Transformer
from gpaw import mpi


class EmptyWaveFunctions:
    def __nonzero__(self):
        return False

    def set_orthonormalized(self, flag):
        pass

class WaveFunctions(EmptyWaveFunctions):
    """...

    setups:
        List of setup objects.
    symmetry:
        Symmetry object.
    kpt_u:
        List of **k**-point objects.
    nbands: int
        Number of bands.
    nspins: int
        Number of spins.
    dtype: dtype
        Data type of wave functions (float or complex).
    bzk_kc: ndarray
        Scaled **k**-points used for sampling the whole
        Brillouin zone - values scaled to [-0.5, 0.5).
    ibzk_kc: ndarray
        Scaled **k**-points in the irreducible part of the
        Brillouin zone.
    weight_k: ndarray
        Weights of the **k**-points in the irreducible part
        of the Brillouin zone (summing up to 1).
    kpt_comm:
        MPI-communicator for parallelization over **k**-points.
    """
    def __init__(self, gd, nspins, setups, nbands, mynbands, dtype,
                 world, kpt_comm, band_comm,
                 gamma, bzk_kc, ibzk_kc, weight_k, symmetry):
        self.gd = gd
        self.nspins = nspins
        self.nbands = nbands
        self.mynbands = mynbands
        self.dtype = dtype
        self.world = world
        self.kpt_comm = kpt_comm
        self.band_comm = band_comm
        self.gamma = gamma
        self.bzk_kc = bzk_kc
        self.ibzk_kc = ibzk_kc
        self.weight_k = weight_k
        self.symmetry = symmetry
        self.rank_a = None

        self.nibzkpts = len(weight_k)

        # Total number of k-point/spin combinations:
        nks = self.nibzkpts * nspins

        # Number of k-point/spin combinations on this cpu:
        mynks = nks // kpt_comm.size

        ks0 = kpt_comm.rank * mynks
        k0 = ks0 % self.nibzkpts
        self.kpt_u = []
        sdisp_cd = gd.domain.sdisp_cd
        for ks in range(ks0, ks0 + mynks):
            s, k = divmod(ks, self.nibzkpts)
            q = k - k0
            weight = weight_k[k] * 2 / nspins
            if gamma:
                phase_cd = None
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * ibzk_kc[k, :, np.newaxis])
            self.kpt_u.append(KPoint(weight, s, k, q, phase_cd))

        self.ibzk_qc = ibzk_kc[k0:k + 1]

        self.eigensolver = None
        self.timer = None
        
        self.set_setups(setups)

    def set_setups(self, setups):
        self.setups = setups

    def __nonzero__(self):
        return True

    def calculate_density(self, density):
        """Calculate density from wave functions."""
        nt_sG = density.nt_sG
        nt_sG.fill(0.0)
        for kpt in self.kpt_u:
            self.add_to_density_from_k_point(nt_sG, kpt)
        self.band_comm.sum(nt_sG)
        self.kpt_comm.sum(nt_sG)

        if self.symmetry:
            for nt_G in nt_sG:
                self.symmetry.symmetrize(nt_G, self.gd)

    def calculate_atomic_density_matrices_k_point(self, D_sii, kpt, a):
        if kpt.rho_MM is not None: 
            P_Mi = kpt.P_aMi[a] 
            D_sii[kpt.s] += np.dot(np.dot(P_Mi.T.conj(), kpt.rho_MM), 
                                   P_Mi).real 
        else: 
            P_ni = kpt.P_ani[a] 
            D_sii[kpt.s] += np.dot(P_ni.T.conj() * kpt.f_n, P_ni).real

        if hasattr(kpt, 'ft_omn'):
            for i in range(len(kpt.ft_omn)):
                D_sii[kpt.s] += (np.dot(P_ni.T.conj(),
                                        np.dot(kpt.ft_omn[i],
                                               P_ni))).real
                
    def calculate_atomic_density_matrices_k_point_with_occupation(self, D_sii,
                                                                  kpt, a, f_n):
        if kpt.rho_MM is not None: 
            P_Mi = kpt.P_aMi[a]
            rho_MM = np.dot(kpt.C_nM.conj().T * f_n, kpt.C_nM)
            D_sii[kpt.s] += np.dot(np.dot(P_Mi.T.conj(), kpt.rho_MM), 
                                   P_Mi).real 
        else: 
            P_ni = kpt.P_ani[a] 
            D_sii[kpt.s] += np.dot(P_ni.T.conj() * f_n, P_ni).real 

    def calculate_atomic_density_matrices(self, density):
        """Calculate atomic density matrices from projections."""
        D_asp = density.D_asp
        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((self.nspins, ni, ni))
            for kpt in self.kpt_u:
                self.calculate_atomic_density_matrices_k_point(D_sii, kpt, a)

            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.band_comm.sum(D_sp)
            self.kpt_comm.sum(D_sp)

        self.symmetrize_atomic_density_matrices(D_asp)

    def calculate_atomic_density_matrices_with_occupation(self, D_asp, f_kn):
        """Calculate atomic density matrices from projections with
        custom occupation f_kn."""
        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((self.nspins, ni, ni))
            for f_n, kpt in zip(f_kn, self.kpt_u):
                self.calculate_atomic_density_matrices_k_point_with_occupation(
                    D_sii, kpt, a, f_n)

            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.band_comm.sum(D_sp)
            self.kpt_comm.sum(D_sp)

        self.symmetrize_atomic_density_matrices(D_asp)

    def symmetrize_atomic_density_matrices(self, D_asp):
        if self.symmetry:
            all_D_asp = []
            for a, setup in enumerate(self.setups):
                D_sp = D_asp.get(a)
                if D_sp is None:
                    ni = setup.ni
                    D_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                self.gd.comm.broadcast(D_sp, self.rank_a[a])
                all_D_asp.append(D_sp)

            for s in range(self.nspins):
                D_aii = [unpack2(D_sp[s]) for D_sp in all_D_asp]
                for a, D_sp in D_asp.items():
                    setup = self.setups[a]
                    D_sp[s] = pack(setup.symmetrize(a, D_aii,
                                                    self.symmetry.maps))

    def set_positions(self, spos_ac):
        self.rank_a = self.gd.domain.get_ranks_from_positions(spos_ac)
        if self.symmetry is not None:
            self.symmetry.check(spos_ac)

    def collect_eigenvalues(self, k, s):
        return self.collect_array('eps_n', k, s)
    
    def collect_occupations(self, k, s):
        return self.collect_array('f_n', k, s)
    
    def collect_array(self, name, k, s):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = divmod(k + self.nibzkpts * s, len(kpt_u))

        if self.kpt_comm.rank == kpt_rank:
            a_n = getattr(kpt_u[u], name)

            # Delta SCF hack
            if name == 'f_n' and hasattr(kpt_u[u], 'ft_omn'):
                for ft_mn in self.kpt_u[u].ft_omn:
                    a_n += np.diagonal(ft_mn).real

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if self.band_comm.size == 1:
                    if kpt_rank == 0:
                        return a_n
                    else:
                        self.kpt_comm.send(a_n, 0, 1301)
                else:
                    if self.band_comm.rank == 0:
                        b_n = np.zeros(self.nbands)
                    else:
                        b_n = None
                    self.band_comm.gather(a_n, 0, b_n)
                    if self.band_comm.rank == 0:
                        if kpt_rank == 0:
                            return b_n
                        else:
                            self.kpt_comm.send(b_n, 0, 1301)

        elif self.world.rank == 0 and kpt_rank != 0:
            b_n = np.zeros(self.nbands)
            self.kpt_comm.receive(b_n, kpt_rank, 1301)
            return b_n


from gpaw.lcao.overlap import TwoCenterIntegrals
class LCAOWaveFunctions(WaveFunctions):
    def __init__(self, *args):
        WaveFunctions.__init__(self, *args)
        self.basis_functions = None
        self.tci = None
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        
        if not self.basis_functions:
            # First time:
            self.basis_functions = BasisFunctions(self.gd,
                                                  [setup.phit_j
                                                   for setup in self.setups],
                                                  self.kpt_comm,
                                                  cut=True)
            if not self.gamma:
                self.basis_functions.set_k_points(self.ibzk_qc)
        self.basis_functions.set_positions(spos_ac)

        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.mynbands
        
        if not self.tci:
            # First time:
            self.tci = TwoCenterIntegrals(self.gd.domain, self.setups,
                                          self.gamma, self.ibzk_qc)
            
            self.S_qMM = np.empty((nq, nao, nao), self.dtype)
            self.T_qMM = np.empty((nq, nao, nao), self.dtype)
            for kpt in self.kpt_u:
                q = kpt.q
                kpt.S_MM = self.S_qMM[q]
                kpt.T_MM = self.T_qMM[q]
                kpt.C_nM = np.empty((mynbands, nao), self.dtype)

        for kpt in self.kpt_u:
            kpt.P_ani = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            for kpt in self.kpt_u:
                kpt.P_ani[a] = np.empty((mynbands, ni), self.dtype)
            
        self.P_aqMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            self.P_aqMi[a] = np.empty((nq, nao, ni), self.dtype)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.P_aMi = dict([(a, P_qMi[q])
                              for a, P_qMi in self.P_aqMi.items()])

        self.tci.set_positions(spos_ac)
        self.tci.calculate(spos_ac, self.S_qMM, self.T_qMM, self.P_aqMi,
                           self.dtype)
            
    def initialize(self, density, hamiltonian, spos_ac):
        density.initialize_from_atomic_densities(self.basis_functions)
        comp_charge = density.calculate_multipole_moments()
        density.normalize(comp_charge)
        density.mix(comp_charge)
        hamiltonian.update(density)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        
        rho_MM = np.dot(kpt.C_nM.conj().T * f_n, kpt.C_nM)
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.k)

    def add_to_density_from_k_point(self, nt_sG, kpt):
        """Add contribution to pseudo electron-density. """
                
        if kpt.rho_MM is not None:
            rho_MM = kpt.rho_MM
        else:
            rho_MM = np.dot(kpt.C_nM.conj().T * kpt.f_n, kpt.C_nM)
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.q)

    def calculate_forces(self, hamiltonian, F_av):
        spos_ac = hamiltonian.vbar.spos_ac # XXX ugly way to obtain spos_ac

        # This will recalculate everything, which again is not necessary
        # But it won't bother non-force calculations
        tci = TwoCenterIntegrals(self.gd.domain, self.setups,
                                 self.gamma, self.ibzk_qc)
        tci.lcao_forces = True
        tci.set_positions(spos_ac)
        S_qMM = np.empty(self.S_qMM.shape, self.dtype)
        T_qMM = np.empty(self.T_qMM.shape, self.dtype)
        P_aqMi = dict([(a, np.zeros(P.shape, self.dtype))
                       for a, P in self.P_aqMi.items()])
        
        tci.calculate(spos_ac, S_qMM, T_qMM, P_aqMi, self.dtype)

        # Things required for force calculations
        dSdR_qvMM = tci.dSdR_kcmm
        dTdR_qvMM = tci.dTdR_kcmm
        dPdR_aqvMi = tci.dPdR_akcmi
        dThetadR_qvMM = tci.dThetadR_kcmm
        mask_aMM = tci.mask_amm

        for kpt in self.kpt_u:            
            self.calculate_forces_kpoint_lcao(kpt, hamiltonian,
                                              F_av, tci,
                                              S_qMM[kpt.k],
                                              T_qMM[kpt.k],
                                              P_aqMi)

    def get_projector_derivatives(self, tci, a, c, k):
        # Get dPdRa, i.e. derivative of all projector overlaps
        # with respect to the position of *this* atom.  That includes
        # projectors from *all* atoms.
        #
        # Some overlap derivatives must be multiplied by 0 or -1
        # depending on which atom is moved.  This is a temporary hack.
        # 
        # For some reason the "masks" for atoms *before* this one must be
        # multiplied by -1, whereas those *after* must not.
        #
        # Also, for this atom, we must apply a mask which is -1 for
        # m < self.m, and +1 for m > self.m + self.setup.niAO
        dPdRa_ami = {}
        m = tci.M_a[a]
        mask_m = np.zeros(self.setups.nao)
        mask_m[m:m + self.setups[a].niAO] = 1.
        
        for b in self.basis_functions.my_atom_indices:
            if b == a:
                ownmask_m = np.zeros(self.setups.nao)
                m1 = m
                m2 = m + self.setups[a].niAO
                ownmask_m[:m1] = -1.
                ownmask_m[m2:] = +1.
                selfcontrib = (tci.dPdR_akcmi[a][k, c, :, :] * 
                               ownmask_m[None].T)
                dPdRa_ami[b] = selfcontrib
            else:
                if b > a:
                    factor = 1.0
                else:
                    factor = -1.0
                dPdRa_mi = (tci.dPdR_akcmi[b][k, c, :, :] * 
                            mask_m[None].T * factor)
                dPdRa_ami[b] = dPdRa_mi
        return dPdRa_ami

    def get_overlap_derivatives(self, tci, a, c, dPdRa_ami, k):
        nao = self.setups.nao
        dThetadR_mm = tci.dThetadR_kcmm[k, c, :, :]
        pawcorrection_mm = np.zeros((nao, nao), self.dtype)

        for b in self.basis_functions.my_atom_indices:
            O_ii = self.setups[b].O_ii
            dPdRa_mi = dPdRa_ami[b]            
            P_mi = self.P_aqMi[b][k]
            A_mm = np.dot(dPdRa_mi, np.dot(O_ii, P_mi.T.conj()))
            B_mm = np.dot(P_mi, np.dot(O_ii, dPdRa_mi.T.conj()))
            pawcorrection_mm += A_mm + B_mm
        self.basis_functions.gd.comm.sum(pawcorrection_mm)
        
        return dThetadR_mm * tci.mask_amm[a] + pawcorrection_mm

    def calculate_all_potential_derivatives(self, tci, hamiltonian, kpt,
                                            rho_MM):
        nao = self.setups.nao
        vt_G = hamiltonian.vt_sG[kpt.s]
        dEdndndR_av = np.zeros((len(self.setups), 3))
        rho_hc_MM = rho_MM.T.conj()

        DVt_MMv = np.zeros((nao, nao, 3), self.dtype)
        self.basis_functions.calculate_potential_matrix_derivative(vt_G,
                                                                   DVt_MMv,
                                                                   kpt.q)
        self.gd.comm.sum(DVt_MMv)

        for b in self.basis_functions.my_atom_indices:
            M1 = self.basis_functions.M_a[b]
            M2 = M1 + self.setups[b].niAO
            for v in range(3):
                forcecontrib = -2 * np.dot(DVt_MMv[M1:M2, :, v],
                                           rho_hc_MM[:, M1:M2]).real.trace()
                dEdndndR_av[b, v] = forcecontrib

        #self.gd.comm.sum(dEdndndR_av) #??
        return dEdndndR_av

    def calculate_potential_derivatives(self, tci, a, hamiltonian, kpt,
                                        grid_bfs, rho_MM):
        raise DeprecationWarning('code to be removed, using new LF interface')
        nao = self.setups.nao
        my_nao = self.setups[a].niAO
        m1 = tci.M_a[a]
        m2 = m1 + my_nao
        dtype = self.dtype
        vt_G = hamiltonian.vt_sG[kpt.s]
        dVtdRa_mMc = np.zeros((nao, my_nao, 3), dtype)

        phit_mG = np.zeros((nao,) + vt_G.shape, dtype)
        for b, setup in enumerate(self.setups):
            its_nao = setup.niAO
            M1 = tci.M_a[b]
            M2 = M1 + its_nao

            coef_MM = np.identity(its_nao)
            grid_bfs.lfs_a[b].add(phit_mG[M1:M2], coef_MM, kpt.k)

        for phit_G in phit_mG:
            phit_G *= vt_G

        # Maybe it's possible to do this in a less loop-intensive way
        grid_bfs.lfs_a[a].derivative(phit_mG, dVtdRa_mMc, kpt.k)

        dVtdRa_mmc = np.zeros((nao, nao, 3), dtype)
        dVtdRa_mmc[:, m1:m2, :] -= dVtdRa_mMc

        dVtdRa_MMv = dVtdRa_mmc

        dEdndndR_v = np.empty(3)
        for v in range(3):
            X = np.dot(rho_MM, dVtdRa_MMv[:, :, v])
            dEdndndR_v[v] = 2 * X.real.trace()
        return dEdndndR_v

    def calculate_forces_kpoint_lcao(self, kpt, hamiltonian,
                                     F_av, tci, S_MM, T_MM, P_aqMi):
        k = kpt.k
        if kpt.rho_MM is None:
            rho_MM = np.dot(kpt.C_nM.T.conj() * kpt.f_n, kpt.C_nM)
        else:
            rho_MM = kpt.rho_MM

        dEdndndR_av = self.calculate_all_potential_derivatives(tci,
                                                               hamiltonian,
                                                               kpt,
                                                               rho_MM)
        dTdR_vMM = tci.dTdR_kcmm[k]

        self.eigensolver.calculate_hamiltonian_matrix(hamiltonian, self, kpt)
        H_MM = self.eigensolver.H_MM.copy()
        # H_MM is halfway full of garbage!  Only lower triangle is
        # actually correct.  Create correct H_MM:
        nao = self.setups.nao
        ltri = np.tri(nao)
        utri = np.tri(nao, nao, -1).T
        H_MM[:] = H_MM * ltri + H_MM.T.conj() * utri

        ChcEFC_MM = np.dot(np.dot(np.linalg.inv(S_MM), H_MM), rho_MM)

        # Useful check - whether C^dagger eps f C == S^(-1) H rho
        # Although this won't work if people are supplying a customized rho
        #assert abs(ChcEFC_MM - np.dot(kpt.C_nM.T.conj() * kpt.f_n * kpt.eps_n,
        #                              kpt.C_nM)).max() < 1e-8
        
        for a in range(len(self.setups)):
            dPdR_vMi = tci.dPdR_akcmi[a][k]
            mask_MM = tci.mask_amm[a]
            self.calculate_force_on_atom_lcao(a, kpt, hamiltonian, F_av[a],
                                              tci, T_MM,
                                              P_aqMi,
                                              rho_MM, dTdR_vMM, dPdR_vMi,
                                              mask_MM, H_MM, ChcEFC_MM,
                                              dEdndndR_av)

    def calculate_force_on_atom_lcao(self, a, kpt, hamiltonian, F_v, tci,
                                     T_MM, P_aqMi, rho_MM,
                                     dTdR_vMM, dPdR_vMi, mask_MM, H_MM,
                                     ChcEFC_MM, dEdndndR_av):
        k = kpt.k
        
        for v, (dTdR_MM, dPdR_Mi) in enumerate(zip(dTdR_vMM, dPdR_vMi)):
            dPdRa_aMi = self.get_projector_derivatives(tci, a, v, k)
            dSdRa_MM = self.get_overlap_derivatives(tci, a, v, dPdRa_aMi, k)
            
            dEdrhodrhodR = - np.dot(ChcEFC_MM, dSdRa_MM).real.trace()
            
            dEdTdTdR = np.dot(rho_MM, dTdR_MM * mask_MM).real.trace()
            
            dEdDdDdR = np.zeros(1)
            for b, dPdRa_Mi in dPdRa_aMi.items():
                A_ii = np.dot(dPdRa_Mi.T.conj(), 
                              np.dot(rho_MM, self.P_aqMi[b][k]))
                Hb_ii = unpack(hamiltonian.dH_asp[b][kpt.s])
                dEdDdDdR += 2 * np.dot(Hb_ii, A_ii).real.trace()
            self.gd.comm.sum(dEdDdDdR)
            
            dEdndndR = dEdndndR_av[a][v]
            F = - (dEdrhodrhodR + dEdTdTdR + dEdDdDdR + dEdndndR)
            if a in dPdRa_aMi:
                F_v[v] += F


from gpaw.eigensolvers import get_eigensolver
from gpaw.overlap import Overlap
from gpaw.operators import Laplace
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import unpack
from gpaw.io.tar import TarFileReference

class GridWaveFunctions(WaveFunctions):
    def __init__(self, stencil, *args):
        WaveFunctions.__init__(self, *args)
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype)
        self.set_orthonormalized(False)

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        if not self.gamma:
            self.pt.set_k_points(self.ibzk_qc)

        self.overlap = None

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)

        mynbands = self.mynbands
        for kpt in self.kpt_u:
            kpt.P_ani = {}
        for a in self.pt.my_atom_indices:
            ni = self.setups[a].ni
            for kpt in self.kpt_u:
                kpt.P_ani[a] = np.empty((mynbands, ni), self.dtype)

        if not self.overlap:
            self.overlap = Overlap(self)

    def initialize(self, density, hamiltonian, spos_ac):
        if self.kpt_u[0].psit_nG is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             cut=True)
            if not self.gamma:
                basis_functions.set_k_points(self.ibzk_qc)
            basis_functions.set_positions(spos_ac)
        elif isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            self.initialize_wave_functions_from_restart_file()

        if self.kpt_u[0].psit_nG is not None:
            density.nt_sG = self.gd.empty(self.nspins)
            self.calculate_density(density)
            density.nt_sG += density.nct_G
        elif density.nt_sG is None:
            density.initialize_from_atomic_densities(basis_functions)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xcfunc.gllb:
                hamiltonian.xcfunc.xc.initialize_from_atomic_orbitals(
                    basis_functions)

        comp_charge = density.calculate_multipole_moments()
        density.normalize(comp_charge)
        density.mix(comp_charge)

        hamiltonian.update(density)

        if self.kpt_u[0].psit_nG is None:
            self.initialize_wave_functions_from_basis_functions(
                basis_functions, density, hamiltonian, spos_ac)

    def initialize_wave_functions_from_basis_functions(self,
                                                       basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        if 0:
            for kpt in self.kpt_u:
                kpt.psit_nG = self.gd.zeros(self.mynbands, self.dtype)
            self.random_wave_functions(0)
            return
        
        if self.nbands < self.setups.nao:
            lcaonbands = self.nbands
            lcaomynbands = self.mynbands
        else:
            lcaonbands = self.setups.nao
            lcaomynbands = self.setups.nao
            assert self.band_comm.size == 1

        lcaowfs = LCAOWaveFunctions(self.gd, self.nspins, self.setups,
                                    lcaonbands,
                                    lcaomynbands, self.dtype,
                                    self.world, self.kpt_comm,
                                    self.band_comm,
                                    self.gamma, self.bzk_kc, self.ibzk_kc,
                                    self.weight_k, self.symmetry)
        lcaowfs.basis_functions = basis_functions
        lcaowfs.timer = self.timer
        lcaowfs.set_positions(spos_ac)
        hamiltonian.update(density)
        eigensolver = get_eigensolver('lcao', 'lcao')
        eigensolver.iterate(hamiltonian, lcaowfs)

        # Transfer coefficients ...
        for kpt, lcaokpt in zip(self.kpt_u, lcaowfs.kpt_u):
            kpt.C_nM = lcaokpt.C_nM

        # and get rid of potentially big arrays early:
        del eigensolver, lcaowfs

        for kpt in self.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.mynbands, self.dtype)
            basis_functions.lcao_to_grid(kpt.C_nM, 
                                         kpt.psit_nG[:lcaomynbands], kpt.q)
            kpt.C_nM = None

        if self.mynbands > lcaomynbands:
            # Add extra states.  If the number of atomic orbitals is
            # less than the desired number of bands, then extra random
            # wave functions are added.
            self.random_wave_functions(lcaomynbands)

    def initialize_wave_functions_from_restart_file(self):
        if not isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            return

        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        for kpt in self.kpt_u:
            file_nG = kpt.psit_nG
            kpt.psit_nG = self.gd.empty(self.mynbands, self.dtype)
            # Read band by band to save memory
            for n, psit_G in enumerate(kpt.psit_nG):
                if self.world.rank == 0:
                    big_psit_G = file_nG[n][:]
                else:
                    big_psit_G = None
                self.gd.distribute(big_psit_G, psit_G)
        
    def random_wave_functions(self, nao):
        """Generate random wave functions"""

        gd1 = self.gd.coarsen()
        gd2 = gd1.coarsen()

        psit_G1 = gd1.empty(dtype=self.dtype)
        psit_G2 = gd2.empty(dtype=self.dtype)

        interpolate2 = Transformer(gd2, gd1, 1, self.dtype).apply
        interpolate1 = Transformer(gd1, self.gd, 1, self.dtype).apply

        shape = tuple(gd2.n_c)

        scale = np.sqrt(12 / np.product(gd2.domain.cell_c))

        from numpy.random import random, seed

        seed(4 + mpi.rank)

        for kpt in self.kpt_u:
            for psit_G in kpt.psit_nG[nao:]:
                if self.dtype == float:
                    psit_G2[:] = (random(shape) - 0.5) * scale
                else:
                    psit_G2.real = (random(shape) - 0.5) * scale
                    psit_G2.imag = (random(shape) - 0.5) * scale
                    
                interpolate2(psit_G2, psit_G1, kpt.phase_cd)
                interpolate1(psit_G1, psit_G, kpt.phase_cd)

    def add_to_density_from_k_point(self, nt_sG, kpt):
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                nt_G += f * (psit_G * psit_G.conj()).real

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'ft_omn'):
            for ft_mn in kpt.ft_omn:
                for ft_n, psi_m in zip(ft_mn, kpt.psit_nG):
                    for ft, psi_n in zip(ft_n, kpt.psit_nG):
                        if abs(ft) > 1.e-12:
                            nt_G += (psi_m.conj() * ft * psi_n).real

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                nt_G += f * (psit_G * psit_G.conj()).real

    def orthonormalize(self):
        for kpt in self.kpt_u:
            self.overlap.orthonormalize(self, kpt)
        self.set_orthonormalized(True)

    def initialize2(self, paw):
        khjgkjhgkhjg
        hamiltonian = paw.hamiltonian
        density = paw.density
        eigensolver = paw.eigensolver
        assert not eigensolver.lcao
        self.overlap = paw.overlap
        if not eigensolver.initialized:
            eigensolver.initialize(paw)
        if not self.initialized:
            if self.kpt_u[0].psit_nG is None:
                paw.text('Atomic orbitals used for initialization:', paw.nao)
                if paw.nbands > paw.nao:
                    paw.text('Random orbitals used for initialization:',
                             paw.nbands - paw.nao)
            
                # Now we should find out whether init'ing from file or
                # something else
                self.initialize_wave_functions_from_atomic_orbitals(paw)

            else:
                self.initialize_wave_functions_from_restart_file(paw)

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        nk = len(self.ibzk_kc)
        mynu = len(self.kpt_u)
        kpt_rank, u = divmod(k + nk * s, mynu)
        nn, band_rank = divmod(n, self.band_comm.size)

        psit_nG = self.kpt_u[u].psit_nG
        if psit_nG is None:
            raise RuntimeError('This calculator has no wave functions!')

        size = self.world.size
        rank = self.world.rank
        if size == 1:
            return psit_nG[nn][:]

        if self.kpt_comm.rank == kpt_rank:
            if self.band_comm.rank == band_rank:
                psit_G = self.gd.collect(psit_nG[nn][:])

                if kpt_rank == 0 and band_rank == 0:
                    if rank == 0:
                        return psit_G

                # Domain master send this to the global master
                if self.gd.comm.rank == 0:
                    self.world.send(psit_G, 0, 1398)

        if rank == 0:
            # allocate full wavefunction and receive
            psit_G = self.gd.empty(dtype=self.dtype, global_array=True)
            world_rank = (kpt_rank * self.gd.comm.size *
                          self.band_comm.size +
                          band_rank * self.gd.comm.size)
            self.world.receive(psit_G, world_rank, 1398)
            return psit_G

    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_aniv = self.pt.dict(self.nbands, derivative=True)
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = hamiltonian.setups[a].O_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)
