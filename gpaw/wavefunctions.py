import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.utilities import pack, unpack2
from gpaw.utilities.tools import tri2full
from gpaw.kpoint import KPoint
from gpaw.transformers import Transformer
from gpaw.operators import Gradient
from gpaw import mpi


class EmptyWaveFunctions:
    def __nonzero__(self):
        return False

    def set_orthonormalized(self, flag):
        pass

    def estimate_memory(self, mem):
        mem.set('Unknown WFs', 0)

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
        kpt_u = []
        sdisp_cd = gd.sdisp_cd
        for ks in range(ks0, ks0 + mynks):
            s, k = divmod(ks, self.nibzkpts)
            q = k - k0
            weight = weight_k[k] * 2 / nspins
            if gamma:
                phase_cd = np.ones((3, 2), complex)
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * ibzk_kc[k, :, np.newaxis])
            kpt_u.append(KPoint(weight, s, k, q, phase_cd))

        self.kpt_u = kpt_u
        self.ibzk_qc = ibzk_kc[k0:k + 1]

        self.eigensolver = None
        self.timer = None
        self.positions_set = False
        
        self.set_setups(setups)

    def set_setups(self, setups):
        self.setups = setups

    def set_eigensolver(self, eigensolver):
        self.eigensolver = eigensolver

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

        if hasattr(kpt, 'c_on'):
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                ft_mn = ne * np.outer(c_n.conj(), c_n)
                D_sii[kpt.s] += (np.dot(P_ni.T.conj(),
                                        np.dot(ft_mn, P_ni))).real

    def calculate_atomic_density_matrices_k_point_with_occupation(self, D_sii,
                                                                  kpt, a, f_n):
        # XXX This method appears to be unused, deprecating
        raise DeprecationWarning
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
        # XXX This method appears to be unused, deprecating
        raise DeprecationWarning
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
        self.rank_a = self.gd.get_ranks_from_positions(spos_ac)
        if self.symmetry is not None:
            self.symmetry.check(spos_ac)

    def allocate_arrays_for_projections(self, my_atom_indices):
        if not self.positions_set and self.kpt_u[0].P_ani is not None:
            # Projections have been read from file - don't delete them!
            if self.gd.comm.size == 1:
                pass
            else:
                # Redistribute P_ani among domains.  Not implemented:
                self.kpt_u[0].P_ani = None
                self.allocate_arrays_for_projections(my_atom_indices)
        else:
            for kpt in self.kpt_u:
                kpt.P_ani = {}
            for a in my_atom_indices:
                ni = self.setups[a].ni
                for kpt in self.kpt_u:
                    kpt.P_ani[a] = np.empty((self.mynbands, ni), self.dtype)

    def collect_eigenvalues(self, k, s):
        return self.collect_array('eps_n', k, s)
    
    def collect_occupations(self, k, s):
        return self.collect_array('f_n', k, s)

    def collect_array(self, name, k, s, subset=None):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = divmod(k + self.nibzkpts * s, len(kpt_u))

        if self.kpt_comm.rank == kpt_rank:
            a_n = getattr(kpt_u[u], name)

            if subset is not None:
                a_n = a_n[subset]

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

    def collect_auxiliary(self, name, k, s, shape=1):
        """Helper method for collecting band-independent scalars/arrays.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = divmod(k + self.nibzkpts * s, len(kpt_u))

        if self.kpt_comm.rank == kpt_rank:
            a_o = getattr(kpt_u[u], name)

            # Make sure data is a mutable object
            a_o = np.asarray(a_o)

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if kpt_rank == 0:
                    return a_o
                else:
                    self.kpt_comm.send(a_o, 0, 1302)

        elif self.world.rank == 0 and kpt_rank != 0:
            b_o = np.zeros(shape)
            self.kpt_comm.receive(b_o, kpt_rank, 1302)
            return b_o


from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.utilities.blas import gemm
class LCAOWaveFunctions(WaveFunctions):
    def __init__(self, *args):
        WaveFunctions.__init__(self, *args)
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None
        self.tci = TwoCenterIntegrals(self.gd, self.setups,
                                      self.gamma, self.ibzk_qc)
        self.basis_functions = BasisFunctions(self.gd,
                                              [setup.phit_j
                                               for setup in self.setups],
                                              self.kpt_comm,
                                              cut=True)
        if not self.gamma:
            self.basis_functions.set_k_points(self.ibzk_qc)

    def set_eigensolver(self, eigensolver):
        WaveFunctions.set_eigensolver(self, eigensolver)
        eigensolver.initialize(self.gd, self.band_comm, self.dtype, 
                               self.setups.nao, self.mynbands)

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)        
        self.basis_functions.set_positions(spos_ac)

        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.mynbands
        
        if self.S_qMM is None: # XXX
            # First time:
            self.S_qMM = np.empty((nq, nao, nao), self.dtype)
            self.T_qMM = np.empty((nq, nao, nao), self.dtype)
            for kpt in self.kpt_u:
                q = kpt.q
                kpt.S_MM = self.S_qMM[q]
                kpt.T_MM = self.T_qMM[q]
                kpt.C_nM = np.empty((mynbands, nao), self.dtype)

        self.allocate_arrays_for_projections(
            self.basis_functions.my_atom_indices)
            
        self.P_aqMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            self.P_aqMi[a] = np.empty((nq, nao, ni), self.dtype)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.P_aMi = dict([(a, P_qMi[q])
                              for a, P_qMi in self.P_aqMi.items()])

        self.tci.set_positions(spos_ac)
        self.tci.calculate(spos_ac, self.S_qMM, self.T_qMM, self.P_aqMi)
            
        self.positions_set = True

    def initialize(self, density, hamiltonian, spos_ac):
        if density.nt_sG is None:
            density.initialize_from_atomic_densities(self.basis_functions)
        comp_charge = density.calculate_multipole_moments()
        density.normalize(comp_charge)
        density.mix(comp_charge)
        hamiltonian.update(density)

    def calculate_density_matrix(self, f_n, C_nM, rho_MM):
        # XXX Should not conjugate, but call gemm(..., 'c')
        # Although that requires knowing C_Mn and not C_nM.
        # (that also conforms better to the usual conventions in literature)
        Cf_Mn = C_nM.T.conj() * f_n
        gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        # Where is this function used? XXX deprecate/remove if not used.
        raise DeprecationWarning
        rho_MM = np.dot(kpt.C_nM.conj().T * f_n, kpt.C_nM)
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.k)

    def add_to_density_from_k_point(self, nt_sG, kpt):
        """Add contribution to pseudo electron-density. """
        if kpt.rho_MM is not None:
            rho_MM = kpt.rho_MM
        else:
            # XXX do we really want to allocate this array each time?
            nao = self.setups.nao
            rho_MM = np.empty((nao, nao), self.dtype)
            self.calculate_density_matrix(kpt.f_n, kpt.C_nM, rho_MM)
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.q)

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        raise NotImplementedError('Kinetic density calculation for LCAO '
                                  'wavefunctions is not implemented.')

    def calculate_forces(self, hamiltonian, F_av):
        spos_ac = hamiltonian.vbar.spos_ac # XXX ugly way to obtain spos_ac
        nao = self.setups.nao
        nq = len(self.ibzk_qc)
        dtype = self.dtype
        dThetadR_qvMM = np.empty((nq, 3, nao, nao), dtype)
        dTdR_qvMM = np.empty((nq, 3, nao, nao), dtype)
        dPdR_aqvMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
        self.tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM,
                                      dPdR_aqvMi)
        
        for kpt in self.kpt_u:
            self.calculate_forces_by_kpoint(kpt, hamiltonian,
                                            F_av, self.tci,
                                            self.S_qMM[kpt.q],
                                            self.T_qMM[kpt.q],
                                            self.P_aqMi,
                                            dThetadR_qvMM[kpt.q],
                                            dTdR_qvMM[kpt.q],
                                            dPdR_aqvMi)

    def print_arrays_with_ranks(self, names, arrays_nax):
        # Debugging function for checking properties of distributed arrays
        # Prints rank, label, list of atomic indices, and element sum
        # for parts of array on this cpu as a primitive "hash" function
        my_atom_indices = self.basis_functions.my_atom_indices
        from gpaw.mpi import rank
        for name, array_ax in zip(names, arrays_nax):
            sums = [array_ax[a].sum() for a in my_atom_indices]
            print rank, name, my_atom_indices, sums

    def calculate_forces_by_kpoint(self, kpt, hamiltonian,
                                   F_av, tci, S_MM, T_MM, P_aqMi,
                                   dThetadR_vMM, dTdR_vMM, dPdR_aqvMi):
        k = kpt.k
        q = kpt.q
        nao = self.setups.nao
        dtype = self.dtype
        if kpt.rho_MM is None:
            rho_MM = np.empty((nao, nao), dtype)
            self.calculate_density_matrix(kpt.f_n, kpt.C_nM, rho_MM)
        else:
            rho_MM = kpt.rho_MM
        
        basis_functions = self.basis_functions
        my_atom_indices = basis_functions.my_atom_indices
        atom_indices = basis_functions.atom_indices        
        
        def gemmdot(a_ik, b_kj):
            assert a_ik.flags.contiguous
            assert b_kj.flags.contiguous
            assert a_ik.dtype == b_kj.dtype
            c_ij = np.empty((a_ik.shape[0], b_kj.shape[-1]), a_ik.dtype)
            gemm(1.0, b_kj, a_ik, 0.0, c_ij, 'n')
            return c_ij
        
        def _slices(indices):
            for a in indices:
                M1 = basis_functions.M_a[a]
                M2 = M1 + self.setups[a].niAO
                yield a, M1, M2
        
        def slices():
            return _slices(atom_indices)
        
        def my_slices():
            return _slices(my_atom_indices)
        
        
        self.eigensolver.calculate_hamiltonian_matrix(hamiltonian, self, kpt)
        H_MM = self.eigensolver.H_MM
        tri2full(H_MM)
        
        #
        #         -----                    -----
        #          \    -1                  \    *
        # E      =  )  S     H    rho     =  )  c     eps  f  c
        #  mu nu   /    mu x  x z    z nu   /    n mu    n  n  n nu
        #         -----                    -----
        #          x z                       n
        #
        # We use the transpose of that matrix
        ET_MM = np.linalg.solve(S_MM, gemmdot(H_MM, rho_MM)).T.copy()
        
        # Useful check - whether C^dagger eps f C == S^(-1) H rho
        # Although this won't work if people are supplying a customized rho
        #assert abs(ET_MM - np.dot(kpt.C_nM.T.conj() * kpt.f_n * kpt.eps_n,
        #                          kpt.C_nM)).max() < 1e-8
        
        rhoT_MM = rho_MM.T.copy()
        del rho_MM
        
        # Kinetic energy contribution
        dET_av = np.zeros_like(F_av)
        dEdTrhoT_vMM = (dTdR_vMM * rhoT_MM[np.newaxis]).real
        for a, M1, M2 in my_slices():
            dET_av[a, :] = -2 * dEdTrhoT_vMM[:, M1:M2].sum(-1).sum(-1)
        del dEdTrhoT_vMM
        
        # Potential contribution
        dEn_av = np.zeros_like(F_av)
        vt_G = hamiltonian.vt_sG[kpt.s]
        DVt_MMv = np.zeros((nao, nao, 3), dtype)
        basis_functions.calculate_potential_matrix_derivative(vt_G, DVt_MMv, q)
        for a, M1, M2 in slices():
            for v in range(3):
                dEn_av[a, v] = -2 * (DVt_MMv[M1:M2, :, v]
                                     * rhoT_MM[M1:M2, :]).real.sum()
        del DVt_MMv
        
        # Density matrix contribution due to basis overlap
        dErho_av = np.zeros_like(F_av)
        dThetadRE_vMM = (dThetadR_vMM * ET_MM[np.newaxis]).real
        for a, M1, M2 in my_slices():
            dErho_av[a, :] = 2 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
        del dThetadRE_vMM

        # Density matrix contribution from PAW correction
        dPdR_avMi = dict([(a, dPdR_aqvMi[a][q]) for a in my_atom_indices])
        work_MM = np.empty((nao, nao), dtype)
        for v in range(3):
            for b in my_atom_indices:
                setup = self.setups[b]
                O_ii = np.asarray(setup.O_ii, dtype)
                dOP_iM = np.empty((setup.ni, nao), dtype)
                gemm(1.0, self.P_aqMi[b][q], O_ii, 0.0, dOP_iM, 'c')
                gemm(1.0, dOP_iM, dPdR_avMi[b][v], 0.0, work_MM, 'n')
                ZE_MM = (work_MM * ET_MM).real
                for a, M1, M2 in slices():
                    if a != b:
                        dE = np.sign(a - b) * ZE_MM[M1:M2].sum()
                    if a == b:
                        dE = ZE_MM[:M1].sum() - ZE_MM[M2:].sum()
                    dErho_av[a, v] += 2 * dE
        del work_MM, ZE_MM
        
        # Atomic density contribution
        dED_av = np.zeros_like(F_av)
        for v in range(3):
            for b in my_atom_indices:
                dPdR_Mi = dPdR_avMi[b][v]
                Hb_ii = unpack(hamiltonian.dH_asp[b][kpt.s])
                PH_Mi = gemmdot(self.P_aqMi[b][q], np.asarray(Hb_ii, dtype))
                PHPrhoT_MM = gemmdot(PH_Mi, np.conj(dPdR_Mi.T)) * rhoT_MM
                for a, M1, M2 in slices():
                    if a != b:
                        dE = np.sign(b - a) * PHPrhoT_MM[:, M1:M2].real.sum()
                    else:
                        dE = (PHPrhoT_MM[:, M2:].real.sum()
                              - PHPrhoT_MM[:, :M1].real.sum())
                    dED_av[a, v] += 2 * dE
        
        F_av -= (dET_av + dEn_av + dErho_av + dED_av)

    def estimate_memory(self, mem):
        nbands = self.nbands
        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        ni_total = sum([setup.ni for setup in self.setups])
        itemsize = np.array(1, self.dtype).itemsize
        mem.subnode('C [qnM]', nq * nbands * nao * itemsize)
        mem.subnode('T, S [qMM]', 2 * nq * nao * nao * itemsize)
        mem.subnode('P [aqMi]', nq * nao * ni_total / self.gd.comm.size)
        self.tci.estimate_memory(mem.subnode('TCI'))
        self.basis_functions.estimate_memory(mem.subnode('BasisFunctions'))
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'))


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

        self.allocate_arrays_for_projections(self.pt.my_atom_indices)

        if not self.overlap:
            self.overlap = Overlap(self)

        self.positions_set = True

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
            self.timer.start('Wavefunction: random')
            for kpt in self.kpt_u:
                kpt.psit_nG = self.gd.zeros(self.mynbands, self.dtype)
            self.random_wave_functions(0)
            self.timer.stop('Wavefunction: random')
            return
        
        self.timer.start('Wavefunction: lcao')
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
        eigensolver.initialize(self.gd, self.band_comm, self.dtype,
                               self.setups.nao, lcaomynbands)
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
        self.timer.stop('Wavefunction: lcao')

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
                if self.gd.comm.rank == 0:
                    big_psit_G = np.array(file_nG[n][:], self.dtype)
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

        scale = np.sqrt(12 / np.product(gd2.cell_c))

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
        if hasattr(kpt, 'c_on'):
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                ft_mn = ne * np.outer(c_n.conj(), c_n)
                for ft_n, psi_m in zip(ft_mn, kpt.psit_nG):
                    for ft, psi_n in zip(ft_n, kpt.psit_nG):
                        if abs(ft) > 1.e-12:
                            nt_G += (psi_m.conj() * ft * psi_n).real

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        # Appears to be unused
        raise DeprecationWarning
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                nt_G += f * (psit_G * psit_G.conj()).real

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        """Add contribution to pseudo kinetic energy density."""
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
            dwork_G = self.gd.empty(dtype=self.dtype)
            if self.dtype == float:
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    ft_mn = ne * np.outer(c_n.conj(), c_n)
                    for ft_n, psit_m in zip(ft_mn, kpt.psit_nG):
                        d_c[c](psit_m, dpsit_G)
                        for ft, psit_n in zip(ft_n, kpt.psit_nG):
                            if abs(ft) > 1.e-12:
                                d_c[c](psit_n, dwork_G)
                                axpy(0.5*ft, dpsit_G * dwork_G, taut_G) #taut_G += 0.5*f*dpsit_G*dwork_G
            else:
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    ft_mn = ne * np.outer(c_n.conj(), c_n)
                    for ft_n, psit_m in zip(ft_mn, kpt.psit_nG):
                        d_c[c](psit_m, dpsit_G, kpt.phase_cd)
                        for ft, psit_n in zip(ft_n, kpt.psit_nG):
                            if abs(ft) > 1.e-12:
                                d_c[c](psit_n, dwork_G, kpt.phase_cd)
                                taut_G += 0.5 * (dpsit_G.conj() * ft * dwork_G).real

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

    def estimate_memory(self, mem):
        # XXX Laplacian operator?
        gridbytes = self.gd.bytecount(self.dtype)
        mem.subnode('psit_unG', len(self.kpt_u) * self.mynbands * gridbytes)
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'), self.gd,
                                         self.dtype, self.mynbands,
                                         self.nbands)
