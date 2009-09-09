import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy, gemm
from gpaw.utilities import pack, unpack2
from gpaw.utilities.tools import tri2full, get_matrix_index
from gpaw.kpoint import KPoint
from gpaw.transformers import Transformer
from gpaw.operators import Gradient
from gpaw.utilities.timing import nulltimer
from gpaw.band_descriptor import BandDescriptor
import gpaw.mpi as mpi
from gpaw import extra_parameters, debug

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
    def __init__(self, gd, nspins, setups, bd, dtype, world, kpt_comm,
                 gamma, bzk_kc, ibzk_kc, weight_k, symmetry, timer=nulltimer):
        self.gd = gd
        self.nspins = nspins
        self.bd = bd
        self.nbands = self.bd.nbands #XXX
        self.mynbands = self.bd.mynbands #XXX
        self.dtype = dtype
        self.world = world
        self.kpt_comm = kpt_comm
        self.band_comm = self.bd.comm #XXX
        self.gamma = gamma
        self.bzk_kc = bzk_kc
        self.ibzk_kc = ibzk_kc
        self.weight_k = weight_k
        self.symmetry = symmetry
        self.timer = timer
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
        self.positions_set = False
        
        self.set_setups(setups)

    def set_setups(self, setups):
        self.setups = setups

    def set_eigensolver(self, eigensolver):
        self.eigensolver = eigensolver

    def __nonzero__(self):
        return True

    def calculate_density_contribution(self, nt_sG):
        """Calculate contribution to pseudo density from wave functions."""
        nt_sG.fill(0.0)
        for kpt in self.kpt_u:
            self.add_to_density_from_k_point(nt_sG, kpt)
        self.band_comm.sum(nt_sG)
        self.kpt_comm.sum(nt_sG)
        
        if self.symmetry:
            for nt_G in nt_sG:
                self.symmetry.symmetrize(nt_G, self.gd)

    def add_to_density_from_k_point(self, nt_sG, kpt):
        self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)
    

    def get_orbital_density_matrix(self, a, kpt, n):
        """Add the nth band density from kpt to density matrix D_sp"""
        ni = self.setups[a].ni
        D_sii = np.zeros((self.nspins, ni, ni))
        P_i = kpt.P_ani[a][n]
        D_sii[kpt.s] += np.outer(P_i.conj(), P_i).real
        D_sp = [pack(D_ii) for D_ii in D_sii]
        return D_sp
    
    def calculate_atomic_density_matrices_k_point(self, D_sii, kpt, a, f_n):
        if kpt.rho_MM is not None:
            P_Mi = kpt.P_aMi[a]
            #P_Mi = kpt.P_aMi_sparse[a]
            #ind = get_matrix_index(kpt.P_aMi_index[a])
            #D_sii[kpt.s] += np.dot(np.dot(P_Mi.T.conj(), kpt.rho_MM),
            #                       P_Mi).real
            rhoP_Mi = np.zeros_like(P_Mi)
            D_ii = np.zeros(D_sii[kpt.s].shape, kpt.rho_MM.dtype)
            #gemm(1.0, P_Mi, kpt.rho_MM[ind.T, ind], 0.0, tmp)
            gemm(1.0, P_Mi, kpt.rho_MM, 0.0, rhoP_Mi)
            gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
            D_sii[kpt.s] += D_ii.real
            #D_sii[kpt.s] += dot(dot(P_Mi.T.conj().copy(),
            #                        kpt.rho_MM[ind.T, ind]), P_Mi).real
        else:
            P_ni = kpt.P_ani[a]
            D_sii[kpt.s] += np.dot(P_ni.T.conj() * f_n, P_ni).real

        if hasattr(kpt, 'c_on'):
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                ft_mn = ne * np.outer(c_n.conj(), c_n)
                D_sii[kpt.s] += (np.dot(P_ni.T.conj(),
                                        np.dot(ft_mn, P_ni))).real
    
    def calculate_atomic_density_matrices(self, D_asp):
        """Calculate atomic density matrices from projections."""
        f_un = [kpt.f_n for kpt in self.kpt_u]
        self.calculate_atomic_density_matrices_with_occupation(D_asp, f_un)

    def calculate_atomic_density_matrices_with_occupation(self, D_asp, f_un):
        """Calculate atomic density matrices from projections with
        custom occupation f_un."""
        # Varying f_n used in calculation of response part of GLLB-potential
        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((self.nspins, ni, ni))
            for f_n, kpt in zip(f_un, self.kpt_u):
                self.calculate_atomic_density_matrices_k_point(D_sii, kpt, a,
                                                               f_n)
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
        rank_a = self.gd.get_ranks_from_positions(spos_ac)

        """
        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        if (self.rank_a is not None and rank_a is not None and
            self.kpt_u[0].P_ani is None and (rank_a == self.gd.comm.rank).any()):
            for kpt in self.kpt_u:
                kpt.P_ani = {}
        """

        # Should we use pt.my_atom_indices or basis_functions.my_atom_indices?
        # Regardless, they are updated after this invocation, so here's a hack:
        my_atom_indices = np.argwhere(rank_a == self.gd.comm.rank).ravel()

        if self.rank_a is not None and self.kpt_u[0].P_ani is not None:
            requests = []
            P_auni = {}
            for a in my_atom_indices:
                if a in self.kpt_u[0].P_ani:
                    P_uni = np.array([kpt.P_ani.pop(a) for kpt in self.kpt_u])
                else:
                    # Get matrix from old domain:
                    mynks = len(self.kpt_u)
                    ni = self.setups[a].ni
                    P_uni = np.empty((mynks, self.mynbands, ni), self.dtype)
                    requests.append(self.gd.comm.receive(P_uni, self.rank_a[a],
                                                         tag=a, block=False))
                P_auni[a] = P_uni
            for a in self.kpt_u[0].P_ani.keys():
                # Send matrix to new domain:
                P_uni = np.array([kpt.P_ani.pop(a) for kpt in self.kpt_u])
                requests.append(self.gd.comm.send(P_uni, rank_a[a],
                                                  tag=a, block=False))
            for request in requests:
                self.gd.comm.wait(request)
            for u, kpt in enumerate(self.kpt_u):
                assert len(kpt.P_ani.keys()) == 0
                kpt.P_ani = dict([(a,P_uni[u],) for a,P_uni in P_auni.items()])

        self.rank_a = rank_a

        if self.symmetry is not None:
            self.symmetry.check(spos_ac)

    def allocate_arrays_for_projections(self, my_atom_indices):
        if not self.positions_set and self.kpt_u[0].P_ani is not None:
            # Projections have been read from file - don't delete them!
            pass
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

    def collect_array(self, name, k, s, subset=None, dtype=float):
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

            if a_n.dtype is not dtype:
                a_n = a_n.astype(dtype)

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if self.band_comm.size == 1:
                    if kpt_rank == 0:
                        return a_n
                    else:
                        self.kpt_comm.send(a_n, 0, 1301)
                else:
                    b_n = self.bd.collect(a_n)
                    if self.band_comm.rank == 0:
                        if kpt_rank == 0:
                            return b_n
                        else:
                            self.kpt_comm.send(b_n, 0, 1301)

        elif self.world.rank == 0 and kpt_rank != 0:
            b_n = np.zeros(self.nbands, dtype=dtype)
            self.kpt_comm.receive(b_n, kpt_rank, 1301)
            return b_n

    def collect_auxiliary(self, name, k, s, shape=1, dtype=float):
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

            if a_o.dtype is not dtype:
                a_o = a_o.astype(dtype)

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if kpt_rank == 0:
                    return a_o
                else:
                    self.kpt_comm.send(a_o, 0, 1302)

        elif self.world.rank == 0 and kpt_rank != 0:
            b_o = np.zeros(shape, dtype=dtype)
            self.kpt_comm.receive(b_o, kpt_rank, 1302)
            return b_o

    def collect_projections(self, k, s):
        """Helper method for collecting projector overlaps across domains.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, send to the global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = divmod(k + self.nibzkpts * s, len(kpt_u))
        P_ani = kpt_u[u].P_ani

        natoms = len(self.rank_a) # it's a hack...
        nproj = sum([setup.ni for setup in self.setups])

        if self.world.rank == 0:
            mynu = len(kpt_u)
            all_P_ni = np.empty((self.nbands, nproj), self.dtype)
            for band_rank in range(self.band_comm.size):
                nslice = self.bd.get_slice(band_rank)
                i = 0
                for a in range(natoms):
                    ni = self.setups[a].ni
                    if kpt_rank == 0 and band_rank == 0 and a in P_ani:
                        P_ni = P_ani[a]
                    else:
                        P_ni = np.empty((self.mynbands, ni), self.dtype)
                        world_rank = (self.rank_a[a] +
                                      kpt_rank * self.gd.comm.size *
                                      self.band_comm.size +
                                      band_rank * self.gd.comm.size)
                        self.world.receive(P_ni, world_rank, 1303 + a)
                    all_P_ni[nslice, i:i + ni] = P_ni
                    i += ni
                assert i == nproj
            return all_P_ni

        elif self.kpt_comm.rank == kpt_rank: # plain else works too...
            for a in range(natoms):
                if a in P_ani:
                    self.world.send(P_ani[a], 0, 1303 + a)

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        nk = len(self.ibzk_kc)
        mynu = len(self.kpt_u)
        kpt_rank, u = divmod(k + nk * s, mynu)
        band_rank, myn = self.bd.who_has(n)

        psit1_G = self._get_wave_function_array(u, myn)
        size = self.world.size
        rank = self.world.rank
        if size == 1:
            return psit1_G

        if self.kpt_comm.rank == kpt_rank:
            if self.band_comm.rank == band_rank:
                psit_G = self.gd.collect(psit1_G)

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

    def _get_wave_function_array(self, u, n):
        raise NotImplementedError


from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.utilities.blas import gemm, gemmdot
if extra_parameters.get('blacs'):
    from gpaw.lcao.overlap import BlacsTwoCenterIntegrals as TwoCenterIntegrals

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
        eigensolver.initialize(self.kpt_comm, self.gd, self.band_comm, self.dtype, 
                               self.setups.nao, self.mynbands, self.world)

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)        
        self.basis_functions.set_positions(spos_ac)

        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.mynbands
        
        if self.S_qMM is None: # XXX
            # First time:
            if extra_parameters.get('blacs'):
                self.basis_functions.set_matrix_distribution(self.band_comm)
                Mstop = self.basis_functions.Mstop
                Mstart = self.basis_functions.Mstart
                mynao = Mstop - Mstart
                self.tci.set_matrix_distribution(self.band_comm, Mstart, Mstop)
            else:
                mynao = nao
                
            self.S_qMM = np.empty((nq, mynao, nao), self.dtype)
            self.T_qMM = np.empty((nq, mynao, nao), self.dtype)
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
         
        if debug:
            from numpy.linalg import eigvalsh
            for S_MM in self.S_qMM:
                smin = eigvalsh(S_MM).real.min()
                if smin < 0:
                    raise RuntimeError('Overlap matrix has negative '
                                       'eigenvalue: %e' % smin)
        self.positions_set = True

    def initialize(self, density, hamiltonian, spos_ac):
        if density.nt_sG is None:
            if self.kpt_u[0].f_n is None or self.kpt_u[0].C_nM is None:
                density.initialize_from_atomic_densities(self.basis_functions)
            else:
                # We have the info we need for a density matrix, so initialize
                # from that instead of from scratch.  This will be the case
                # after set_positions() during a relaxation
                density.initialize_from_wavefunctions(self)
        else:
            # After a restart, nt_sg doesn't exist yet, so we'll have to
            # make sure it does.  Of course, this should have been taken care
            # of already by this time, so we should improve the code elsewhere
            density.calculate_normalized_charges_and_mix()
        hamiltonian.update(density)
           
    def calculate_density_matrix(self, f_n, C_nM, rho_MM):
        # ATLAS can't handle uninitialized output array:
        rho_MM.fill(42)

        if 1:
            # XXX Should not conjugate, but call gemm(..., 'c')
            # Although that requires knowing C_Mn and not C_nM.
            # that also conforms better to the usual conventions in literature
            Cf_Mn = C_nM.T.conj() * f_n
            gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
        else:
            # Alternative suggestion. Might be faster. Someone should test this
            C_Mn = C_nM.T.copy()
            r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
            tri2full(rho_MM)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        # Used in calculation of response potential in GLLB-potential
        if kpt.rho_MM is not None:
            rho_MM = kpt.rho_MM
        else:
            # XXX do we really want to allocate this array each time?
            nao = self.setups.nao
            rho_MM = np.empty((nao, nao), self.dtype)
            self.calculate_density_matrix(f_n, kpt.C_nM, rho_MM)
        self.timer.start('LCAO WaveFunctions: construct density')
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.q)
        self.timer.stop('LCAO WaveFunctions: construct density')

    def add_to_density_from_k_point(self, nt_sG, kpt):
        """Add contribution to pseudo electron-density. """
        self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        raise NotImplementedError('Kinetic density calculation for LCAO '
                                  'wavefunctions is not implemented.')

    def calculate_forces(self, hamiltonian, F_av):
        self.timer.start('LCAO forces')
        spos_ac = self.tci.atoms.get_scaled_positions()
        nao = self.setups.nao
        nq = len(self.ibzk_qc)
        dtype = self.dtype
        dThetadR_qvMM = np.empty((nq, 3, nao, nao), dtype)
        dTdR_qvMM = np.empty((nq, 3, nao, nao), dtype)
        dPdR_aqvMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
        self.timer.start('LCAO forces: tci derivative')
        self.tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM,
                                      dPdR_aqvMi)
        self.timer.stop('LCAO forces: tci derivative')
        
        # TODO: Most contributions will be the same for each spin.
        
        for kpt in self.kpt_u:
            self.calculate_forces_by_kpoint(kpt, hamiltonian,
                                            F_av, self.tci,
                                            self.S_qMM[kpt.q],
                                            self.T_qMM[kpt.q],
                                            self.P_aqMi,
                                            dThetadR_qvMM[kpt.q],
                                            dTdR_qvMM[kpt.q],
                                            dPdR_aqvMi)
        self.timer.stop('LCAO forces')

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
        
        def _slices(indices):
            for a in indices:
                M1 = basis_functions.M_a[a]
                M2 = M1 + self.setups[a].niAO
                yield a, M1, M2
        
        def slices():
            return _slices(atom_indices)
        
        def my_slices():
            return _slices(my_atom_indices)
        
        # TODO: in gamma point calculations, Hamiltonian has the matrix already
        # But not for both spins, if spinpolarized
        # Hmmm....
        self.timer.start('LCAO forces: initial')
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
        self.timer.stop('LCAO forces: initial')
        
        # Kinetic energy contribution
        #
        #           ----- d T
        #  a         \       mu nu
        # F += 2 Re   )   -------- rho
        #            /      d R       nu mu
        #           -----      a
        #        mu in a; nu
        #
        Fkin_av = np.zeros_like(F_av)
        dEdTrhoT_vMM = (dTdR_vMM * rhoT_MM[np.newaxis]).real
        for a, M1, M2 in my_slices():
            Fkin_av[a, :] = 2 * dEdTrhoT_vMM[:, M1:M2].sum(-1).sum(-1)
        del dEdTrhoT_vMM
        
        # Potential contribution
        #
        #          -----     /  d Phi  (r)
        #  a        \       |        mu    ~
        # F += 2 Re  )      |   ---------- v (r)  Phi  (r) dr rho
        #           /       |       R                nu          nu mu
        #          -----   /         a
        #      mu in a; nu
        #
        self.timer.start('LCAO forces: potential')
        Fpot_av = np.zeros_like(F_av)
        vt_G = hamiltonian.vt_sG[kpt.s]
        DVt_vMM = np.zeros((3, nao, nao), dtype)
        basis_functions.calculate_potential_matrix_derivative(vt_G, DVt_vMM, q)
        for a, M1, M2 in slices():
            for v in range(3):
                Fpot_av[a, v] = 2 * (DVt_vMM[v, M1:M2, :]
                                     * rhoT_MM[M1:M2, :]).real.sum()
        del DVt_vMM
        self.timer.stop('LCAO forces: potential')
        
        # Density matrix contribution due to basis overlap
        #
        #            ----- d Theta
        #  a          \           mu nu
        # F  += -2 Re  )   ------------  E
        #             /        d R        nu mu
        #            -----        a
        #         mu in a; nu
        #
        Frho_av = np.zeros_like(F_av)
        dThetadRE_vMM = (dThetadR_vMM * ET_MM[np.newaxis]).real
        for a, M1, M2 in my_slices():
            Frho_av[a, :] = -2 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
        del dThetadRE_vMM

        # Density matrix contribution from PAW correction
        #
        #           -----                        -----
        #  a         \      a                     \     b
        # F += -2 Re  )    Z      E        + 2 Re  )   Z      E
        #            /      mu nu  nu mu          /     mu nu  nu mu
        #           -----                        -----
        #           mu nu                    b; mu in a; nu
        #
        # with
        #                  b*
        #         -----  dP
        #   b      \       i mu    b   b
        #  Z     =  )   -------  dS   P
        #   mu nu  /      dR       ij  j nu
        #         -----     a
        #           ij
        #
        self.timer.start('LCAO forces: paw correction')
        dPdR_avMi = dict([(a, dPdR_aqvMi[a][q]) for a in my_atom_indices])
        work_MM = np.zeros((nao, nao), dtype)
        ZE_MM = None
        for b in my_atom_indices:
            setup = self.setups[b]
            O_ii = np.asarray(setup.O_ii, dtype)
            dOP_iM = np.zeros((setup.ni, nao), dtype)
            gemm(1.0, self.P_aqMi[b][q], O_ii, 0.0, dOP_iM, 'c')
            for v in range(3):
                gemm(1.0, dOP_iM, dPdR_avMi[b][v], 0.0, work_MM, 'n')
                ZE_MM = (work_MM * ET_MM).real
                for a, M1, M2 in slices():
                    dE = 2 * ZE_MM[M1:M2].sum()
                    Frho_av[a, v] += dE # the "b; mu in a; nu" term
                    Frho_av[b, v] -= dE # the "mu nu" term
        del work_MM, ZE_MM
        self.timer.stop('LCAO forces: paw correction')
        
        # Atomic density contribution
        #           -----                         -----
        #  a         \     a                       \     b
        # F  += 2 Re  )   A      rho       - 2 Re   )   A      rho
        #            /     mu nu    nu mu          /     mu nu    nu mu
        #           -----                         -----
        #           mu nu                     b; mu in a; nu
        #
        #                  b*
        #         ----- d P
        #  b       \       i mu   b   b
        # A     =   )   ------- dH   P
        #  mu nu   /      d R     ij  j nu
        #         -----      a
        #           ij
        #
        self.timer.start('LCAO forces: atomic density')
        Fatom_av = np.zeros_like(F_av)
        for b in my_atom_indices:
            H_ii = np.asarray(unpack(hamiltonian.dH_asp[b][kpt.s]), dtype)
            HP_iM = gemmdot(H_ii, np.conj(self.P_aqMi[b][q].T))
            for v in range(3):
                dPdR_Mi = dPdR_avMi[b][v]
                ArhoT_MM = (gemmdot(dPdR_Mi, HP_iM) * rhoT_MM).real
                for a, M1, M2 in slices():
                    dE = 2 * ArhoT_MM[M1:M2].sum()
                    Fatom_av[a, v] -= dE # the "b; mu in a; nu" term
                    Fatom_av[b, v] += dE # the "mu nu" term
        self.timer.stop('LCAO forces: atomic density')
        
        F_av += Fkin_av + Fpot_av + Frho_av + Fatom_av

    def _get_wave_function_array(self, u, n):
        kpt = self.kpt_u[u]
        C_nM = kpt.C_nM
        if C_nM is None:
            # Hack to make sure things are available after restart
            self.lazyloader.load(self)
        
        psit_G = self.gd.zeros(dtype=self.dtype)
        psit_1G = psit_G.reshape(1, -1)
        C_1M = kpt.C_nM[n].reshape(1, -1)
        q = kpt.q # Should we enforce q=-1 for gamma-point?
        if self.gamma:
            q = -1
        self.basis_functions.lcao_to_grid(C_1M, psit_1G, q)
        return psit_G

    def load_lazily(self, hamiltonian, spos_ac):
        """Horrible hack to recalculate lcao coefficients after restart."""
        class LazyLoader:
            def __init__(self, hamiltonian, spos_ac):
                self.hamiltonian = hamiltonian
                self.spos_ac = spos_ac
            
            def load(self, wfs):
                wfs.set_positions(spos_ac)
                wfs.eigensolver.iterate(hamiltonian, wfs)
                del wfs.lazyloader
        
        self.lazyloader = LazyLoader(hamiltonian, spos_ac)
        
    def estimate_memory(self, mem):
        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        ni_total = sum([setup.ni for setup in self.setups])
        itemsize = mem.itemsize[self.dtype]
        mem.subnode('C [qnM]', nq * self.mynbands * nao * itemsize)
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
        self.overlap = Overlap(self) # Object needed by memory estimate
        # (it has to be overwritten on each initialize() anyway, because of
        # weird object reuse issues, but we don't care)

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        if not self.gamma:
            self.pt.set_k_points(self.ibzk_qc)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def initialize(self, density, hamiltonian, spos_ac):
        self.overlap = Overlap(self)
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
            density.initialize_from_wavefunctions(self)
        elif density.nt_sG is None:
            density.initialize_from_atomic_densities(basis_functions)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xcfunc.gllb:
                hamiltonian.xcfunc.xc.initialize_from_atomic_orbitals(
                    basis_functions)
        else: # XXX???
            # We didn't even touch density, but some combinations in paw.set()
            # will make it necessary to do this for some reason.
            density.calculate_normalized_charges_and_mix()
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
        
        self.timer.start('Wavefunction: lcao initialization')
        if self.nbands <= self.setups.nao:
            lcaonbands = self.nbands
            lcaomynbands = self.mynbands
        else:
            lcaonbands = self.setups.nao
            lcaomynbands = self.setups.nao
            assert self.band_comm.size == 1

        lcaobd = BandDescriptor(lcaonbands, self.band_comm, self.bd.strided)
        assert lcaobd.mynbands == lcaomynbands #XXX

        lcaowfs = LCAOWaveFunctions(self.gd, self.nspins, self.setups, lcaobd,
                                    self.dtype, self.world, self.kpt_comm,
                                    self.gamma, self.bzk_kc, self.ibzk_kc,
                                    self.weight_k, self.symmetry)
        lcaowfs.basis_functions = basis_functions
        lcaowfs.timer = self.timer
        lcaowfs.set_positions(spos_ac)
        eigensolver = get_eigensolver('lcao', 'lcao')
        eigensolver.initialize(self.kpt_comm, self.gd, self.band_comm, self.dtype,
                               self.setups.nao, lcaomynbands, self.world)
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
        self.timer.stop('Wavefunction: lcao initialization')

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

        scale = np.sqrt(12 / abs(np.linalg.det(gd2.cell_cv)))

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

    #def add_to_density_from_k_point(self, nt_sG, kpt):
    #    self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)

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
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                ft_mn = ne * np.outer(c_n.conj(), c_n)
                for ft_n, psi_m in zip(ft_mn, kpt.psit_nG):
                    for ft, psi_n in zip(ft_n, kpt.psit_nG):
                        if abs(ft) > 1.e-12:
                            nt_G += (psi_m.conj() * ft * psi_n).real

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

    def _get_wave_function_array(self, u, n):
        psit_nG = self.kpt_u[u].psit_nG
        if psit_nG is None:
            raise RuntimeError('This calculator has no wave functions!')
        return psit_nG[n][:] # dereference possible tar-file content

    def estimate_memory(self, mem):
        # XXX Laplacian operator?
        gridbytes = self.gd.bytecount(self.dtype)
        mem.subnode('Arrays psit_nG', 
                    len(self.kpt_u) * self.mynbands * gridbytes)
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'), self.gd,
                                         self.dtype, self.mynbands,
                                         self.nbands)
        self.pt.estimate_memory(mem.subnode('Projectors'))
        self.overlap.estimate_memory(mem.subnode('Overlap op'), self.dtype)


