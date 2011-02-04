import numpy as np

from gpaw.utilities.blas import gemm
from gpaw.utilities import pack, unpack2
from gpaw.utilities.timing import nulltimer


class EmptyWaveFunctions:
    def __nonzero__(self):
        return False
    
    def set_eigensolver(self, eigensolver):
        pass

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
    def __init__(self, gd, nvalence, setups, bd, dtype,
                 world, kd, timer=None):
        if timer is None:
            timer = nulltimer
            
        self.gd = gd
        self.nspins = kd.nspins
        self.nvalence = nvalence
        self.bd = bd
        self.nbands = self.bd.nbands #XXX
        self.mynbands = self.bd.mynbands #XXX
        self.dtype = dtype
        self.world = world
        self.kd = kd
        self.band_comm = self.bd.comm #XXX
        self.timer = timer
        self.rank_a = None

        # XXX Remember to modify aseinterface when removing the following
        # attributes from the wfs object
        self.gamma = kd.gamma
        self.kpt_comm = kd.comm
        self.bzk_kc = kd.bzk_kc
        self.ibzk_kc = kd.ibzk_kc
        self.ibzk_qc = kd.ibzk_qc
        self.weight_k = kd.weight_k
        self.symmetry = kd.symmetry
        self.nibzkpts = kd.nibzkpts
            
        self.kpt_u = kd.create_k_points(self.gd)

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
        
        self.timer.start('Symmetrize density')
        for nt_G in nt_sG:
            self.symmetry.symmetrize(nt_G, self.gd)
        self.timer.stop('Symmetrize density')

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
                d_nn = ne * np.outer(c_n.conj(), c_n)
                D_sii[kpt.s] += np.dot(P_ni.T.conj(), np.dot(d_nn, P_ni)).real
    
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
        if len(self.symmetry.op_scc) > 1:
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
                                                    self.symmetry.a_sa))

    def set_positions(self, spos_ac):
        self.positions_set = False
        rank_a = self.gd.get_ranks_from_positions(spos_ac)

        """
        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        if (self.rank_a is not None and rank_a is not None and
            self.kpt_u[0].P_ani is None and (rank_a == self.gd.comm.rank).any()):
            for kpt in self.kpt_u:
                kpt.P_ani = {}
        """

        if self.rank_a is not None and self.kpt_u[0].P_ani is not None:
            self.timer.start('Redistribute')
            requests = []
            mynks = len(self.kpt_u)
            flags = (self.rank_a != rank_a)
            my_incoming_atom_indices = np.argwhere(np.bitwise_and(flags, \
                rank_a == self.gd.comm.rank)).ravel()
            my_outgoing_atom_indices = np.argwhere(np.bitwise_and(flags, \
                self.rank_a == self.gd.comm.rank)).ravel()

            for a in my_incoming_atom_indices:
                # Get matrix from old domain:
                ni = self.setups[a].ni
                P_uni = np.empty((mynks, self.mynbands, ni), self.dtype)
                requests.append(self.gd.comm.receive(P_uni, self.rank_a[a],
                                                     tag=a, block=False))
                for myu, kpt in enumerate(self.kpt_u):
                    assert a not in kpt.P_ani
                    kpt.P_ani[a] = P_uni[myu]

            for a in my_outgoing_atom_indices:
                # Send matrix to new domain:
                P_uni = np.array([kpt.P_ani.pop(a) for kpt in self.kpt_u])
                requests.append(self.gd.comm.send(P_uni, rank_a[a],
                                                  tag=a, block=False))
            self.gd.comm.waitall(requests)
            self.timer.stop('Redistribute')

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

    def collect_array(self, name, k, s, subset=None):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = self.kd.get_rank_and_index(s, k)

        if self.kpt_comm.rank == kpt_rank:
            a_nx = getattr(kpt_u[u], name)

            if subset is not None:
                a_nx = a_nx[subset]

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if self.band_comm.size == 1:
                    if kpt_rank == 0:
                        return a_nx
                    else:
                        self.kpt_comm.ssend(a_nx, 0, 1301)
                else:
                    b_nx = self.bd.collect(a_nx)
                    if self.band_comm.rank == 0:
                        if kpt_rank == 0:
                            return b_nx
                        else:
                            self.kpt_comm.ssend(b_nx, 0, 1301)

        elif self.world.rank == 0 and kpt_rank != 0:
            # Find shape and dtype:
            a_nx = getattr(kpt_u[0], name)
            shape = (self.nbands,) + a_nx.shape[1:]
            dtype = a_nx.dtype
            b_nx = np.zeros(shape, dtype=dtype)
            self.kpt_comm.receive(b_nx, kpt_rank, 1301)
            return b_nx

    def collect_auxiliary(self, value, k, s, shape=1, dtype=float):
        """Helper method for collecting band-independent scalars/arrays.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = self.kd.get_rank_and_index(s, k)

        if self.kpt_comm.rank == kpt_rank:
            if isinstance(value, str):
                a_o = getattr(kpt_u[u], value)
            else:
                a_o = value[u] # assumed list

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

        kpt_rank, u = self.kd.get_rank_and_index(s, k)

        natoms = len(self.rank_a) # it's a hack...
        nproj = sum([setup.ni for setup in self.setups])

        if self.world.rank == 0:
            if kpt_rank == 0:
                P_ani = self.kpt_u[u].P_ani
            mynu = len(self.kpt_u)
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
            P_ani = self.kpt_u[u].P_ani
            for a in range(natoms):
                if a in P_ani:
                    self.world.ssend(P_ani[a], 0, 1303 + a)

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = self.kd.get_rank_and_index(s, k)
        band_rank, myn = self.bd.who_has(n)

        size = self.world.size
        rank = self.world.rank

        if self.kpt_comm.rank == kpt_rank:
            psit1_G = self._get_wave_function_array(u, myn)
            if size == 1:
                return psit1_G
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
