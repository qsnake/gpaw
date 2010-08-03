import numpy as np

from gpaw.utilities.blas import gemm
from gpaw.utilities import pack, unpack2
from gpaw.kpoint import KPoint
from gpaw.utilities.timing import nulltimer


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
    def __init__(self, gd, nspins, nvalence, setups, bd, dtype,
                 world, kpt_comm,
                 gamma, bzk_kc, ibzk_kc, weight_k, symmetry, timer=None):
        if timer is None:
            timer = nulltimer
            
        self.gd = gd
        self.nspins = nspins
        self.nvalence = nvalence
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
        self.kpt_u = []
        sdisp_cd = gd.sdisp_cd
        for ks in range(ks0, ks0 + mynks):
            s, k = divmod(ks, self.nibzkpts)
            q = (ks - ks0) % self.nibzkpts
            weight = weight_k[k] * 2 / nspins
            if gamma:
                phase_cd = np.ones((3, 2), complex)
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * ibzk_kc[k, :, np.newaxis])
            self.kpt_u.append(KPoint(weight, s, k, q, phase_cd))

        if nspins == 2 and kpt_comm.size == 1:
            # Avoid duplicating k-points in local list of k-points.
            self.ibzk_qc = ibzk_kc.copy()
        else:
            self.ibzk_qc = np.vstack((ibzk_kc, ibzk_kc))[ks0:ks0 + mynks]
        
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

    def collect_array(self, name, k, s, subset=None):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = divmod(k + self.nibzkpts * s, len(kpt_u))

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
        kpt_rank, u = divmod(k + self.nibzkpts * s, len(kpt_u))

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
                    self.world.ssend(P_ani[a], 0, 1303 + a)

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
