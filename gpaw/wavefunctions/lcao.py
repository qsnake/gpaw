import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
from gpaw import debug
from gpaw.lcao.overlap import NewTwoCenterIntegrals as NewTCI
from gpaw.utilities.blas import gemm, gemmdot
from gpaw.wavefunctions.base import WaveFunctions


class LCAOWaveFunctions(WaveFunctions):
    def __init__(self, ksl, gd, nspins, nvalence, setups, bd,
                 dtype, world, kpt_comm,
                 gamma, bzk_kc, ibzk_kc, weight_k, symmetry, timer=None):
        WaveFunctions.__init__(self, gd, nspins, nvalence, setups, bd,
                               dtype, world, kpt_comm,
                               gamma, bzk_kc, ibzk_kc, weight_k, symmetry,
                               timer)
        self.ksl = ksl
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None
        
        self.timer.start('TCI: Evaluate splines')
        self.tci = NewTCI(gd.cell_cv, gd.pbc_c, setups, self.ibzk_qc, gamma)
        self.timer.stop('TCI: Evaluate splines')
        
        self.basis_functions = BasisFunctions(gd,
                                              [setup.phit_j
                                               for setup in setups],
                                              kpt_comm,
                                              cut=True)
        if not gamma:
            self.basis_functions.set_k_points(self.ibzk_qc)

    def summary(self, fd):
        fd.write('Mode: LCAO\n')
        
    def set_eigensolver(self, eigensolver):
        WaveFunctions.set_eigensolver(self, eigensolver)
        eigensolver.initialize(self.gd, self.dtype, self.setups.nao, self.ksl)

    def set_positions(self, spos_ac):
        self.timer.start('Basic WFS set positions')
        WaveFunctions.set_positions(self, spos_ac)
        self.timer.stop('Basic WFS set positions')
        self.timer.start('Basis functions set positions')
        self.basis_functions.set_positions(spos_ac)
        self.timer.stop('Basis functions set positions')
        if self.ksl is not None:
            self.basis_functions.set_matrix_distribution(self.ksl.Mstart,
                                                         self.ksl.Mstop)

        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.mynbands
        
        Mstop = self.ksl.Mstop
        Mstart = self.ksl.Mstart
        mynao = Mstop - Mstart

        if self.ksl.using_blacs: # XXX
            # S and T have been distributed to a layout with blacs, so
            # discard them to force reallocation from scratch.
            #
            # TODO: evaluate S and T when they *are* distributed, thus saving
            # memory and avoiding this problem
            self.S_qMM = None
            self.T_qMM = None
        
        S_qMM = self.S_qMM
        T_qMM = self.T_qMM
        
        if S_qMM is None: # XXX
            # First time:
            assert T_qMM is None
            if self.ksl.using_blacs: # XXX
                self.tci.set_matrix_distribution(Mstart, mynao)
                
            S_qMM = np.empty((nq, mynao, nao), self.dtype)
            T_qMM = np.empty((nq, mynao, nao), self.dtype)
        
        for kpt in self.kpt_u:
            if kpt.C_nM is None:
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

        self.timer.start('TCI: Calculate S, T, P')
        self.tci.calculate(spos_ac, S_qMM, T_qMM, self.P_aqMi)
        nao = self.setups.nao
        for a, P_qMi in self.P_aqMi.items():
            dO_ii = np.asarray(self.setups[a].dO_ii, P_qMi.dtype)
            for S_MM, P_Mi in zip(S_qMM, P_qMi):
                dOP_iM = np.zeros((dO_ii.shape[1], nao), P_Mi.dtype)
                # (ATLAS can't handle uninitialized output array)
                gemm(1.0, P_Mi, dO_ii, 0.0, dOP_iM, 'c')
                gemm(1.0, dOP_iM, P_Mi[Mstart:Mstop], 1.0, S_MM, 'n')

        self.timer.stop('TCI: Calculate S, T, P')

        S_MM = None # allow garbage collection of old S_qMM after redist
        S_qMM = self.ksl.distribute_overlap_matrix(S_qMM)
        T_qMM = self.ksl.distribute_overlap_matrix(T_qMM)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.S_MM = S_qMM[q]
            kpt.T_MM = T_qMM[q]


        if debug and self.band_comm.size == 1 and self.gd.comm.rank == 0:
            # S and T are summed only on comm master, so check only there
            from numpy.linalg import eigvalsh
            self.timer.start('Check positive definiteness')
            for S_MM in S_qMM:
                tri2full(S_MM, UL='U')
                smin = eigvalsh(S_MM).real.min()
                if smin < 0:
                    raise RuntimeError('Overlap matrix has negative '
                                       'eigenvalue: %e' % smin)
            self.timer.stop('Check positive definiteness')
        self.positions_set = True
        self.S_qMM = S_qMM
        self.T_qMM = T_qMM

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
           
    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        # ATLAS can't handle uninitialized output array:
        #rho_MM.fill(42)

        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix(f_n, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

        # ----------------------------
        if 1:
            # XXX Should not conjugate, but call gemm(..., 'c')
            # Although that requires knowing C_Mn and not C_nM.
            # that also conforms better to the usual conventions in literature
            Cf_Mn = C_nM.T.conj() * f_n
            gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
            self.bd.comm.sum(rho_MM)
        else:
            # Alternative suggestion. Might be faster. Someone should test this
            C_Mn = C_nM.T.copy()
            r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
            tri2full(rho_MM)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        # Custom occupations are used in calculation of response potential
        # with GLLB-potential
        Mstart = self.basis_functions.Mstart
        Mstop = self.basis_functions.Mstop
        if kpt.rho_MM is None:
            rho_MM = self.calculate_density_matrix(f_n, kpt.C_nM)
        else:
            rho_MM = kpt.rho_MM
        self.timer.start('Construct density')
        self.basis_functions.construct_density(rho_MM,
                                               nt_sG[kpt.s], kpt.q)
        self.timer.stop('Construct density')

    def add_to_density_from_k_point(self, nt_sG, kpt):
        """Add contribution to pseudo electron-density. """
        self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        raise NotImplementedError('Kinetic density calculation for LCAO '
                                  'wavefunctions is not implemented.')

    def calculate_forces(self, hamiltonian, F_av):
        self.timer.start('LCAO forces')
        spos_ac = self.tci.atoms.get_scaled_positions() % 1.0
        nao = self.ksl.nao
        mynao = self.ksl.mynao
        nq = len(self.ibzk_qc)
        dtype = self.dtype
        dThetadR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
        dTdR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
        dPdR_aqvMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
        self.timer.start('LCAO forces: tci derivative')
        self.tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM,
                                      dPdR_aqvMi)
        #if not hasattr(self.tci, 'set_positions'): # XXX newtci
        comm = self.gd.comm
        comm.sum(dThetadR_qvMM)
        comm.sum(dTdR_qvMM)
        self.timer.stop('LCAO forces: tci derivative')
        
        # TODO: Most contributions will be the same for each spin.
        
        for kpt in self.kpt_u:
            self.calculate_forces_by_kpoint(kpt, hamiltonian,
                                            F_av, self.tci,
                                            self.P_aqMi,
                                            dThetadR_qvMM[kpt.q],
                                            dTdR_qvMM[kpt.q],
                                            dPdR_aqvMi)
        self.ksl.orbital_comm.sum(F_av)
        if self.bd.comm.rank == 0:
            self.kpt_comm.sum(F_av, 0)
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
                                   F_av, tci, P_aqMi,
                                   dThetadR_vMM, dTdR_vMM, dPdR_aqvMi):
        k = kpt.k
        q = kpt.q
        mynao = self.ksl.mynao
        nao = self.ksl.nao
        dtype = self.dtype

        Mstart = self.ksl.Mstart
        Mstop = self.ksl.Mstop
        
        basis_functions = self.basis_functions
        my_atom_indices = basis_functions.my_atom_indices
        atom_indices = basis_functions.atom_indices        
        
        def _slices(indices):
            for a in indices:
                M1 = basis_functions.M_a[a] - Mstart
                M2 = M1 + self.setups[a].niAO
                yield a, M1, M2
        
        def slices():
            return _slices(atom_indices)
        
        def my_slices():
            return _slices(my_atom_indices)
        
        #
        #         -----                    -----
        #          \    -1                  \    *
        # E      =  )  S     H    rho     =  )  c     eps  f  c
        #  mu nu   /    mu x  x z    z nu   /    n mu    n  n  n nu
        #         -----                    -----
        #          x z                       n
        #
        # We use the transpose of that matrix.  The first form is used
        # if rho is given, otherwise the coefficients are used.
        self.timer.start('LCAO forces: initial')
        if kpt.rho_MM is None:
            rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n, kpt.C_nM)
            ET_MM = self.ksl.get_transposed_density_matrix(kpt.f_n * kpt.eps_n,
                                                           kpt.C_nM)
        else:
            H_MM = self.eigensolver.calculate_hamiltonian_matrix(hamiltonian,
                                                                 self,
                                                                 kpt)
            tri2full(H_MM)
            S_MM = self.S_qMM[q].copy()
            tri2full(S_MM)
            ET_MM = np.linalg.solve(S_MM, gemmdot(H_MM, kpt.rho_MM)).T.copy()
            del S_MM, H_MM
            rhoT_MM = kpt.rho_MM.T.copy()
        self.timer.stop('LCAO forces: initial')

        
        # Kinetic energy contribution
        #
        #           ----- d T
        #  a         \       mu nu
        # F += 2 Re   )   -------- rho
        #            /    d R         nu mu
        #           -----    mu nu
        #        mu in a; nu
        #
        Fkin_av = np.zeros_like(F_av)
        dEdTrhoT_vMM = (dTdR_vMM * rhoT_MM[np.newaxis]).real
        for a, M1, M2 in my_slices():
            Fkin_av[a, :] = 2 * dEdTrhoT_vMM[:, M1:M2].sum(-1).sum(-1)
        del dEdTrhoT_vMM
        
        # Potential contribution
        #
        #           -----      /  d Phi  (r)
        #  a         \        |        mu    ~
        # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
        #            /        |     d R                nu          nu mu
        #           -----    /         a
        #        mu in a; nu
        #
        self.timer.start('LCAO forces: potential')
        Fpot_av = np.zeros_like(F_av)
        vt_G = hamiltonian.vt_sG[kpt.s]
        DVt_vMM = np.zeros((3, mynao, nao), dtype)
        # Note that DVt_vMM contains dPhi(r) / dr = - dPhi(r) / dR^a
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
        #            -----        mu nu
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
        # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
        #            /      mu nu  nu mu          /     mu nu  nu mu
        #           -----                        -----
        #           mu nu                    b; mu in a; nu
        #
        # with
        #                  b*
        #         -----  dP
        #   b      \       i mu    b   b
        #  Z     =  )   -------- dS   P
        #   mu nu  /     dR        ij  j nu
        #         -----    b mu
        #           ij
        #
        self.timer.start('LCAO forces: paw correction')
        dPdR_avMi = dict([(a, dPdR_aqvMi[a][q]) for a in my_atom_indices])
        work_MM = np.zeros((mynao, nao), dtype)
        ZE_MM = None
        for b in my_atom_indices:
            setup = self.setups[b]
            dO_ii = np.asarray(setup.dO_ii, dtype)
            dOP_iM = np.zeros((setup.ni, nao), dtype)
            gemm(1.0, self.P_aqMi[b][q], dO_ii, 0.0, dOP_iM, 'c')
            for v in range(3):
                gemm(1.0, dOP_iM, dPdR_avMi[b][v][Mstart:Mstop], 0.0,
                     work_MM, 'n')
                ZE_MM = (work_MM * ET_MM).real
                for a, M1, M2 in slices():
                    dE = 2 * ZE_MM[M1:M2].sum()
                    Frho_av[a, v] -= dE # the "b; mu in a; nu" term
                    Frho_av[b, v] += dE # the "mu nu" term
        del work_MM, ZE_MM
        self.timer.stop('LCAO forces: paw correction')
        
        # Atomic density contribution
        #            -----                         -----
        #  a          \     a                       \     b
        # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
        #             /     mu nu    nu mu          /     mu nu    nu mu
        #            -----                         -----
        #            mu nu                     b; mu in a; nu
        #
        #                  b*
        #         ----- d P
        #  b       \       i mu   b   b
        # A     =   )   ------- dH   P
        #  mu nu   /    d R       ij  j nu
        #         -----    b mu
        #           ij
        #
        self.timer.start('LCAO forces: atomic density')
        Fatom_av = np.zeros_like(F_av)
        for b in my_atom_indices:
            H_ii = np.asarray(unpack(hamiltonian.dH_asp[b][kpt.s]), dtype)
            HP_iM = gemmdot(H_ii, np.conj(self.P_aqMi[b][q].T))
            for v in range(3):
                dPdR_Mi = dPdR_avMi[b][v][Mstart:Mstop]
                ArhoT_MM = (gemmdot(dPdR_Mi, HP_iM) * rhoT_MM).real
                for a, M1, M2 in slices():
                    dE = 2 * ArhoT_MM[M1:M2].sum()
                    Fatom_av[a, v] += dE # the "b; mu in a; nu" term
                    Fatom_av[b, v] -= dE # the "mu nu" term
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
                wfs.set_positions(self.spos_ac)
                wfs.eigensolver.iterate(hamiltonian, wfs)
                del wfs.lazyloader
        
        self.lazyloader = LazyLoader(hamiltonian, spos_ac)
        
    def write_wave_functions(self, writer):
        if self.world.rank == 0:
            writer.dimension('nbasis', self.setups.nao)
            writer.add('WaveFunctionCoefficients',
                       ('nspins', 'nibzkpts', 'nbands', 'nbasis'),
                       dtype=self.dtype)

        for s in range(self.nspins):
            for k in range(self.nibzkpts):
                C_nM = self.collect_array('C_nM', k, s)
                if self.world.rank == 0:
                    writer.fill(C_nM, s, k)

    def read_coefficients(self, reader):
        for kpt in self.kpt_u:
            kpt.C_nM = self.bd.empty(self.setups.nao, dtype=self.dtype)
            for n in self.bd.get_band_indices():
                kpt.C_nM[n] = reader.get('WaveFunctionCoefficients',
                                         kpt.s, kpt.k, n)

    def estimate_memory(self, mem):
        nq = len(self.ibzk_qc)
        nao = self.setups.nao
        ni_total = sum([setup.ni for setup in self.setups])
        itemsize = mem.itemsize[self.dtype]
        mem.subnode('C [qnM]', nq * self.mynbands * nao * itemsize)
        nM1, nM2 = self.ksl.get_overlap_matrix_shape()
        mem.subnode('S, T [2 x qmm]', 2 * nq * nM1 * nM2 * itemsize)
        mem.subnode('P [aqMi]', nq * nao * ni_total // self.gd.comm.size)
        self.tci.estimate_memory(mem.subnode('TCI'))
        self.basis_functions.estimate_memory(mem.subnode('BasisFunctions'))
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'),
                                         self.dtype)
