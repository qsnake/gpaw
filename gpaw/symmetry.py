# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np

from gpaw import debug

class Symmetry:
    
    def __init__(self, id_a, cell_cv, pbc_c=np.ones(3, bool), tolerance=1e-11):
        """Symmetry object.

        Two atoms can only be identical if they have the same atomic numbers,
        setup types and magnetic moments. If it is an LCAO type of calculation,
        they must have the same atomic basis set also.

        """

        self.id_a = id_a
        self.cell_cv = np.array(cell_cv, float)
        assert self.cell_cv.shape == (3, 3)
        self.pbc_c = np.array(pbc_c, bool)
        self.tol = tolerance

        self.op_scc = np.identity(3, int).reshape((1, 3, 3))
        
    def analyze(self, spos_ac):
        """Determine list of symmetry operations.

        First determine all symmetry operations of the cell. Then call
        ``prune_symmetries`` to remove those symmetries that are not satisfied
        by the atoms.

        """

        # Symmetry operations as matrices in 123 basis
        self.op_scc = [] 
        
        # Metric tensor
        metric_cc = np.dot(self.cell_cv, self.cell_cv.T)

        # Generate all possible 3x3 symmetry matrices using base-3 integers
        power = (6561, 2187, 729, 243, 81, 27, 9, 3, 1)

        # operation is a 3x3 matrix, with possible elements -1, 0, 1, thus
        # there are 3**9 = 19683 possible matrices
        for base3id in xrange(19683):
            op_cc = np.empty((3, 3), dtype=int)
            m = base3id
            for ip, p in enumerate(power):
                d, m = divmod(m, p)
                op_cc[ip // 3, ip % 3] = 1 - d

            # The metric of the cell should be conserved after applying
            # the operation
            opmetric_cc = np.dot(np.dot(op_cc, metric_cc), op_cc.T)
                                       
            if np.abs(metric_cc - opmetric_cc).sum() > self.tol:
                continue

            # Operation must not swap axes that are not both periodic
            pbc_cc = np.logical_and.outer(self.pbc_c, self.pbc_c)
            if op_cc[~(pbc_cc | np.identity(3, bool))].any():
                continue

            # Operation must not invert axes that are not periodic
            pbc_cc = np.logical_and.outer(self.pbc_c, self.pbc_c)
            if not (op_cc[np.diag(~self.pbc_c)] == 1).all():
                continue

            # operation is a valid symmetry of the unit cell
            self.op_scc.append(op_cc)

        self.op_scc = np.array(self.op_scc)
        
        # Check if symmetry operations are also valid when taking account
        # of atomic positions
        self.prune_symmetries(spos_ac)
        
    def prune_symmetries(self, spos_ac):
        """Remove symmetries that are not satisfied by the atoms."""

        # Build lists of (atom number, scaled position) tuples.  One
        # list for each combination of atomic number, setup type,
        # magnetic moment and basis set
        species = {}
        for a, id in enumerate(self.id_a):
            spos_c = spos_ac[a]
            if id in species:
                species[id].append((a, spos_c))
            else:
                species[id] = [(a, spos_c)]

        opok = []
        maps = []
        # Reduce point group using operation matrices
        for op_cc in self.op_scc:
            map = np.zeros(len(spos_ac), int)
            for specie in species.values():
                for a1, spos1_c in specie:
                    spos1_c = np.dot(spos1_c, op_cc)
                    ok = False
                    for a2, spos2_c in specie:
                        sdiff = spos1_c - spos2_c
                        sdiff -= np.floor(sdiff + 0.5)
                        if np.dot(sdiff, sdiff) < self.tol:
                            ok = True
                            map[a1] = a2
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                opok.append(op_cc)
                maps.append(map)

        if debug:
            for op_cc, map_a in zip(opok, maps):
                for a1, id1 in enumerate(self.id_a):
                    a2 = map_a[a1]
                    assert id1 == self.id_a[a2]
                    spos1_c = spos_ac[a1]
                    spos2_c = spos_ac[a2]
                    sdiff = np.dot(spos1_c, op_cc) - spos2_c
                    sdiff -= np.floor(sdiff + 0.5)
                    assert np.dot(sdiff, sdiff) < self.tol

        self.maps = maps
        self.op_scc = np.array(opok)

    def check(self, spos_ac):
        """Check if positions satisfy symmetry operations."""

        nsymold = len(self.op_scc)
        self.prune_symmetries(spos_ac)
        if len(self.op_scc) < nsymold:
            raise RuntimeError('Broken symmetry!')

    def reduce(self, bzk_kc):
        """Reduce k-points to irreducible part of the BZ.

        Returns the irreducible k-points and the weights.
        
        """
        
        op_scc = self.op_scc
        inv_cc = -np.identity(3, int)
        have_inversion_symmetry = False
        
        for op_cc in op_scc:
            if (op_cc == inv_cc).all():
                have_inversion_symmetry = True
                break

        # Use time-reversal symmetry when inversion symmetry is absent
        if not have_inversion_symmetry:
            op_scc = np.concatenate((op_scc, -op_scc))
            
        nbzkpts = len(bzk_kc)
        ibzk0_kc = np.empty((nbzkpts, 3))
        ibzk_kc = ibzk0_kc[:0]
        weight_k = np.ones(nbzkpts)
        nibzkpts = 0
        kbz = nbzkpts
        
        # Mapping between k and symmetry related point in the irreducible BZ
        kibz_k = np.empty(nbzkpts, int)
        # Symmetry operation mapping the k-point in the irreducible BZ to k
        sym_k = np.empty(nbzkpts, int)
        # Time-reversal symmetry used on top of the point group operation
        time_reversal_k = np.array(nbzkpts * [False])
        
        for k_c in bzk_kc[::-1]:
            kbz -= 1
            found = False
            
            for s, op_cc in enumerate(op_scc):
                if len(ibzk_kc) == 0:
                    break
                diff_kc = np.dot(ibzk_kc, op_cc.T) - k_c
                b_k = ((diff_kc - diff_kc.round())**2).sum(1) < self.tol
                if b_k.any():
                    found = True
                    kibz = np.where(b_k)[0][0]
                    weight_k[kibz] += 1.0
                    kibz_k[kbz] = kibz
                    sym_k[kbz] = s
                    # Time-reversal symmetry combined with point group symmetry
                    if s >= len(self.op_scc):
                        sym_k[kbz] = s - len(self.op_scc)
                        time_reversal_k[kbz] = True
                    break
            if not found:
                kibz_k[kbz] = nibzkpts
                sym_k[kbz] = 0
                nibzkpts += 1
                ibzk_kc = ibzk0_kc[:nibzkpts]
                ibzk_kc[-1] = k_c

        self.sym_k = sym_k
        self.time_reversal_k = time_reversal_k
        # Reverse order (looks more natural)
        self.kibz_k = nibzkpts - 1 - kibz_k
        
        return ibzk_kc[::-1].copy(), weight_k[:nibzkpts][::-1] / nbzkpts

    def prune_symmetries_grid(self, N_c):
        """Remove symmetries that are not satisfied by the grid."""

        U_scc = []
        a_sa = []
        for U_cc, a_a in zip(self.op_scc, self.maps):
            if not (U_cc * N_c - (U_cc.T * N_c).T).any():
                U_scc.append(U_cc)
                a_sa.append(a_a)
                
        self.maps = np.array(a_sa)
        self.op_scc = np.array(U_scc)

    def symmetrize(self, a, gd):
        """Symmetrize array."""
        
        gd.symmetrize(a, self.op_scc)

    def symmetrize_wavefunction(self, a_g, kibz_c, kbz_c, op_cc, time_reversal):
        """Generate Bloch function from symmetry related function in the IBZ.

        a_g: ndarray
            Array with Bloch function from the irreducible BZ.
        kibz_c: ndarray
            Corresponing k-point coordinates.
        kbz_c: ndarray
            K-point coordinates of the symmetry related k-point.
        op_cc: ndarray
            Point group operation connecting the two k-points.
        time-reversal: bool
            Time-reversal symmetry required in addition to the point group
            symmetry to connect the two k-points.
        
        """

        # Identity
        if (np.abs(op_cc - np.eye(3, dtype=int)) < 1e-10).all():
            if time_reversal:
                return a_g.conj()
            else:
                return a_g
        # Inversion symmetry
        elif (np.abs(op_cc + np.eye(3, dtype=int)) < 1e-10).all():
            return a_g.conj()
        # General point group symmetry
        else:
            import _gpaw
            b_g = np.zeros_like(a_g)
            if time_reversal:
                _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc, kibz_c, -kbz_c)
                return b_g.conj()
            else:
                _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc, kibz_c, kbz_c)
                return b_g
        
    def symmetrize_forces(self, F0_av):
        """Symmetrice forces."""
        
        F_ac = np.zeros_like(F0_av)
        for map_a, op_cc in zip(self.maps, self.op_scc):
            op_vv = np.dot(np.linalg.inv(self.cell_cv),
                           np.dot(op_cc, self.cell_cv))
            for a1, a2 in enumerate(map_a):
                F_ac[a2] += np.dot(F0_av[a1], op_vv)
        return F_ac / len(self.op_scc)
        
    def print_symmetries(self, text):
        
        n = len(self.op_scc)
        text('Symmetries present: %s' % n)
