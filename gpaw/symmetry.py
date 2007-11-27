# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gpaw import debug


class Symmetry:
    def __init__(self, Z_a, type_a, magmom_a, basis_a,
                 domain, tolerance=1e-9):
        """Symmetry object.

        Two atoms can only be identical if they have the same atomic
        numbers, setup types and magnetic moments.  If it is an LCAO
        type of calculation, they must have the same atomic basis
        set also."""
        
        self.ZTMB_a = zip(Z_a, type_a, magmom_a, basis_a)
        self.cell_c = domain.cell_c
        self.periodic_c = domain.periodic_c
        self.scale_position = domain.scale_position  # XXX ref to domain!
        self.tol = tolerance
        # The identity:
        self.symmetries = [((0, 1, 2), num.array((1, 1, 1)))]

    def analyze(self, pos_ac):
        """Analyse(atoms)

        Find a list of symmetry operations."""

                    
        # There are six orderings of the axes:
        allpossibleswaps = [(0, 1, 2), (0, 2, 1),
                            (1, 0, 2), (1, 2, 0),
                            (2, 0, 1), (2, 1, 0)]
        # Only swap axes of equal length:
        cellsyms = [[abs(self.cell_c[c1] - self.cell_c[c2]) < self.tol and
                     self.periodic_c[c1] and self.periodic_c[c2]
                     for c1 in range(3)]
                    for c2 in range(3)]
        swaps = []
        for swap in allpossibleswaps:
            ok = True
            for c1, c2 in enumerate(swap):
                if c1 == c2 or cellsyms[c1][c2]:
                    continue
                else:
                    ok = False
                    break
            if ok:
                swaps.append(swap)

        mirrors = [[1.0], [1.0], [1.0]]
        for c in range(3):
            if self.periodic_c[c]:
                mirrors[c].append(-1.0)
        mirrors = [num.array((m0, m1, m2))
                   for m0 in mirrors[0]
                   for m1 in mirrors[1]
                   for m2 in mirrors[2]]
                            
        self.symmetries = [(swap, mirror)
                           for swap in swaps for mirror in mirrors]
        
        self.prune_symmetries(pos_ac)

    def prune_symmetries(self, pos_ac):
        """prune_symmetries(atoms)

        Remove symmetries that are not satisfied."""

        # Build lists of (atom number, scaled position) tuples.  One
        # list for each combination of atomic number, setup type,
        # magnetic moment and basis set:
        species = {}
        for a, ZTMB in enumerate(self.ZTMB_a):
            spos_c = self.scale_position(pos_ac[a])
            if species.has_key(ZTMB):
                species[ZTMB].append((a, spos_c))
            else:
                species[ZTMB] = [(a, spos_c)]

        symmok = []
        maps = []
        for swap, mirror in self.symmetries:
            map = num.zeros(len(pos_ac))
            for specie in species.values():
                for a1, spos1_c in specie:
                    spos1_c = num.take(spos1_c * mirror, swap)
                    ok = False
                    for a2, spos2_c in specie:
                        sdiff = spos1_c - spos2_c
                        sdiff -= num.floor(sdiff + 0.5)
                        if num.dot(sdiff, sdiff) < self.tol:
                            ok = True
                            map[a1] = a2
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                symmok.append((swap, mirror))
                maps.append(map)

        if debug:
            for symmetry, map in zip(symmok, maps):
                swap, mirror = symmetry
                for a1, ZTMB1 in enumerate(self.ZTMB_a):
                    a2 = map[a1]
                    assert ZTMB1 == self.ZTMB_a[a2]
                    spos1_c = self.scale_position(pos_ac[a1])
                    spos2_c = self.scale_position(pos_ac[a2])
                    sdiff = num.take(spos1_c * mirror, swap) - spos2_c
                    sdiff -= num.floor(sdiff + 0.5)
                    assert num.dot(sdiff, sdiff) < self.tol

        self.maps = maps
        self.symmetries = symmok
                
    def check(self, pos_ac):
        """Check(positions) -> boolean

        Check if positions satisfy symmetry operations."""

        nsymold = len(self.symmetries)
        self.prune_symmetries(pos_ac)
        if len(self.symmetries) < nsymold:
            raise RuntimeError('Broken symmetry!')

    def reduce(self, bzk_kc):
        # Add inversion symmetry if it's not there:
        have_inversion_symmetry = False
        for swap_c, mirror_c in self.symmetries:
            if swap_c == (0, 1, 2) and not num.sometrue(mirror_c + 1):
                have_inversion_symmetry = True
                break
        nsym = len(self.symmetries)
        if not have_inversion_symmetry:
            for swap_c, mirror_c in self.symmetries[:nsym]:
                self.symmetries.append((swap_c, -mirror_c))

        nbzkpts = len(bzk_kc)
        ibzk0_kc = num.empty((nbzkpts, 3), num.Float)
        ibzk_kc = ibzk0_kc[:0]
        weight_k = num.ones(nbzkpts, num.Float)
        nibzkpts = 0
        for k_c in bzk_kc[::-1]:
            found = False
            for swap_c, mirror_c in self.symmetries:
                d_kc = num.take(ibzk_kc * mirror_c, swap_c, 1) - k_c
                d_kc *= d_kc
                d_k = num.sum(d_kc, 1) < self.tol
                if num.sometrue(d_k):
                    found = True
                    weight_k[:nibzkpts] += d_k
                    break
            if not found:
                nibzkpts += 1
                ibzk_kc = ibzk0_kc[:nibzkpts]
                ibzk_kc[-1] = k_c

        del self.symmetries[nsym:]

        return ibzk_kc[::-1].copy(), weight_k[:nibzkpts][::-1] / nbzkpts

    def symmetrize(self, a, gd):
        b = a.copy()
        a[:] = 0.0
        for swap, mirror in self.symmetries:
            d = b
            for c, m in enumerate(mirror):
                if m == -1:
                    d = gd.mirror(d, c)
            a += gd.swap_axes(d, swap)
        a /= len(self.symmetries)

    def print_symmetries(self, text):
        n = len(self.symmetries)
        if n == 48:
            text('symmetries: all')
            return
        line1 = []
        line2 = []
        for swap, mirror in self.symmetries:
            line1.extend(['_  '[int(s) + 1] for s in mirror] + [' '])
            line2.extend(['XYZ'[c] for c in swap] + [' '])
        line1 = ''.join(line1)
        line2 = ''.join(line2)
        n1 = 0
        n2 = 64
        text('symmetries:')
        while n1 < 4 * n:
            text('%s\n%s\n' % (line1[n1:n2], line2[n1:n2]), end='')
            n1 = n2
            n2 += 64
