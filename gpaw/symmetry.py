# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gpaw import debug


class Symmetry:
    def __init__(self, Z_a, magmom_a, domain, tolerance=1e-9):
        self.Z_a = Z_a
        self.magmom_a = magmom_a
        self.cell_c = domain.cell_c
        self.periodic_c = domain.periodic_c
        self.scale_position = domain.scale_position  # XXX ref to domain!
        self.tol = tolerance
        # The identity:
        self.symmetries = [((0, 1, 2), (1, 1, 1))]

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

        mirrors = [[1], [1], [1]]
        for c in range(3):
            if self.periodic_c[c]:
                mirrors[c].append(-1)
        mirrors = [(m0, m1, m2)
                   for m0 in mirrors[0]
                   for m1 in mirrors[1]
                   for m2 in mirrors[2]]
                            
        self.symmetries = [(swap, mirror)
                           for swap in swaps for mirror in mirrors]
        
        self.prune_symmetries(pos_ac)

    def prune_symmetries(self, pos_ac):
        """prune_symmetries(atoms)

        Remove symmetries that are not satisfied."""

        # Build lists of (atom number, scaled position) tuples.  One list for
        # each atomic number:
        species = {}
        for a, ZM in enumerate(zip(self.Z_a, self.magmom_a)):
            spos_c = self.scale_position(pos_ac[a])
            if species.has_key(ZM):
                species[ZM].append((a, spos_c))
            else:
                species[ZM] = [(a, spos_c)]

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
                for a1, (Z1, M1) in enumerate(zip(self.Z_a, self.magmom_a)):
                    a2 = map[a1]
                    Z2 = self.Z_a[a2]
                    M2 = self.magmom_a[a2]
                    assert Z1 == Z2
                    assert M1 == M2
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

    def reduce(self, bzkpts):
        # Add inversion symmetry if it's not there:
        inversion = ((0, 1, 2), (-1, -1, -1))
        if inversion not in self.symmetries:
            nsym = len(self.symmetries)
            for swap, mirror in self.symmetries[:nsym]:
                self.symmetries.append((swap,
                                        (-mirror[0], -mirror[1], -mirror[2])))
            inversionadded = True
        else:
            inversionadded = False

        groups = []
        # k1 < k2:
        for k2, kpt2 in enumerate(bzkpts):
            found = False
            for group in groups:
                k1 = group[0]
                kpt1 = bzkpts[k1]
                found = False
                for swap, mirror in self.symmetries:
                    diff = num.take(kpt1 * mirror, swap) - kpt2
                    if num.dot(diff, diff) < self.tol:
                        group.append(k2)
                        found = True
                        break
                if found:
                    break
            if not found:
                groups.append([k2])

        if inversionadded:
            del self.symmetries[nsym:]

        weight = 1.0 / len(bzkpts)
        kw = [(group[-1], len(group) * weight) for group in groups]
        kw.sort()
##        print groups
##        print kw
        return ([bzkpts[k] for k, w in kw], [w for k, w in kw])

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

    def print_symmetries(self, out):
        n = len(self.symmetries)
        if n == 48:
            print >> out, 'symmetries: all'
            return
        line1 = []
        line2 = []
        for swap, mirror in self.symmetries:
            line1.extend(['_  '[s + 1] for s in mirror] + [' '])
            line2.extend(['XYZ'[c] for c in swap] + [' '])
        line1 = ''.join(line1)
        line2 = ''.join(line2)
        n1 = 0
        n2 = 64
        print >> out, 'symmetries:'
        while n1 < 4 * n:
            out.write('%s\n%s\n' % (line1[n1:n2], line2[n1:n2]))
            n1 = n2
            n2 += 64
