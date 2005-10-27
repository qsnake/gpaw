# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw import debug, enumerate


class Symmetry:
    def __init__(self, numbers, domain, tolerance=1e-9):
        self.numbers = numbers
        self.cell = domain.cell_i
        self.bc = domain.periodic_i
        self.normalize = domain.normalize
        self.tol = tolerance
        # The identity:
        self.symmetries = [((0, 1, 2), (1, 1, 1))]

    def analyze(self, positions):
        """Analyse(atoms)

        Find a list of symmetry operations."""

                    
        # There are six orderings of the axes:
        allpossibleswaps = [(0, 1, 2), (0, 2, 1),
                            (1, 0, 2), (1, 2, 0),
                            (2, 0, 1), (2, 1, 0)]
        # Only swap axes of equal length:
        cellsyms = [[abs(self.cell[axis1] - self.cell[axis2]) < self.tol and
                     self.bc[axis1] and self.bc[axis2]
                     for axis1 in range(3)]
                    for axis2 in range(3)]
        swaps = []
        for swap in allpossibleswaps:
            ok = True
            for axis1, axis2 in enumerate(swap):
                if axis1 == axis2 or cellsyms[axis1][axis2]:
                    continue
                else:
                    ok = False
                    break
            if ok:
                swaps.append(swap)

        mirrors = [[1], [1], [1]]
        for axis in range(3):
            if self.bc[axis]:
                mirrors[axis].append(-1)
        mirrors = [(m0, m1, m2)
                   for m0 in mirrors[0]
                   for m1 in mirrors[1]
                   for m2 in mirrors[2]]
                            
        self.symmetries = [(swap, mirror)
                           for swap in swaps for mirror in mirrors]
        
        self.prune_symmetries(positions)

    def prune_symmetries(self, positions):
        """prune_symmetries(atoms)

        Remove symmetries that are not satisfied."""

        # Build lists of (atom number, scaled position) tuples.  One list for
        # each atomic number:
        species = {}
        for a, Z in enumerate(self.numbers):
            spos = self.normalize(positions[a])
            if species.has_key(Z):
                species[Z].append((a, spos))
            else:
                species[Z] = [(a, spos)]

        symmok = []
        maps = []
        for swap, mirror in self.symmetries:
            map = num.zeros(len(positions))
            for specie in species.values():
                for a1, spos1 in specie:
                    spos1 = num.take(spos1 * mirror, swap)
                    ok = False
                    for a2, spos2 in specie:
                        sdiff = spos1 - spos2
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
                for a1, Z1 in enumerate(self.numbers):
                    a2 = map[a1]
                    Z2 = self.numbers[a2]
                    assert Z1 == Z2
                    spos1 = self.normalize(positions[a1])
                    spos2 = self.normalize(positions[a2])
                    sdiff = num.take(spos1 * mirror, swap) - spos2
                    sdiff -= num.floor(sdiff + 0.5)
                    assert num.dot(sdiff, sdiff) < self.tol

        self.maps = maps
        self.symmetries = symmok
                
    def check(self, positions):
        """Check(positions) -> boolean

        Check if positions satisfy symmetry operations."""

        nsymold = len(self.symmetries)
        self.prune_symmetries(positions)
        if len(self.symmetries) < nsymold:
            raise RuntimeError('Boken symmetry!')

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
            c = b
            for axis, m in enumerate(mirror):
                if m == -1:
                    c = gd.mirror(c, axis)
            a += gd.swap_axes(c, swap)
        a /= len(self.symmetries)

    def print_symmetries(self, out):
        n = len(self.symmetries)
        print >> out
        if n == 48:
            print >> out, 'symmetries: all'
            return
        line1 = []
        line2 = []
        for swap, mirror in self.symmetries:
            line1.extend(['_  '[s + 1] for s in mirror] + [' '])
            line2.extend(['XYZ'[i] for i in swap] + [' '])
        line1 = ''.join(line1)
        line2 = ''.join(line2)
        n1 = 0
        n2 = 64
        print >> out, 'symmetries:'
        while n1 < 4 * n:
            out.write('%s\n%s\n' % (line1[n1:n2], line2[n1:n2]))
            n1 = n2
            n2 += 64
