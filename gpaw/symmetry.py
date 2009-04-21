# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np

from gpaw import debug


class Symmetry:
    def __init__(self, id_a, cell_cv, pbc_c, tolerance=1e-11):
        """Symmetry object.

        Two atoms can only be identical if they have the same atomic
        numbers, setup types and magnetic moments.  If it is an LCAO
        type of calculation, they must have the same atomic basis
        set also."""

        self.id_a = id_a
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c
        self.tol = tolerance

        # Use full set of symmetries if set to True. False detects only orthogonal
        # symmetry matrices (subset of full symmetry operations in non orthogonal cell)
        self.use_all_symmetries = False

        self.symmetries = [((0, 1, 2), np.array((1, 1, 1)))]
        self.operations = [[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]]
        self.op2sym = [0]

    def analyze(self, spos_ac):
        """Analyse(atoms)

        Find a list of symmetry operations."""

        self.symmetries = [] # Symmetry operations as pairs of swaps and mirrors
        self.operations = [] # Orthogonal symmetry operations as matrices
        self.op2sym = []     # Orthogonal operation to (swap, mirror) symmetry pointer
        
        if self.use_all_symmetries:
            # Analyze all possible symmetry operations
            self.analyze_operations(spos_ac) 
        else:
            # Analyze only orthogonal operations
            self.analyze_symmetries(spos_ac) 

    def analyze_operations(self, spos_ac):

        # Only swap axes of equal length:
        cellsyms = [[abs(np.vdot(self.cell_cv[c1],self.cell_cv[c1])-np.vdot(self.cell_cv[c2],self.cell_cv[c2]))<self.tol and
                     self.pbc_c[c1] and self.pbc_c[c2]
                     for c1 in range(3)]
                    for c2 in range(3)]
        # Metric tensor
        cell_cdt=np.dot(np.transpose(self.cell_cv),self.cell_cv)
        
        # Build auxiliary matrices
        b = []
        for i in range(9):
            a = np.zeros(9, dtype=int)
            a[i] = 1
            b.append(a.reshape(3, 3))
        # Generate all possible 3x3 symmetry matrices using base-3 integers
        power = (6561, 2187, 729, 243, 81, 27, 9, 3, 1)
        for base3id in range(19683):
            operation = np.zeros((3,3))
            m = base3id
            for ip, p in enumerate(power):
                d, m = divmod(m, p)
                operation += (1-d) * b[ip]
            # No zero rows
            if not np.all(np.sum(abs(operation), axis=1)):
                continue
            else:
                # Criterion of a matrix being a symmetry operation 
                cell_cdo = np.dot(self.cell_cv, operation)
                cell_cdodt = np.dot(np.transpose(cell_cdo), cell_cdo)

                if not np.sometrue(cell_cdt - cell_cdodt):
                    if not np.any([(abs(operation[i][np.mod(i + 1, 3)]) + abs(operation[np.mod(i + 1, 3)][i])!=0) and
                                   not cellsyms[i][np.mod(i + 1, 3)] for i in range(3)]):
                        self.operations.append(operation)
                    
        self.prune_symmetries(spos_ac)
        
    def analyze_symmetries(self, spos_ac):
        
        # Generate all possible 3x3 orthogonal matrices
        # There are six orderings of the axes:
        allpossibleswaps = [(0, 1, 2), (0, 2, 1),
                            (1, 0, 2), (1, 2, 0),
                            (2, 0, 1), (2, 1, 0)]
        # Only swap axes of equal length:
        cellsyms = [[abs(np.vdot(self.cell_cv[c1],self.cell_cv[c1])-np.vdot(self.cell_cv[c2],self.cell_cv[c2]))<self.tol and
                     self.pbc_c[c1] and self.pbc_c[c2]
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
            if self.pbc_c[c]:
                mirrors[c].append(-1.0)
        mirrors = [np.array((m0, m1, m2))
                   for m0 in mirrors[0]
                   for m1 in mirrors[1]
                   for m2 in mirrors[2]]

        # Metric tensor
        cell_cdt=np.dot(np.transpose(self.cell_cv),self.cell_cv)

        # Make orthogonal operation matrices out of (swap, mirror) pairs
        for swap in swaps:
            for mirror in mirrors:
                operation=np.array(([[1,0,0],[0,1,0],[0,0,1]]),dtype=int)

                for i1 in range(3):
                    operation[i1]=np.take(operation[i1]*mirror,swap)

                # Generalized criterion of a matrix being a symmetry operation 
                cell_cdo = np.dot(self.cell_cv, operation)
                cell_cdodt = np.dot(np.transpose(cell_cdo), cell_cdo)

                if not np.sometrue(cell_cdt - cell_cdodt):
                    self.operations.append(operation)
                    
        self.prune_symmetries(spos_ac)

    def prune_symmetries(self, spos_ac):
        """prune_symmetries(atoms)

        Remove symmetries that are not satisfied."""

        # Build lists of (atom number, scaled position) tuples.  One
        # list for each combination of atomic number, setup type,
        # magnetic moment and basis set:
        species = {}
        for a, id in enumerate(self.id_a):
            spos_c = spos_ac[a]
            if species.has_key(id):
                species[id].append((a, spos_c))
            else:
                species[id] = [(a, spos_c)]

        opok = []
        maps = []
        # Reduce point group using operation matrices
        for ioperation, operation in enumerate(self.operations):
            map = np.zeros(len(spos_ac), int)
            for specie in species.values():
                for a1, spos1_c in specie:
                    spos1_c = np.dot(operation,spos1_c)
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
                opok.append(operation)
                maps.append(map)

        if debug:
            for symmetry, map in zip(opok, maps):
                for a1, id1 in enumerate(self.id_a):
                    a2 = map[a1]
                    assert id1 == self.id_a[a2]
                    spos1_c = spos_ac[a1]
                    spos2_c = spos_ac[a2]
                    sdiff = np.dot(symmetry, spos1_c) - spos2_c
                    sdiff -= np.floor(sdiff + 0.5)
                    assert np.dot(sdiff, sdiff) < self.tol

        self.opmaps = maps
        self.operations = opok
                
    def check(self, spos_ac):
        """Check(positions) -> boolean

        Check if positions satisfy symmetry operations."""

        nsymold = len(self.operations)
        self.prune_symmetries(spos_ac)
        if len(self.operations) < nsymold:
            raise RuntimeError('Broken symmetry!')

    def reduce(self, bzk_kc):
        # Add inversion symmetry if it's not there:
        have_inversion_symmetry = False
        identity=np.identity(3).ravel()
        for operation in self.operations:
            if sum(abs(np.array(operation).ravel()+identity))<self.tol:
                have_inversion_symmetry = True
                break
        nsym = len(self.operations)
        if not have_inversion_symmetry:
            for operation in self.operations[:nsym]:
                self.operations.append(np.negative(operation))

        nbzkpts = len(bzk_kc)
        ibzk0_kc = np.empty((nbzkpts, 3))
        ibzk_kc = ibzk0_kc[:0]
        weight_k = np.ones(nbzkpts)
        nibzkpts = 0
        for k_c in bzk_kc[::-1]:
            found = False
            for operation in self.operations:
                if len(ibzk_kc)==0:
                    break
                opit=np.transpose(np.linalg.inv(operation))
                d_kc = [np.dot(opit,ibzk_kc[i1]) for i1 in range(len(ibzk_kc))] - k_c
                d_kc *= d_kc
                d_k = d_kc.sum(1) < self.tol
                if d_k.any():
                    found = True
                    weight_k[:nibzkpts] += d_k
                    break
            if not found:
                nibzkpts += 1
                ibzk_kc = ibzk0_kc[:nibzkpts]
                ibzk_kc[-1] = k_c

        del self.operations[nsym:]
        self.symmetries, self.maps, self.op2sym = self.convert_operations(self.operations)

        return ibzk_kc[::-1].copy(), weight_k[:nibzkpts][::-1] / nbzkpts

    def convert_operations(self,operations):
        # Create (swap, mirror) pairs for orthogonal matrices and pointers
        symmetries=[]
        maps=[]
        op2sym=[]
        for ioperation, operation in enumerate(self.operations):
            # Create pair if operation is orthogonal
            if not np.sometrue(np.dot(operation,np.transpose(operation)) - np.identity(3)):
                swap_c,mirror_c=self.break_operation(operation)
                symmetries.append((swap_c, mirror_c))
                op2sym.append(len(symmetries) - 1)
                maps.append(self.opmaps[ioperation]) 
            else:
                op2sym.append(-1)
        return symmetries, maps, op2sym

    def break_operation(self,operation):
        # Break an orthogonal matrix to a (swap, mirror) pair. Auxiliary method.
        swap=[0,0,0]; mirror=np.array([0.,0.,0.])
        for i1 in range(3):
            for i2 in range(3):
                if abs(operation[i1][i2])>0:
                    swap[i1]=i2
                    mirror[i2]=operation[i1][i2]
        return (tuple(swap),mirror)

    def symmetrize(self, a, gd):
        b = a.copy()
        a[:] = 0.0
        for ioperation, operation in enumerate(self.operations):
            # Apply non orthogonal operation
            if self.op2sym[ioperation]==-1:
                a += gd.apply_operation(b, operation)
            else:
                # Apply orthogonal operation using (swap, mirror) pair
                swap, mirror = self.symmetries[self.op2sym[ioperation]]
                d = b
                for c, m in enumerate(mirror):
                    if m == -1:
                        d = gd.mirror(d, c)
                a += gd.swap_axes(d, swap)
        a /= len(self.operations)

    def symmetrize_forces(self, F0_av):
        F_ac = np.zeros_like(F0_av)
        for map_a, symmetry in zip(self.maps, self.symmetries):
            swap, mirror = symmetry
            for a1, a2 in enumerate(map_a):
                F_ac[a2] += np.take(F0_av[a1] * mirror, swap)
        return F_ac / len(self.symmetries)
        
    def print_symmetries(self, text):
        n = len(self.operations)
        text('Symmetries present: %s' % (str(n), 'all')[n == 48])
        n = len(self.symmetries)
        if n == 48:
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
        text('Orthogonal symmetry operations:')
        while n1 < 4 * n:
            text('%s\n%s\n' % (line1[n1:n2], line2[n1:n2]), end='')
            n1 = n2
            n2 += 64
