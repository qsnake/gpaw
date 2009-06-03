from math import sqrt, pi

import numpy as np
from numpy.fft import ifft

from ase import Atoms
from ase.calculators.neighborlist import NeighborList

from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import Yl, nablaYL
from gpaw.spline import Spline
from gpaw.utilities import fac
from gpaw.utilities.tools import tri2full
from gpaw.utilities.blas import gemm


# Generate the coefficients for the Fourier-Bessel transform
C = []
a = 0.0
n = 5
for n in range(n):
    c = np.zeros(n+1, complex)
    for s in range(n + 1):
        a = (1.0j)**s * fac[n + s] / (fac[s] * 2**s * fac[n - s])
        a *= (-1.0j)**(n + 1)
        c[s] = a
    C.append(c)


def fbt(l, f, r, k):
    """Fast Bessel transform.

    The following integral is calculated using 2l+1 FFTs::

                    oo
                   /
              l+1 |  2           l
      g(k) = k    | r dr j (kr) r f (r)
                  |       l
                 /
                  0
    """

    dr = r[1]
    m = len(k)
    g = np.zeros(m)
    for n in range(l + 1):
        g += (dr * 2 * m * k**(l - n) *
              ifft(C[l][n] * f * r**(1 + l - n), 2 * m)[:m].real)
    return g


class OverlapExpansion:
    """A list of real-space splines representing an overlap integral."""
    def __init__(self, la, lb, spline_l):
        self.la = la
        self.lb = lb
        self.spline_l = spline_l

    def get_gaunt(self, l):
        la = self.la
        lb = self.lb
        G_mmm = gaunt[lb**2:(lb + 1)**2,
                      la**2:(la + 1)**2,
                      l**2:(l + 1)**2].swapaxes(-1, -2)
        return G_mmm

    def gaunt_iter(self):
        la = self.la
        lb = self.lb
        l = (la + lb) % 2
        for spline in self.spline_l:
            G_mmm = self.get_gaunt(l)
            yield l, spline, G_mmm
            l += 2

    def zeros(self, shape=()):
        return np.zeros(shape + (2 * self.lb + 1, 2 * self.la + 1))

    def evaluate(self, r, rlY_lm):
        """Get overlap between localized functions.

        Apply Gaunt coefficients to the list of real-space splines
        describing the overlap integral."""
        x_mi = self.zeros()
        for l, spline, G_mmm in self.gaunt_iter():
            x_mi += spline(r) * np.dot(rlY_lm[l], G_mmm)
        return x_mi

    def derivative(self, r, R, rlY_lm, drlYdR_lmc):
        """Get derivative of overlap between localized functions.

        This function assumes r > 0.  If r = 0, i.e. if the functions
        reside on the same atom, the derivative is zero in any case."""
        dxdR_cmi = self.zeros((3,))
        for l, spline, G_mmm in self.gaunt_iter():
            x, dxdr = spline.get_value_and_derivative(r)
            GrlY_mi = np.dot(rlY_lm[l], G_mmm)
            dxdR_cmi += dxdr / r * GrlY_mi * R[:, None, None]
            dxdR_cmi += x * np.dot(drlYdR_lmc[l].T, G_mmm)
        return dxdR_cmi


def spherical_harmonics(R, lmax=5):
    R = np.asarray(R)
    rlY_lm = []
    for l in range(lmax):
        rlY_m = np.empty(2 * l + 1)
        Yl(l, R, rlY_m)
        rlY_lm.append(rlY_m)
    return rlY_lm


def spherical_harmonics_and_derivatives(R, lmax=5):
    R = np.asarray(R)
    drlYdR_lmc = []
    rlY_lm = spherical_harmonics(R, lmax)
    for l, rlY_m in enumerate(rlY_lm):
        drlYdR_mc = np.empty((2 * l + 1, 3))
        for m in range(2 * l + 1):
            L = l**2 + m
            drlYdR_mc[m, :] = nablaYL(L, R)
        drlYdR_lmc.append(drlYdR_mc)
    return rlY_lm, drlYdR_lmc

class TwoCenterIntegralSplines:
    """ Two-center integrals class.

    This class implements a Fourier-space calculation of two-center
    integrals.
    """

    def __init__(self, rcmax):
        self.rcmax = rcmax
        self.set_ng(2**12)

    def set_ng(self, ng):
        # The ng parameter is rather sensitive.  2**11 might be sufficient,
        # although this will cause the lcao_force test to fail, tripling
        # the error.
        self.ng = ng
        self.dr = self.rcmax / self.ng
        self.r_g = np.arange(self.ng) * self.dr
        self.Q = 4 * self.ng
        self.dk = 2 * pi / self.Q / self.dr
        self.k_q = np.arange(self.Q // 2) * self.dk

    def calculate_dicts(self, symbol_a, phit_aj, pt_aj):
        phit_jq = self.calculate_fft_dict(symbol_a, phit_aj)
        pt_jq = self.calculate_fft_dict(symbol_a, pt_aj)
        
        S = {}
        T = {}
        P = {}
        
        for id1, (l1,  phit1_q) in phit_jq.items():
            for id2, (l2, phit2_q) in phit_jq.items():
                s = self.calculate_splines(phit1_q, phit2_q, l1, l2)
                S[(id1, id2)] = OverlapExpansion(l1, l2, s)
                t = self.calculate_splines(0.5 * phit1_q * self.k_q**2,
                                           phit2_q, l1, l2)
                T[(id1, id2)] = OverlapExpansion(l1, l2, t)
        for id1, (l1, phit1_q) in phit_jq.items():
            for id2, (l2, pt2_q) in pt_jq.items():
                p = self.calculate_splines(phit1_q, pt2_q, l2, l1) #???
                P[(id1, id2)] = OverlapExpansion(l1, l2, p) #???

        return S, T, P

    def calculate_fft_dict_element(self, l, f, f_g):
        f_g[:] = map(f, self.r_g)
        f_q = fbt(l, f_g, self.r_g, self.k_q)
        return f_q
    
    def calculate_fft_dict(self, symbols, f_aj):
        """Calculate Fourier transforms of functions.

        Parameters: list of symbols and list of corresponding splines.

        Returns: a dictionary with keys (symbol, j) and mapping to
        tuples (l, f_q), where f_q are the fourier transforms of f_aj."""
        f_ajq = {}
        f_g = np.empty(self.ng)
        for symbol, f_j in zip(symbols, f_aj):
            for j, f in enumerate(f_j):
                l = f.get_angular_momentum_number()
                id = (symbol, j)
                f_q = self.calculate_fft_dict_element(l, f, f_g)
                f_ajq[id] = (l, f_q)
        return f_ajq

    def calculate_splines(self, phit1_q, phit2_q, l1, l2):
        """Calculate list of splines for one overlap integral.

        Given two Fourier transformed functions, return list of splines
        in real space necessary to evaluate their overlap.

          phi  (q) * phi  (q) --> [phi    (r), ..., phi    (r)] .
             l1         l2            lmin             lmax

        The overlap <phi1 | phi2> can then be calculated by linear
        combinations of the returned splines with appropriate Gaunt
        coefficients.
        """
        lmax = l1 + l2
        splines = []
        R = np.arange(self.Q // 2) * self.dr
        R1 = R.copy()
        R1[0] = 1.0
        k1 = self.k_q.copy()
        k1[0] = 1.0
        a_q = phit1_q * phit2_q
        for l in range(lmax % 2, lmax + 1, 2):
            a_g = (8 * fbt(l, a_q * k1**(-2 - lmax - l), self.k_q, R) /
                   R1**(2 * l + 1))
            if l == 0:
                a_g[0] = 8 * np.sum(a_q * k1**(-lmax)) * self.dk
            else:
                a_g[0] = a_g[1]  # XXXX
            a_g *= (-1)**((-l1 + l2 - l) / 2)
            s = Spline(l, self.Q // self.ng / 2 * self.rcmax, a_g)
            splines.append(s)
        return splines

    def estimate_allocation(self, symbol_a, phit_aj, pt_aj):
        nq = len(self.k_q)
        ng = self.ng

        # The loops and datastructures in this class are very complicated, so
        # we'll subclass it and override all the allocations and expensive
        # operations to just add the array sizes together along the way
        class MemEstimateHack(TwoCenterIntegralSplines):
            def __init__(self, rcmax):
                TwoCenterIntegralSplines.__init__(self, rcmax)
                self.count_fft = 0
                self.count_realspace = 0
            def calculate_fft_dict_element(self, l, f, f_g):
                self.count_fft += nq
                assert ng == len(f_g)
                return 1.0
            def calculate_splines(self, phit1, phit2, l1, l2):
                lmax = l1 + l2
                for l in range(lmax % 2, lmax + 1, 2):
                    self.count_realspace += ng

        meh = MemEstimateHack(self.rcmax)
        meh.calculate_dicts(symbol_a, phit_aj, pt_aj)
        return meh.count_fft, meh.count_realspace

class TwoCenterIntegrals:
    def __init__(self, gd, setups, gamma=True, ibzk_qc=None):
        self.gd = gd
        self.setups = setups
        self.gamma = gamma
        self.ibzk_qc = ibzk_qc
        self.neighbors = None
        self.atoms = None
        self.M_a = None

        rcmax = 0.0
        symbols_a, phit_aj, pt_aj = self.get_symbols_and_phit_and_pt()
        
        for phit_j in phit_aj:
            for phit in phit_j:
                rcmax = max(rcmax, phit.get_cutoff())
        for pt_j in pt_aj:
            for pt in pt_j:
                assert pt.get_cutoff() < rcmax

        self.tci = TwoCenterIntegralSplines(rcmax)
        self.positions_set = False

    def get_symbols_and_phit_and_pt(self):
        """Get the tuple of lists ([symbols...], [phit...], [pt...]).

        Random order."""
        return zip(*[(setup.symbol, setup.phit_j, setup.pt_j) 
                     for setup in self.setups.setups.values()])

    def set_positions(self, spos_ac):
        if not self.positions_set: # First time
            self.positions_set = True # Yuck!  Should be coded properly
            setups = self.setups.setups.values()
            symbols_a, phit_aj, pt_aj = self.get_symbols_and_phit_and_pt()
            S, T, P = self.tci.calculate_dicts(symbols_a, phit_aj, pt_aj)
            self.S = S
            self.T = T
            self.P = P
            
            natoms = len(spos_ac)
            cutoff_a = np.empty(natoms)
            self.M_a = np.empty(natoms, int)
            M = 0
            for a, setup in enumerate(self.setups):
                cutoff_a[a] = max([phit.get_cutoff()
                                   for phit in setup.phit_j])
                self.M_a[a] = M
                M += setup.niAO
            
            self.neighbors = NeighborList(cutoff_a, skin=0, sorted=True)
            self.atoms = Atoms(scaled_positions=spos_ac,
                               cell=self.gd.cell_cv,
                               pbc=self.gd.pbc_c)
        else:
            self.atoms.set_scaled_positions(spos_ac)
        
        self.neighbors.update(self.atoms)

    def calculate(self, spos_ac, S_qMM, T_qMM, P_aqMi):
        """Calculate values of two-center integrals."""
        self._calculate(spos_ac, S_qMM, T_qMM, P_aqMi, derivative=False)

    def calculate_derivative(self, spos_ac, dThetadR_qvMM, dTdR_qvMM,
                             dPdR_aqvMi):
        """Calculate derivatives of two-center integrals."""
        self._calculate(spos_ac, dThetadR_qvMM, dTdR_qvMM, dPdR_aqvMi,
                        derivative=True)

    def _calculate(self, spos_ac, S_qxMM, T_qxMM, P_aqxMi, derivative):
        # Whether we're calculating values or derivatives, most operations
        # are the same.  For this reason the "public" calculate and
        # calculate_derivative methods merely point to this implementation
        # (which would itself appear to have illogical variable names)
        S_qxMM[:] = 0.0
        T_qxMM[:] = 0.0
        for P_qxMi in P_aqxMi.values():
            P_qxMi[:] = 0.0

        for (a1, a2, r, R, phase_q, offset) in self.atom_iter(spos_ac,
                                                              P_aqxMi):
            if derivative and a1 == a2:
                continue

            selfinteraction = (a1 == a2 and offset.any())
            P1_qxMi = P_aqxMi.get(a1)
            P2_qxMi = P_aqxMi.get(a2)

            if derivative:
                rlY_lm, drlYdR_lmc = spherical_harmonics_and_derivatives(R, 5)
            else:
                rlY_lm = spherical_harmonics(R, 5)
                drlYdR_lmc = []

            self.stp_overlaps(S_qxMM, T_qxMM, P1_qxMi, P2_qxMi, a1, a2,
                              r, R, rlY_lm, drlYdR_lmc, phase_q,
                              selfinteraction, offset, derivative=derivative)

        # Add adjustment from O_ii, having already calculated <phi_m1|phi_m2>:
        #
        #                         -----
        #                          \            ~a   a   ~a
        # S    = <phi  | phi  > +   )   <phi  | p > O   <p | phi  >
        #  m1m2      m1     m2     /        m1   i   ij   j     m2
        #                         -----
        #                          aij
        #
        nao = self.setups.nao
        if not derivative:
            dOP_iM = None # Assign explicitly in case loop runs 0 times
            for a, P_qxMi in P_aqxMi.items():
                dO_ii = np.asarray(self.setups[a].O_ii, P_qxMi.dtype)
                for S_MM, P_Mi in zip(S_qxMM, P_qxMi):
                    dOP_iM = np.zeros((dO_ii.shape[1], nao), P_Mi.dtype)
                    # (ATLAS can't handle uninitialized output array)
                    gemm(1.0, P_Mi, dO_ii, 0.0, dOP_iM, 'c')
                    gemm(1.0, dOP_iM, P_Mi, 1.0, S_MM, 'n')
            del dOP_iM

        # As it is now, the derivative calculation does not add the PAW
        # correction.  Rather this is done in the force code.  Perhaps
        # this should be changed.
        comm = self.gd.comm
        comm.sum(S_qxMM)
        comm.sum(T_qxMM)

        for S_MM, T_MM in zip(S_qxMM.reshape(-1, nao, nao),
                              T_qxMM.reshape(-1, nao, nao)):
            if not derivative:
                tri2full(S_MM)
                tri2full(T_MM)
            if derivative:
                for M in range(nao - 1):
                    S_MM[M, M + 1:] = -S_MM[M + 1:, M].conj()
                    T_MM[M, M + 1:] = -T_MM[M + 1:, M].conj()
                                     
    def atom_iter(self, spos_ac, P_aqMi):
        cell_cv = self.gd.cell_cv
        for a1, spos1_c in enumerate(spos_ac):
            P1_qMi = P_aqMi.get(a1)
            i, offsets = self.neighbors.get_neighbors(a1)
            for a2, offset in zip(i, offsets):
                P2_qMi = P_aqMi.get(a2)
                if P1_qMi is None and P2_qMi is None:
                    continue

                assert a2 >= a1
                spos2_c = spos_ac[a2] + offset

                R = -np.dot(spos2_c - spos1_c, cell_cv)
                r = sqrt(np.dot(R, R))

                phase_q = np.exp(-2j * pi * np.dot(self.ibzk_qc, offset))
                phase_q.shape = (-1, 1, 1)

                yield (a1, a2, r, R, phase_q, offset)

    def stp_overlaps(self, S_qMM, T_qMM, P1_qMi, P2_qMi, a1, a2,
                     r, R, rlY_lm, drlYdR_lmc, phase_q, selfinteraction,
                     offset, derivative):
        setup1 = self.setups[a1]
        setup2 = self.setups[a2]
        M1 = self.M_a[a1]
        M2 = self.M_a[a2]

        def reverse_Y(rlY_lm):
            return [rlY_m * (-1)**l
                    for l, rlY_m in enumerate(rlY_lm)]

        if P2_qMi is not None:
            # Calculate basis-basis overlaps:
            for X, X_qMM in zip([self.S, self.T], [S_qMM, T_qMM]):
                self.overlap(X, setup1.symbol, setup2.symbol,
                             setup1.phit_j, setup2.phit_j,
                             r, R, rlY_lm, drlYdR_lmc,
                             phase_q, selfinteraction,
                             M1, M2, X_qMM)
            self.overlap(self.P,
                         setup1.symbol, setup2.symbol,
                         setup1.phit_j, setup2.pt_j,
                         r, R, reverse_Y(rlY_lm), reverse_Y(drlYdR_lmc),
                         phase_q.conj(), False, # XXX conj()?
                         M1, 0, P2_qMi.swapaxes(-1, -2))

        if P1_qMi is not None and (a1 != a2 or offset.any()):
            # XXX phase should not be conjugated here?
            # also, what if P2 as well as P1 are not None?
            self.overlap(self.P, setup2.symbol, setup1.symbol,
                         setup2.phit_j, setup1.pt_j, r, R,
                         rlY_lm, drlYdR_lmc, phase_q,
                         False, M2, 0, P1_qMi.swapaxes(-1, -2))
        
    def overlap(self, X, symbol1, symbol2, spline1_j, spline2_j,
                r, R, rlY_lm, drlYdR_lmc, phase_q, selfinteraction, M1, M2,
                X_qxMM):
        """Calculate overlaps or kinetic energy matrix elements for the
        (a,b) pair of atoms."""
        for M1a, M1b, M2a, M2b, splines in self.slice_iter(symbol1, symbol2,
                                                           M1, M2, spline1_j,
                                                           spline2_j, X):
            if not drlYdR_lmc:
                X_xmm = splines.evaluate(r, rlY_lm)
            else:
                assert r != 0
                X_xmm = splines.derivative(r, R, rlY_lm, drlYdR_lmc)
            X_qxmm = X_xmm
            if not self.gamma:
                dims = np.rank(X_xmm)
                # phase_q has a sort of funny shape (nq, 1, 1) and not (nq,)
                phase_q = phase_q.reshape(-1, *np.ones(np.rank(X_xmm)))
                X_qxmm = X_qxmm * phase_q.conj()
            X_qxMM[..., M2a:M2b, M1a:M1b] += X_qxmm
            if selfinteraction:
                X_qxMM[..., M1a:M1b, M2a:M2b] += X_qxmm.swapaxes(-1, -2).conj()

    def slice_iter(self, symbol1, symbol2, M1, M2, spline1_j, spline2_j, S):
        M1a = M1
        for j1, phit1 in enumerate(spline1_j):
            id1 = (symbol1, j1)
            l1 = phit1.get_angular_momentum_number()
            M2a = M2
            M1b = M1a + 2 * l1 + 1
            for j2, phit2 in enumerate(spline2_j):
                id2 = (symbol2, j2)
                l2 = phit2.get_angular_momentum_number()
                M2b = M2a + 2 * l2 + 1
                splines = S[(id1, id2)]
                yield M1a, M1b, M2a, M2b, splines
                M2a = M2b
            M1a = M1b

    def estimate_memory(self, mem):
        symbol_a, phit_aj, pt_aj = self.get_symbols_and_phit_and_pt()
        fftcount, realspacecount = self.tci.estimate_allocation(symbol_a,
                                                                phit_aj, 
                                                                pt_aj)
        itemsize = np.array(1, dtype=float).itemsize
        mem.subnode('Fourier splines', fftcount * itemsize)
        mem.subnode('Realspace splines', realspacecount * itemsize)


class BlacsTwoCenterIntegrals(TwoCenterIntegrals):
    def set_matrix_distribution(self, band_comm, Mstart, Mstop):
        """Distribute matrices using BLACS."""
        self.band_comm = band_comm
        # Range of basis functions for BLACS distribution of matrices:
        self.Mstart = Mstart
        self.Mstop = Mstop
        
    def set_positions(self, spos_ac):
        TwoCenterIntegrals.set_positions(self, spos_ac)
        natoms = len(spos_ac)
        for a in range(natoms):
            if self.M_a[a] > self.Mstart:
                self.astart = a - 1
                break
        while a < natoms:
            if self.M_a[a] >= self.Mstop:
                self.astop = a
                break
            a += 1
        else:
            self.astop = natoms

    def _calculate(self, spos_ac, S_qxMM, T_qxMM, P_aqxMi, derivative):
        # Whether we're calculating values or derivatives, most operations
        # are the same.  For this reason the "public" calculate and
        # calculate_derivative methods merely point to this implementation
        # (which would itself appear to have illogical variable names)
        S_qxMM[:] = 0.0
        T_qxMM[:] = 0.0
        for P_qxMi in P_aqxMi.values():
            P_qxMi[:] = 0.0

        for (a1, a2, r, R, phase_q, offset) in self.atom_iter(spos_ac,
                                                              P_aqxMi):
            if derivative and a1 == a2:
                continue

            selfinteraction = (a1 == a2 and offset.any())
            P1_qxMi = P_aqxMi.get(a1)
            P2_qxMi = P_aqxMi.get(a2)
            rlY_lm = []
            drlYdR_lmc = []
            for l in range(5):
                rlY_m = np.empty(2 * l + 1)
                Yl(l, R, rlY_m)
                rlY_lm.append(rlY_m)

                if derivative:
                    drlYdR_mc = np.empty((2 * l + 1, 3))
                    for m in range(2 * l + 1):
                        L = l**2 + m
                        drlYdR_mc[m, :] = nablaYL(L, R)
                    drlYdR_lmc.append(drlYdR_mc)

            self.stp_overlaps(S_qxMM, T_qxMM, P1_qxMi, P2_qxMi, a1, a2,
                              r, R, rlY_lm, drlYdR_lmc, phase_q,
                              selfinteraction, offset, derivative=derivative)

        # Add adjustment from O_ii, having already calculated <phi_m1|phi_m2>:
        #
        #                         -----
        #                          \            ~a   a   ~a
        # S    = <phi  | phi  > +   )   <phi  | p > O   <p | phi  >
        #  m1m2      m1     m2     /        m1   i   ij   j     m2
        #                         -----
        #                          aij
        #
        mynao, nao = S_qxMM.shape[-2:]
        if not derivative:
            dOP_iM = None # Assign explicitly in case loop runs 0 times
            for a, P_qxMi in P_aqxMi.items():
                dO_ii = np.asarray(self.setups[a].O_ii, P_qxMi.dtype)
                for S_MM, P_Mi in zip(S_qxMM, P_qxMi):
                    dOP_iM = np.zeros((dO_ii.shape[1], nao), P_Mi.dtype)
                    # (ATLAS can't handle uninitialized output array)
                    gemm(1.0, P_Mi, dO_ii, 0.0, dOP_iM, 'c')
                    gemm(1.0, dOP_iM, P_Mi[self.Mstart:self.Mstop],
                         1.0, S_MM, 'n')
            del dOP_iM

        # As it is now, the derivative calculation does not add the PAW
        # correction.  Rather this is done in the force code.  Perhaps
        # this should be changed.
        comm = self.gd.comm
        comm.sum(S_qxMM)
        comm.sum(T_qxMM)
                                     
    def blacs_overlap(self, X, symbol1, symbol2, spline1_j, spline2_j,
                r, R, rlY_lm, drlYdR_lmc, phase_q, selfinteraction, M1, M2,
                X_qMM):
        """Calculate overlaps or kinetic energy matrix elements for the
        (a,b) pair of atoms."""
        
        for M1a, M1b, M2a, M2b, splines in self.slice_iter(symbol1, symbol2,
                                                           M1, M2, spline1_j,
                                                           spline2_j, X):
            M0 = self.Mstart
            M3 = self.Mstop
            if M1b <= M0 or M1a >= M3:
                continue

            X_mm = splines.evaluate(r, rlY_lm).T
            M1ap = max(M1a, M0)
            M1bp = min(M1b, M3)
            A_qMM = X_qMM[:, M1ap - M0:M1bp - M0, M2a:M2b]
            X_mm = X_mm[M1ap - M1a:M1bp - M1a]
            A_qMM += X_mm

    def stp_overlaps(self, S_qMM, T_qMM, P1_qMi, P2_qMi, a1, a2,
                     r, R, rlY_lm, drlYdR_lmc, phase_q, selfinteraction,
                     offset, derivative):
        setup1 = self.setups[a1]
        setup2 = self.setups[a2]
        M1 = self.M_a[a1]
        M2 = self.M_a[a2]

        def reverse_Y(rlY_lm):
            return [rlY_m * (-1)**l
                    for l, rlY_m in enumerate(rlY_lm)]

        if P2_qMi is not None:
            # Calculate basis-basis overlaps:
            for X, X_qMM in zip([self.S, self.T], [S_qMM, T_qMM]):
                self.blacs_overlap(X, setup1.symbol, setup2.symbol,
                             setup1.phit_j, setup2.phit_j,
                             r, R, rlY_lm, drlYdR_lmc,
                             phase_q, selfinteraction,
                             M1, M2, X_qMM)
            self.overlap(self.P,
                         setup1.symbol, setup2.symbol,
                         setup1.phit_j, setup2.pt_j,
                         r, R, reverse_Y(rlY_lm), reverse_Y(drlYdR_lmc),
                         phase_q.conj(), False, # XXX conj()?
                         M1, 0, P2_qMi.swapaxes(-1, -2))

        if P1_qMi is not None and (a1 != a2 or offset.any()):
            for X, X_qMM in zip([self.S, self.T], [S_qMM, T_qMM]):
                self.blacs_overlap(X, setup2.symbol, setup1.symbol,
                             setup2.phit_j, setup1.phit_j,
                             r, R, reverse_Y(rlY_lm), reverse_Y(drlYdR_lmc),
                             phase_q.conj(), selfinteraction,
                             M2, M1, X_qMM)
            # XXX phase should not be conjugated here?
            # also, what if P2 as well as P1 are not None?
            self.overlap(self.P, setup2.symbol, setup1.symbol,
                         setup2.phit_j, setup1.pt_j, r, R,
                         rlY_lm, drlYdR_lmc, phase_q,
                         False, M2, 0, P1_qMi.swapaxes(-1, -2))
