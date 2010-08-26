"""Module for evaluating two-center integrals.

Contains classes for evaluating integrals of the form::

             /
            |   _   _a    _   _b   _
    Theta = | f(r - R ) g(r - R ) dr ,
            | 
           /

with f and g each being given as a radial function times a spherical
harmonic.

Important classes
-----------------

Low-level:

 * OverlapExpansion: evaluate the overlap between a pair of functions (or a
   function with itself) for some displacement vector: <f | g>.  An overlap
   expansion can be created once for a pair of splines f and g, and actual
   values of the overlap can then be evaluated for several different
   displacement vectors.
 * FourierTransformer: create OverlapExpansion object from pair of splines.

Mid-level:

 * TwoSiteOverlapExpansions: evaluate overlaps between two *sets* of functions,
   where functions in the same set reside on the same location: <f_j | g_j>.
 * TwoSiteOverlapCalculator: create TwoSiteOverlapExpansions object from
   pair of lists of splines.

High-level:

 * ManySiteOverlapExpansions:  evaluate overlaps with many functions in many
   locations: <f_aj | g_aj>.
 * ManySiteOverlapCalculator: create ManySiteOverlapExpansions object from
   pair of lists of splines nested by atom and orbital number.

The low-level classes do the actual work, while the higher-level ones depend
on the lower-level ones.

"""

from math import sqrt, pi

import numpy as np
from numpy.fft import ifft

from ase import Atoms
from ase.calculators.neighborlist import NeighborList

from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import Yl, nablaYL
from gpaw.spline import Spline
from gpaw.utilities import _fact
from gpaw.utilities.tools import tri2full
from gpaw.utilities.blas import gemm
from gpaw import extra_parameters

UL = 'L'

# Generate the coefficients for the Fourier-Bessel transform
C = []
a = 0.0
LMAX = 5
if extra_parameters.get('fprojectors'):
    LMAX = 7
for n in range(LMAX):
    c = np.zeros(n + 1, complex)
    for s in range(n + 1):
        a = (1.0j)**s * _fact[n + s] / (_fact[s] * 2**s * _fact[n - s])
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


def spherical_harmonics(R_c, lmax=LMAX):
    R_c = np.asarray(R_c)
    rlY_lm = []
    for l in range(lmax):
        rlY_m = np.empty(2 * l + 1)
        Yl(l, R_c, rlY_m)
        rlY_lm.append(rlY_m)
    return rlY_lm


def spherical_harmonics_and_derivatives(R_c, lmax=LMAX):
    R_c = np.asarray(R_c)
    drlYdR_lmc = []
    rlY_lm = spherical_harmonics(R_c, lmax)
    for l, rlY_m in enumerate(rlY_lm):
        drlYdR_mc = np.empty((2 * l + 1, 3))
        for m in range(2 * l + 1):
            L = l**2 + m
            drlYdR_mc[m, :] = nablaYL(L, R_c)
        drlYdR_lmc.append(drlYdR_mc)
    return rlY_lm, drlYdR_lmc


class BaseOverlapExpansionSet:
    def __init__(self, shape):
        self.shape = shape

    def zeros(self, shape=(), dtype=float):
        return np.zeros(shape + self.shape, dtype=dtype)


class OverlapExpansion(BaseOverlapExpansionSet):
    """A list of real-space splines representing an overlap integral."""
    def __init__(self, la, lb, spline_l):
        self.la = la
        self.lb = lb
        self.spline_l = spline_l
        BaseOverlapExpansionSet.__init__(self, (2 * la + 1, 2 * lb + 1))

    def get_gaunt(self, l):
        la = self.la
        lb = self.lb
        G_mmm = gaunt[la**2:(la + 1)**2,
                      lb**2:(lb + 1)**2,
                      l**2:(l + 1)**2]
        return G_mmm

    def gaunt_iter(self):
        la = self.la
        lb = self.lb
        l = (la + lb) % 2
        for spline in self.spline_l:
            G_mmm = self.get_gaunt(l)
            yield l, spline, G_mmm
            l += 2

    def evaluate(self, r, rlY_lm):
        """Get overlap between localized functions.

        Apply Gaunt coefficients to the list of real-space splines
        describing the overlap integral."""
        x_mi = self.zeros()
        for l, spline, G_mmm in self.gaunt_iter():
            x_mi += spline(r) * np.dot(G_mmm, rlY_lm[l])
        return x_mi

    def derivative(self, r, Rhat_c, rlY_lm, drlYdR_lmc):
        """Get derivative of overlap between localized functions.

        This function assumes r > 0.  If r = 0, i.e. if the functions
        reside on the same atom, the derivative is zero in any case."""
        dxdR_cmi = self.zeros((3,))
        for l, spline, G_mmm in self.gaunt_iter():
            x, dxdr = spline.get_value_and_derivative(r)
            GrlY_mi = np.dot(G_mmm, rlY_lm[l])
            dxdR_cmi += dxdr * GrlY_mi * Rhat_c[:, None, None]
            dxdR_cmi += x * np.dot(G_mmm, drlYdR_lmc[l]).transpose(2, 0, 1)
        return dxdR_cmi


class TwoSiteOverlapExpansions(BaseOverlapExpansionSet):
    def __init__(self, la_j, lb_j, oe_jj):
        self.oe_jj = oe_jj
        shape = (sum([2 * l + 1 for l in la_j]),
                 sum([2 * l + 1 for l in lb_j]))
        BaseOverlapExpansionSet.__init__(self, shape)

    def slice(self, x_xMM):
        assert x_xMM.shape[-2:] == self.shape
        Ma1 = 0
        for j1, oe_j in enumerate(self.oe_jj):
            Mb1 = 0
            Ma2 = Ma1
            for j2, oe in enumerate(oe_j):
                Ma2 = Ma1 + oe.shape[0]
                Mb2 = Mb1 + oe.shape[1]
                yield x_xMM[..., Ma1:Ma2, Mb1:Mb2], oe
                Mb1 = Mb2
            Ma1 = Ma2

    def evaluate(self, r, rlY_lm):
        x_MM = self.zeros()
        for x_mm, oe in self.slice(x_MM):
            x_mm += oe.evaluate(r, rlY_lm)
        return x_MM

    def derivative(self, r, Rhat, rlY_lm, drlYdR_lmc):
        x_cMM = self.zeros((3,))
        for x_cmm, oe in self.slice(x_cMM):
            x_cmm += oe.derivative(r, Rhat, rlY_lm, drlYdR_lmc)
        return x_cMM


class ManySiteOverlapExpansions(BaseOverlapExpansionSet):
    def __init__(self, tsoe_II, I1_a, I2_a):
        self.tsoe_II = tsoe_II
        self.I1_a = I1_a
        self.I2_a = I2_a

        M1 = 0
        M1_a = []
        for I in I1_a:
            M1_a.append(M1)
            M1 += tsoe_II[I, 0].shape[0]
        self.M1_a = M1_a

        M2 = 0
        M2_a = []
        for I in I2_a:
            M2_a.append(M2)
            M2 += tsoe_II[0, I].shape[1]
        self.M2_a = M2_a

        shape = (sum([tsoe_II[I, 0].shape[0] for I in I1_a]),
                 sum([tsoe_II[0, I].shape[1] for I in I2_a]))
        assert (M1, M2) == shape
        BaseOverlapExpansionSet.__init__(self, shape)

    def getslice(self, a1, a2, x_xMM):
        I1 = self.I1_a[a1]
        I2 = self.I2_a[a2]
        tsoe = self.tsoe_II[I1, I2]
        Mstart1 = self.M1_a[a1]
        Mend1 = Mstart1 + tsoe.shape[0]
        Mstart2 = self.M2_a[a2]
        Mend2 = Mstart2 + tsoe.shape[1]
        return x_xMM[..., Mstart1:Mend1, Mstart2:Mend2], tsoe

    def evaluate_slice(self, disp, x_qxMM):
        x_qxmm, oe = self.getslice(disp.a1, disp.a2, x_qxMM)
        disp.evaluate_overlap(oe, x_qxmm)


class DomainDecomposedExpansions(BaseOverlapExpansionSet):
    def __init__(self, msoe, local_indices):
        self.msoe = msoe
        self.local_indices = local_indices
        BaseOverlapExpansionSet.__init__(self, msoe.shape)

    def evaluate_slice(self, disp, x_xqMM):
        if disp.a2 in self.local_indices:
            self.msoe.evaluate_slice(disp, x_xqMM)

class ManySiteDictionaryWrapper(DomainDecomposedExpansions):
    # Used with dictionaries such as P_aqMi and dPdR_aqcMi
    # Works only with NeighborPairs, not SimpleAtomIter, since it
    # compensates for only seeing the atoms once

    def getslice(self, a1, a2, xdict_aqxMi):
        msoe = self.msoe
        tsoe = msoe.tsoe_II[msoe.I1_a[a1], msoe.I2_a[a2]]
        Mstart = self.msoe.M1_a[a1]
        Mend = Mstart + tsoe.shape[0]
        return xdict_aqxMi[a2][..., Mstart:Mend, :], tsoe

    def evaluate_slice(self, disp, x_aqxMi):
        if disp.a2 in x_aqxMi:
            x_qxmi, oe = self.getslice(disp.a1, disp.a2, x_aqxMi)
            disp.evaluate_overlap(oe, x_qxmi)
        if disp.a1 in x_aqxMi and (disp.a1 != disp.a2):
            x2_qxmi, oe2 = self.getslice(disp.a2, disp.a1, x_aqxMi)
            rdisp = disp.reverse()
            rdisp.evaluate_overlap(oe2, x2_qxmi)

class BlacsOverlapExpansions(BaseOverlapExpansionSet):
    def __init__(self, msoe, local_indices, Mmystart, mynao):
        self.msoe = msoe
        self.local_indices = local_indices
        BaseOverlapExpansionSet.__init__(self, msoe.shape)
        
        self.Mmystart = Mmystart
        self.mynao = mynao
        
        M_a = msoe.M1_a
        natoms = len(M_a)
        a = 0
        while a < natoms and M_a[a] <= Mmystart:
            a += 1
        a -= 1
        self.astart = a
        
        while a < natoms and M_a[a] < Mmystart + mynao:
            a += 1
        self.aend = a
            
    def evaluate_slice(self, disp, x_xqNM):
        a1 = disp.a1
        a2 = disp.a2
        if (a2 in self.local_indices and (self.astart <= a1 < self.aend)):
            #assert a2 <= a1
            msoe = self.msoe
            I1 = msoe.I1_a[a1]
            I2 = msoe.I2_a[a2]
            tsoe = msoe.tsoe_II[I1, I2]
            x_qxmm = tsoe.zeros(x_xqNM.shape[:-2], dtype=x_xqNM.dtype)
            disp.evaluate_overlap(tsoe, x_qxmm)
            Mstart1 = msoe.M1_a[a1] - self.Mmystart
            Mend1 = Mstart1 + tsoe.shape[0]
            Mstart1b = max(0, Mstart1)
            Mend1b = min(self.mynao, Mend1)
            Mstart2 = msoe.M2_a[a2]
            Mend2 = Mstart2 + tsoe.shape[1]
            x_xqNM[..., Mstart1b:Mend1b, Mstart2:Mend2] += \
                        x_qxmm[..., Mstart1b - Mstart1:Mend1b - Mstart1, :]
        if a2 < a1:
            # XXX this is necessary to fill out both upper/lower
            #
            # Should not be decided here, and should not be done except
            # in force calculation, *or* force calculation should not require
            # it in the first place
            self.evaluate_slice(disp.reverse(), x_xqNM)
            
class SimpleAtomIter:
    def __init__(self, cell_cv, spos1_ac, spos2_ac, offsetsteps=0):
        self.cell_cv = cell_cv
        self.spos1_ac = spos1_ac
        self.spos2_ac = spos2_ac
        self.offsetsteps = offsetsteps
    
    def iter(self):
        """Yield all atom index pairs and corresponding displacements."""
        offsetsteps = self.offsetsteps
        offsetrange = range(-offsetsteps, offsetsteps + 1)
        offsets = np.array([(i, j, k) for i in offsetrange for j in offsetrange
                            for k in offsetrange])
        for a1, spos1_c in enumerate(self.spos1_ac):
            for a2, spos2_c in enumerate(self.spos2_ac):
                for offset in offsets:
                    R_c = np.dot(spos2_c - spos1_c + offset, self.cell_cv)
                    yield a1, a2, R_c, offset
        

class NeighborPairs:
    """Class for looping over pairs of atoms using a neighbor list."""
    def __init__(self, cutoff_a, cell_cv, pbc_c):
        self.neighbors = NeighborList(cutoff_a, skin=0, sorted=True)
        self.atoms = Atoms('X%d' % len(cutoff_a), cell=cell_cv, pbc=pbc_c)
        # Warning: never use self.atoms.get_scaled_positions() for
        # anything.  Those positions suffer from roundoff errors!
        
    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        self.atoms.set_scaled_positions(spos_ac)
        self.neighbors.update(self.atoms)

    def iter(self):
        cell_cv = self.atoms.cell
        for a1, spos1_c in enumerate(self.spos_ac):
            a2_a, offsets = self.neighbors.get_neighbors(a1)
            for a2, offset in zip(a2_a, offsets):
                spos2_c = self.spos_ac[a2] + offset
                R_c = np.dot(spos2_c - spos1_c, cell_cv)
                yield a1, a2, R_c, offset


class PairFilter:
    def __init__(self, pairs):
        self.pairs = pairs

    def set_positions(self, spos_ac):
        self.pairs.set_positions(spos_ac)

    def iter(self):
        return self.pairs.iter()


class PairsWithSelfinteraction(PairFilter):
    def iter(self):
        for a1, a2, R_c, offset in self.pairs.iter():
            yield a1, a2, R_c, offset
            if a1 == a2 and offset.any():
                yield a1, a1, -R_c, -offset


class PairsBothWays(PairFilter):
    def iter(self):
        for a1, a2, R_c, offset in self.pairs.iter():
            yield a1, a2, R_c, offset
            yield a2, a1, -R_c, -offset


class OppositeDirection(PairFilter):
    def iter(self):
        for a1, a2, R_c, offset in self.pairs.iter():
            yield a2, a1, -R_c, -offset


class FourierTransformer:
    def __init__(self, rcmax, ng):
        self.ng = ng
        self.rcmax = rcmax
        self.dr = rcmax / self.ng
        self.r_g = np.arange(self.ng) * self.dr
        self.Q = 4 * self.ng
        self.dk = 2 * pi / self.Q / self.dr
        self.k_q = np.arange(self.Q // 2) * self.dk

    def transform(self, spline):
        assert spline.get_cutoff() <= self.rcmax, '%s vs %s' % (spline.get_cutoff(), self.rcmax)
        l = spline.get_angular_momentum_number()
        f_g = spline.map(self.r_g)
        f_q = fbt(l, f_g, self.r_g, self.k_q)
        return f_q

    def calculate_overlap_expansion(self, phit1phit2_q, l1, l2):
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
        a_q = phit1phit2_q
        for l in range(lmax % 2, lmax + 1, 2):
            a_g = (8 * fbt(l, a_q * k1**(-2 - lmax - l), self.k_q, R) /
                   R1**(2 * l + 1))
            if l == 0:
                a_g[0] = 8 * np.sum(a_q * k1**(-lmax)) * self.dk
            else:
                a_g[0] = a_g[1]  # XXXX
            a_g *= (-1)**((l1 - l2 - l) // 2)
            n = len(a_g) // 256
            s = Spline(l, 2 * self.rcmax, np.concatenate((a_g[::n], [0.0])))
            splines.append(s)
        return OverlapExpansion(l1, l2, splines)

    def laplacian(self, f_jq):
        return 0.5 * f_jq * self.k_q**2.0


class TwoSiteOverlapCalculator:
    def __init__(self, transformer):
        self.transformer = transformer

    def transform(self, f_j):
        f_jq = np.array([self.transformer.transform(f) for f in f_j])
        return f_jq

    def calculate_expansions(self, la_j, fa_jq, lb_j, fb_jq):
        nja = len(la_j)
        njb = len(lb_j)
        assert nja == len(fa_jq) and njb == len(fb_jq)
        oe_jj = np.empty((nja, njb), dtype=object)
        for ja, (la, fa_q) in enumerate(zip(la_j, fa_jq)):
            for jb, (lb, fb_q) in enumerate(zip(lb_j, fb_jq)):
                a_q = fa_q * fb_q
                oe = self.transformer.calculate_overlap_expansion(a_q, la, lb)
                oe_jj[ja, jb] = oe
        return TwoSiteOverlapExpansions(la_j, lb_j, oe_jj)

    def calculate_kinetic_expansions(self, l_j, f_jq):
        t_jq = self.transformer.laplacian(f_jq)
        return self.calculate_expansions(l_j, f_jq, l_j, t_jq)

    def laplacian(self, f_jq):
        t_jq = self.transformer.laplacian(f_jq)
        return t_jq


class ManySiteOverlapCalculator:
    def __init__(self, twosite_calculator, I1_a, I2_a):
        """Create VeryManyOverlaps object.
        
        twosite_calculator: instance of TwoSiteOverlapCalculator
        I_a: mapping from atom index (as in spos_a) to unique atom type"""
        self.twosite_calculator = twosite_calculator
        self.I1_a = I1_a
        self.I2_a = I2_a

    def transform(self, f_Ij):
        f_Ijq = [self.twosite_calculator.transform(f_j) for f_j in f_Ij]
        return f_Ijq

    def calculate_expansions(self, l1_Ij, f1_Ijq, l2_Ij, f2_Ijq):
        # Naive implementation, just loop over everything
        # We should only need half of them
        nI1 = len(l1_Ij)
        nI2 = len(l2_Ij)
        assert len(l1_Ij) == len(f1_Ijq) and len(l2_Ij) == len(f2_Ijq)
        tsoe_II = np.empty((nI1, nI2), dtype=object)
        calc = self.twosite_calculator
        for I1, (l1_j, f1_jq) in enumerate(zip(l1_Ij, f1_Ijq)):
            for I2, (l2_j, f2_jq) in enumerate(zip(l2_Ij, f2_Ijq)):
                tsoe = calc.calculate_expansions(l1_j, f1_jq, l2_j, f2_jq)
                tsoe_II[I1, I2] = tsoe
        return ManySiteOverlapExpansions(tsoe_II, self.I1_a, self.I2_a)

    def calculate_kinetic_expansions(self, l_Ij, f_Ijq):
        t_Ijq = [self.twosite_calculator.laplacian(f_jq) for f_jq in f_Ijq]
        return self.calculate_expansions(l_Ij, f_Ijq, l_Ij, t_Ijq)


class AtomicDisplacement:
    def __init__(self, factory, a1, a2, R_c, offset, phases):
        self.factory = factory
        self.a1 = a1
        self.a2 = a2
        self.R_c = R_c
        self.offset = offset
        self.phases = phases
        self.r = np.linalg.norm(R_c)
        self._set_spherical_harmonics(R_c)

    def _set_spherical_harmonics(self, R_c):
        self.rlY_lm = spherical_harmonics(R_c)
        
    def _evaluate_without_phases(self, oe):
        return oe.evaluate(self.r, self.rlY_lm)

    def evaluate_overlap(self, oe, dst_xqmm):
        src_xmm = self._evaluate_without_phases(oe)
        self.phases.apply(src_xmm, dst_xqmm)

    def reverse(self):
        return self.factory.displacementclass(self.factory, self.a2, self.a1,
                                              -self.R_c, -self.offset,
                                              self.phases.inverse())
                                              

class DerivativeAtomicDisplacement(AtomicDisplacement):
    def _set_spherical_harmonics(self, R_c):
        self.rlY_lm, self.drlYdr_lmc = spherical_harmonics_and_derivatives(R_c)
        if R_c.any():
            self.Rhat_c = R_c / self.r
        else:
            self.Rhat_c = np.zeros(3)

    def _evaluate_without_phases(self, oe):
        x = oe.derivative(self.r, self.Rhat_c, self.rlY_lm, self.drlYdr_lmc)
        return x


class NullPhases:
    def __init__(self, ibzk_qc, offset):
        pass
    
    def apply(self, src_xMM, dst_qxMM):
        assert len(dst_qxMM) == 1
        dst_qxMM[0][:] += src_xMM

    def inverse(self):
        return self

    
class BlochPhases:
    def __init__(self, ibzk_qc, offset):
        self.phase_q = np.exp(-2j * pi * np.dot(ibzk_qc, offset))
        self.ibzk_qc = ibzk_qc
        self.offset = offset

    def apply(self, src_xMM, dst_qxMM):
        assert dst_qxMM.dtype == complex, dst_qxMM.dtype
        for phase, dst_xMM in zip(self.phase_q, dst_qxMM):
            dst_xMM[:] += phase * src_xMM

    def inverse(self):
        return BlochPhases(-self.ibzk_qc, self.offset)

class TwoCenterIntegralCalculator:
    # This class knows how to apply phases, and whether to call the
    # various derivative() or evaluate() methods
    def __init__(self, ibzk_qc=None, derivative=False):
        if derivative:
            displacementclass = DerivativeAtomicDisplacement
        else:
            displacementclass = AtomicDisplacement
        self.displacementclass = displacementclass

        if ibzk_qc is None or not ibzk_qc.any():
            self.phaseclass = NullPhases
        else:
            self.phaseclass = BlochPhases
        self.ibzk_qc = ibzk_qc
        self.derivative = derivative

    def calculate(self, atompairs, expansions, arrays):
        for disp in self.iter(atompairs):
            for expansion, X_qxMM in zip(expansions, arrays):
                expansion.evaluate_slice(disp, X_qxMM)

    def iter(self, atompairs):
        for a1, a2, R_c, offset in atompairs.iter():
            #if a1 == a2 and self.derivative:
            #    continue
            phase_applier = self.phaseclass(self.ibzk_qc, offset)
            yield self.displacementclass(self, a1, a2, R_c, offset,
                                         phase_applier)


class NewTwoCenterIntegrals:
    def __init__(self, cell_cv, pbc_c, setups, ibzk_qc, gamma):
        self.cell_cv = cell_cv
        self.pbc_c = pbc_c
        self.ibzk_qc = ibzk_qc
        self.gamma = gamma

        cutoff_I = []
        setups_I = setups.setups.values()
        I_setup = {}
        for I, setup in enumerate(setups_I):
            I_setup[setup] = I
            cutoff_I.append(max([func.get_cutoff()
                                 for func in setup.phit_j + setup.pt_j]))
        
        I_a = []
        for setup in setups:
            I_a.append(I_setup[setup])

        cutoff_a = [cutoff_I[I] for I in I_a]

        self.I_a = I_a
        self.setups_I = setups_I        
        self.atompairs = PairsWithSelfinteraction(NeighborPairs(cutoff_a,
                                                                cell_cv,
                                                                pbc_c))
        self.atoms = self.atompairs.pairs.atoms # XXX compatibility

        rcmax = max(cutoff_I + [0.001])

        ng = 2**extra_parameters.get('log2ng', 10)
        transformer = FourierTransformer(rcmax, ng)
        tsoc = TwoSiteOverlapCalculator(transformer)
        self.msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)
        self.calculate_expansions()

        self.calculate = self.evaluate # XXX compatibility

        self.set_matrix_distribution(None, None)
        
    def set_matrix_distribution(self, Mmystart, mynao):
        """Distribute matrices using BLACS."""
        # Range of basis functions for BLACS distribution of matrices:
        self.Mmystart = Mmystart
        self.mynao = mynao
        self.blacs = mynao is not None
        
    def calculate_expansions(self):
        phit_Ij = [setup.phit_j for setup in self.setups_I]
        l_Ij = []
        for phit_j in phit_Ij:
            l_Ij.append([phit.get_angular_momentum_number()
                         for phit in phit_j])
        
        pt_l_Ij = [setup.l_j for setup in self.setups_I]        
        pt_Ij = [setup.pt_j for setup in self.setups_I]
        phit_Ijq = self.msoc.transform(phit_Ij)
        pt_Ijq = self.msoc.transform(pt_Ij)

        msoc = self.msoc

        self.Theta_expansions = msoc.calculate_expansions(l_Ij, phit_Ijq,
                                                          l_Ij, phit_Ijq)
        self.T_expansions = msoc.calculate_kinetic_expansions(l_Ij, phit_Ijq)
        self.P_expansions = msoc.calculate_expansions(l_Ij, phit_Ijq,
                                                      pt_l_Ij, pt_Ijq)

    def _calculate(self, calc, spos_ac, Theta_qxMM, T_qxMM, P_aqxMi):
        for X_xMM in [Theta_qxMM, T_qxMM] + P_aqxMi.values():
            X_xMM.fill(0.0)
        
        self.atompairs.set_positions(spos_ac)

        if self.blacs:
            # S and T matrices are distributed:
            expansions = [
                BlacsOverlapExpansions(self.Theta_expansions,
                                       P_aqxMi, self.Mmystart, self.mynao),
                BlacsOverlapExpansions(self.T_expansions,
                                       P_aqxMi, self.Mmystart, self.mynao)]
        else:
            expansions = [DomainDecomposedExpansions(self.Theta_expansions,
                                                     P_aqxMi),
                          DomainDecomposedExpansions(self.T_expansions,
                                                     P_aqxMi)]
            
        expansions.append(ManySiteDictionaryWrapper(self.P_expansions,
                                                    P_aqxMi))
        arrays = [Theta_qxMM, T_qxMM, P_aqxMi]
        calc.calculate(OppositeDirection(self.atompairs), expansions, arrays)

    def evaluate(self, spos_ac, Theta_qMM, T_qMM, P_aqMi):
        calc = TwoCenterIntegralCalculator(self.ibzk_qc, derivative=False)
        self._calculate(calc, spos_ac, Theta_qMM, T_qMM, P_aqMi)
        if not self.blacs:
            for X_MM in list(Theta_qMM) + list(T_qMM):
                tri2full(X_MM, UL=UL)

    def derivative(self, spos_ac, dThetadR_qcMM, dTdR_qcMM, dPdR_aqcMi):
        calc = TwoCenterIntegralCalculator(self.ibzk_qc, derivative=True)
        self._calculate(calc, spos_ac, dThetadR_qcMM, dTdR_qcMM, dPdR_aqcMi)

        def antihermitian(src, dst):
            np.conj(-src, dst)        

        if not self.blacs:
            for X_cMM in list(dThetadR_qcMM) + list(dTdR_qcMM):
                for X_MM in X_cMM:
                    tri2full(X_MM, UL=UL, map=antihermitian)

    calculate_derivative = derivative # XXX compatibility

    def estimate_memory(self, mem):
        mem.subnode('TCI not impl.', 0)


class OldOverlapExpansion: # Old version, remove this and use OverlapExpansion
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
                      l**2:(l + 1)**2]
        return G_mmm

    def gaunt_iter(self):
        la = self.la
        lb = self.lb
        l = (la + lb) % 2
        for spline in self.spline_l:
            G_mmm = self.get_gaunt(l)
            yield l, spline, G_mmm
            l += 2

    def zeros(self, shape=(), dtype=float):
        return np.zeros(shape + (2 * self.lb + 1, 2 * self.la + 1),
                        dtype=dtype)
    
    def evaluate(self, r, rlY_lm):
        """Get overlap between localized functions.

        Apply Gaunt coefficients to the list of real-space splines
        describing the overlap integral."""
        x_mi = self.zeros()
        for l, spline, G_mmm in self.gaunt_iter():
            x_mi += spline(r) * np.dot(G_mmm, rlY_lm[l])
        return x_mi

    def derivative(self, r, R, rlY_lm, drlYdR_lmc):
        """Get derivative of overlap between localized functions.

        This function assumes r > 0.  If r = 0, i.e. if the functions
        reside on the same atom, the derivative is zero in any case."""
        dxdR_cmi = self.zeros((3,))
        for l, spline, G_mmm in self.gaunt_iter():
            x, dxdr = spline.get_value_and_derivative(r)
            GrlY_mi = np.dot(G_mmm, rlY_lm[l])
            dxdR_cmi += dxdr / r * GrlY_mi * R[:, None, None]
            dxdR_cmi += x * np.dot(G_mmm, drlYdR_lmc[l]).transpose(2, 0, 1)
        return dxdR_cmi
        

class TwoCenterIntegralSplines:
    """ Two-center integrals class.

    This class implements a Fourier-space calculation of two-center
    integrals.
    """

    def __init__(self, rcmax):
        self.rcmax = rcmax
        self.set_ng(2**extra_parameters.get('log2ng', 10))

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
                S[(id1, id2)] = OldOverlapExpansion(l1, l2, s)
                t = self.calculate_splines(0.5 * phit1_q * self.k_q**2,
                                           phit2_q, l1, l2)
                T[(id1, id2)] = OldOverlapExpansion(l1, l2, t)
        for id1, (l1, phit1_q) in phit_jq.items():
            for id2, (l2, pt2_q) in pt_jq.items():
                p = self.calculate_splines(phit1_q, pt2_q, l2, l1) #???
                P[(id1, id2)] = OldOverlapExpansion(l1, l2, p) #???

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
            a_g *= (-1)**((l1 - l2 - l) // 2)
            n = len(a_g) // 256
            s = Spline(l, 2 * self.rcmax, np.concatenate((a_g[::n], [0.0])))
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
    def __init__(self, gd, setups, gamma=True, ibzk_qc=((0., 0., 0.),)):
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

    def calculate(self, spos_ac, S_qMM, T_qMM, P_aqMi, nopawcorrection=False):
        """Calculate values of two-center integrals."""
        self._calculate(spos_ac, S_qMM, T_qMM, P_aqMi, derivative=False,
                        nopawcorrection=nopawcorrection)

    def calculate_derivative(self, spos_ac, dThetadR_qvMM, dTdR_qvMM,
                             dPdR_aqvMi):
        """Calculate derivatives of two-center integrals."""
        self._calculate(spos_ac, dThetadR_qvMM, dTdR_qvMM, dPdR_aqvMi,
                        derivative=True)

    def _calculate(self, spos_ac, S_qxMM, T_qxMM, P_aqxMi, derivative,
                   nopawcorrection=False):
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

            #print a1, a2, offset
            

            selfinteraction = (a1 == a2 and offset.any())
            P1_qxMi = P_aqxMi.get(a1)
            P2_qxMi = P_aqxMi.get(a2)

            if derivative:
                rlY_lm, drlYdR_lmc = spherical_harmonics_and_derivatives(R, 5)
            else:
                if extra_parameters.get('fprojectors'):
                    rlY_lm = spherical_harmonics(R, 7)
                else:
                    rlY_lm = spherical_harmonics(R, 5)
                drlYdR_lmc = []

            self.stp_overlaps(S_qxMM, T_qxMM, P1_qxMi, P2_qxMi, a1, a2,
                              r, R, rlY_lm, drlYdR_lmc, phase_q,
                              selfinteraction, offset, derivative=derivative)

        # Add adjustment from dO_ii, having already calculated <phi_m1|phi_m2>:
        #
        #                         -----
        #                          \            ~a    a   ~a
        # S    = <phi  | phi  > +   )   <phi  | p > dO   <p | phi  >
        #  m1m2      m1     m2     /        m1   i    ij   j     m2
        #                         -----
        #                          aij
        #
        nao = self.setups.nao
        if not derivative and not nopawcorrection:
            dOP_iM = None # Assign explicitly in case loop runs 0 times
            for a, P_qxMi in P_aqxMi.items():
                dO_ii = np.asarray(self.setups[a].dO_ii, P_qxMi.dtype)
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
                S_MM *= -1.0
                T_MM *= -1.0
                for M in range(nao - 1):
                    S_MM[M, M + 1:] = -S_MM[M + 1:, M].conj()
                    T_MM[M, M + 1:] = -T_MM[M + 1:, M].conj()
                                     
    def atom_iter(self, spos_ac, indomain_a,
                  spos2_ac=None, indomain2_a=None):
        """Loop over pairs of atoms and separation vectors.

        indomain_a is a set containing atoms in this domain, which
        must support the operation "a1 in indomain".

        The optional parameters allow looping over distinct lists
        of positions."""
        assert (spos2_ac is None) == (indomain2_a is None)
        if spos2_ac is None:
            spos2_ac = spos_ac
            indomain2_a = indomain_a
        
        cell_cv = self.gd.cell_cv
        for a1, spos1_c in enumerate(spos_ac):
            i, offsets = self.neighbors.get_neighbors(a1)
            for a2, offset in zip(i, offsets):
                if not (a1 in indomain_a or a2 in indomain2_a):
                    continue

                assert a2 >= a1
                spos2_c = spos2_ac[a2] + offset

                R = np.dot(spos2_c - spos1_c, cell_cv)
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
                             1, r, R, rlY_lm, drlYdR_lmc,
                             phase_q, selfinteraction,
                             M1, M2, X_qMM)
            self.overlap(self.P,
                         setup1.symbol, setup2.symbol,
                         setup1.phit_j, setup2.pt_j, 1,
                         r, R, reverse_Y(rlY_lm), reverse_Y(drlYdR_lmc),
                         phase_q.conj(), False, # XXX conj()?
                         M1, 0, P2_qMi.swapaxes(-1, -2))

        if P1_qMi is not None and (a1 != a2 or offset.any()):
            # XXX phase should not be conjugated here?
            # also, what if P2 as well as P1 are not None?
            self.overlap(self.P, setup2.symbol, setup1.symbol,
                         setup2.phit_j, setup1.pt_j, -1, r, R,
                         rlY_lm, drlYdR_lmc, phase_q,
                         False, M2, 0, P1_qMi.swapaxes(-1, -2))
        
    def overlap(self, X, symbol1, symbol2, spline1_j, spline2_j, sign,
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
                X_xmm = sign * splines.derivative(r, R, rlY_lm, drlYdR_lmc)
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
    def set_matrix_distribution(self, Mstart, mynao):
        """Distribute matrices using BLACS."""
        # Range of basis functions for BLACS distribution of matrices:
        self.Mstart = Mstart
        self.Mstop = Mstart + mynao
        
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

    def _calculate(self, spos_ac, S_qxMM, T_qxMM, P_aqxMi, derivative,
                   nopawcorrection=False):
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
                if extra_parameters.get('fprojectors'):
                    rlY_lm = spherical_harmonics(R, 7)
                else:
                    rlY_lm = spherical_harmonics(R, 5)
                drlYdR_lmc = []

            self.stp_overlaps(S_qxMM, T_qxMM, P1_qxMi, P2_qxMi, a1, a2,
                              r, R, rlY_lm, drlYdR_lmc, phase_q,
                              selfinteraction, offset, derivative=derivative)

        # Add adjustment from dO_ii, having already calculated <phi_m1|phi_m2>:
        #
        #                         -----
        #                          \            ~a    a   ~a
        # S    = <phi  | phi  > +   )   <phi  | p > dO   <p | phi  >
        #  m1m2      m1     m2     /        m1   i    ij   j     m2
        #                         -----
        #                          aij
        #
        mynao, nao = S_qxMM.shape[-2:]
        if not derivative and not nopawcorrection:
            dOP_iM = None # Assign explicitly in case loop runs 0 times
            for a, P_qxMi in P_aqxMi.items():
                dO_ii = np.asarray(self.setups[a].dO_ii, P_qxMi.dtype)
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
                                     
    def blacs_overlap(self, X, symbol1, symbol2, spline1_j, spline2_j, sign,
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

            if not drlYdR_lmc:
                X_mm = splines.evaluate(r, rlY_lm).T
            else:
                assert r != 0
                X_xmm = sign * splines.derivative(r, R, rlY_lm, drlYdR_lmc)

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
                             1, r, R, rlY_lm, drlYdR_lmc,
                             phase_q, selfinteraction,
                             M1, M2, X_qMM)
            self.overlap(self.P,
                         setup1.symbol, setup2.symbol,
                         setup1.phit_j, setup2.pt_j, 1,
                         r, R, reverse_Y(rlY_lm), reverse_Y(drlYdR_lmc),
                         phase_q.conj(), False, # XXX conj()?
                         M1, 0, P2_qMi.swapaxes(-1, -2))

        if P1_qMi is not None and (a1 != a2 or offset.any()):
            for X, X_qMM in zip([self.S, self.T], [S_qMM, T_qMM]):
                self.blacs_overlap(X, setup2.symbol, setup1.symbol,
                             setup2.phit_j, setup1.phit_j, -1,
                             r, R, reverse_Y(rlY_lm), reverse_Y(drlYdR_lmc),
                             phase_q.conj(), selfinteraction,
                             M2, M1, X_qMM)
            # XXX phase should not be conjugated here?
            # also, what if P2 as well as P1 are not None?
            self.overlap(self.P, setup2.symbol, setup1.symbol,
                         setup2.phit_j, setup1.pt_j, -1, r, R,
                         rlY_lm, drlYdR_lmc, phase_q,
                         False, M2, 0, P1_qMi.swapaxes(-1, -2))
