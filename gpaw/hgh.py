from math import sqrt, pi

import numpy as np
from ase.data import atomic_numbers

from gpaw.utilities import pack2, divrl
from gpaw.setup import Setup
from gpaw.setup_data import SetupData
from gpaw.basis_data import Basis
from gpaw.grid_descriptor import AERadialGridDescriptor

setups = {} # Filled out during parsing below
sc_setups = {} # Semicore


# Tabulated values of Gamma(m + 1/2)
half_integer_gamma = [sqrt(pi)]
for m in range(20):
    half_integer_gamma.append(half_integer_gamma[m] * (m + 0.5))

class NullXCCorrection:
    def calculate_energy_and_derivatives(self, D_sp, H_sp, a=None):
        return 0.0

null_xc_correction = NullXCCorrection()

class HGHSetup(SetupData):
    """Setup-compatible class implementing HGH pseudopotential.

    To the PAW code this will appear as a legit PAW setup, but is
    in fact considerably simpler.  In particular, all-electron and
    pseudo partial waves are all zero, and compensation charges do not
    depend on environment.

    A HGH setup has the following form::

                  ----
                   \   
      V = Vlocal +  )  | p  > h   < p |
                   /      i    ij    j
                  ----
                   ij

    Vlocal contains a short-range term which is Gaussian-shaped and
    implemented as vbar of a PAW setup, along with a long-range term
    which goes like 1/r and is implemented in terms of a compensation
    charge.

    The non-local part contains KB projector functions which are
    essentially similar to those in PAW, while h_ij are constants.
    h_ij are determined by setting the K_p variable of the normal
    setup.

    Most other properties of a PAW setup do not exist for HGH setups, for
    which reason they are generally set to zero:

    * All-electron partial waves: always zero
    * Pseudo partial waves: always zero
    * Projectors: HGH projectors
    * Zero potential (vbar): Gaussian times polynomial
    * Compensation charges: One Gaussian-shaped spherically symmetric charge
    * All-electron core density: Delta function corresponding to core electron
      charge
    * Pseudo core density: always zero
    * Pseudo valence density: always zero
    * PS/AE Kinetic energy density: always zero
    * The mysterious constants K_p of a setup correspond to h_ij.

    Note that since the pseudo partial waves are set to zero,
    initialization of atomic orbitals requires loading a custom basis
    set.

    Absolute energies become numerically large since no atomic
    reference is subtracted.
    """
    def __init__(self, hghdata):
        if isinstance(hghdata, str):
            symbol = hghdata
            if symbol.endswith('.sc'):
                hghdata = sc_setups[symbol[:-3]]
            else:
                hghdata = setups[symbol]
        self.hghdata = hghdata

        chemsymbol = hghdata.symbol
        if '.' in chemsymbol:
            chemsymbol, sc = chemsymbol.split('.')
            assert sc == 'sc'
        SetupData.__init__(self, chemsymbol, 'LDA', 'HGH', readxml=False)
        self.initialize_setup_data()

    def get_grid_descriptor(self):
        # Can't replace with truncated rgd, because the code in setup.py
        # will then screw up.  So this method does nothing interesting

        #rcutvbar = 5.0 * self.hghdata.rloc
        # with no v_l, we still have to give an rcut for technical reasons
        #rcutpt = max([1.0] + [242.0 * v.r0 for v in self.hghdata.v_l])
        #rcutmax = max(rcutvbar, rcutpt)
        rgd_big = AERadialGridDescriptor(0.4, 450)
        
        #rcutmax = rgd_big.rc
        #gcut = rgd_big.r2g_ceil(rcutmax)
        #rgd = rgd_big.truncate(gcut)
        return rgd_big # XXX
        
    def initialize_setup_data(self):
        hghdata = self.hghdata
        rgd = self.get_grid_descriptor()
        self.rgd = rgd

        self.vloc_g = create_local_shortrange_potential(rgd.r_g,
                                                        hghdata.rloc,
                                                        hghdata.c_n)
        
        # Code probably breaks if we use different radial grids for
        # projectors/partial waves, so we'll use this as filler
        zerofunction = np.zeros(rgd.ng)
        zerofunction.flags.writeable = False
        
        self.tauc_g = zerofunction
        self.tauct_g = zerofunction
        
        self.Z = hghdata.Z
        self.Nc = hghdata.Z -  hghdata.Nv
        self.Nv = hghdata.Nv
        self.beta = rgd.beta
        self.ng = rgd.ng
        
        self.rcgauss = sqrt(2.0) * hghdata.rloc # this is actually correct
        
        self.e_kinetic = 0.
        self.e_xc = 0.
        self.e_electrostatic = 0.
        self.e_total = 0.
        self.e_kinetic_core = 0.
        
        self.nc_g = zerofunction.copy()
        self.nct_g = zerofunction
        
        # We use nc_g to emulate that the nucleus does not have charge -Z
        # but rather -Zion = -Z + integral(core density)
        if self.Nc > 0:
            imax = 10 # apparently this doesn't matter *at all*!!
            # Except it contributes a lot to Hartree energy
            # XXX This should be excluded from total energies
            for i in range(imax):
                x = float(i) / imax
                self.nc_g[i] = (1. - x**2)**2

            amount = np.dot(self.nc_g, rgd.r_g**2 * rgd.dr_g)
            self.nc_g *= self.Nc / amount / sqrt(4 * pi)

        self.nvt_g = zerofunction # nvt is not actually used!
        self.vbar_g = sqrt(4 * pi) * self.vloc_g

        nj = sum([v.nn for v in hghdata.v_l])
        if nj == 0:
            nj = 1 # Code assumes nj > 0 elsewhere, we fill out with zeroes
        self.e_kin_jj = np.zeros((nj, nj)) # XXXX
        self.generatordata = 'HGH'

        if not hghdata.v_l:
            # No projectors.  But the remaining code assumes that everything
            # has projectors!  We'll just add the zero function then
            hghdata.v_l = [VNonLocal(0, 0.01, np.array([[0.]]), [])]

        v_l = hghdata.v_l

        # Construct projectors
        electroncount = 0 # Occupations are used during initialization,
        # for this reason we'll have to at least specify the right number

        v_j = []
        n_j = []
        l_j = []

        # j ordering is significant, must be nl rather than ln
        for n, l in self.hghdata.nl_iter():
            v_j.append(v_l[l])
            n_j.append(n + 1) # Note: actual n must be positive!
            l_j.append(l)
        assert nj == len(v_j)

        self.l_j = l_j
        self.n_j = n_j

        # Should be different for different elements
        # But so far we use the same.
        # The setup code enforces the same cutoff for vbar and projectors.
        # This is acceptable because the local potentials vloc all
        # have considerably smaller cutoffs than our projectors
        self.max_projector_gcut = rgd.r2g_ceil(6.0)
        rc_projectors = rgd.r_g[self.max_projector_gcut]

        for n, l, v in zip(n_j, l_j, v_j):
            # Note: even pseudopotentials without projectors will get one
            # projector, but the coefficients h_ij should be zero so it
            # doesn't matter
            pt_g = create_hgh_projector(rgd.r_g, l, n, v.r0)
            norm = sqrt(np.dot(rgd.dr_g, pt_g**2 * rgd.r_g**2))
            assert np.abs(1 - norm) < 1e-5

            degeneracy = (2 * l + 1) * 2
            f = min(self.Nv - electroncount, degeneracy)
            electroncount += f
            self.f_j.append(f)
            self.eps_j.append(-1.) # probably doesn't matter
            self.id_j.append('%s-%s%d' % (self.symbol, 'spdf'[l], n))
            self.rcut_j.append(rc_projectors)
            self.pt_jg.append(pt_g)
            self.phi_jg.append(zerofunction)
            self.phit_jg.append(zerofunction)

        self.stdfilename = '%s.hgh.LDA' % self.symbol

    def get_max_projector_cutoff(self):
        return self.max_projector_gcut

    def get_smooth_core_density_integral(self, Delta0):
        return 0.0
    
    def get_overlap_matrix(self, Delta0_ii):
        return np.zeros_like(Delta0_ii)

    def expand_hamiltonian_matrix(self):
        """Construct K_p from individual h_nn for each l."""
        ni = sum([2 * l + 1 for l in self.l_j])

        H_ii = np.zeros((ni, ni))

        # The H_ii used in gpaw is much larger and more general than the one
        # required for HGH pseudopotentials.  This means a lot of the elements
        # must be assigned the same value.  Not a performance issue though,
        # since these are small matrices
        M1start = 0
        for n1, l1 in self.hghdata.nl_iter():
            M1end = M1start + 2 * l1 + 1
            M2start = 0
            v = self.hghdata.v_l[l1]
            for n2, l2 in self.hghdata.nl_iter():
                M2end = M2start + 2 * l2 + 1
                if l1 == l2:
                    H_ii[M1start:M1end, M2start:M2end] = v.h_nn[n1, n2]
                M2start = M2end
            M1start = M1end
        K_p = pack2(H_ii)
        return K_p

    def get_linear_kinetic_correction(self, T0_qp):
        return self.expand_hamiltonian_matrix()

    def find_core_density_cutoff(self, r_g, dr_g, nc_g):
        return 0.5

    def print_info(self, text, _setup):
        self.hghdata.print_info(text)

    def get_ghat(self, lmax, alpha2, r, rcutsoft):
        if lmax > 0:
            raise ValueError('HGH setups support only lmax=0 (lmax=%d)' % lmax)
        ghat_l = SetupData.get_ghat(self, lmax, alpha2, r, rcutsoft)
        return ghat_l

    def get_xc_correction(self, rgd, xcfunc, gcut2, lcut):
        return null_xc_correction

    def create_compensation_charge_functions(self, lmax, r_g, dr_g):
        if lmax != 0:
            raise ValueError('HGH setups support only lmax=0 (lmax=%d)' % lmax)
        g_lg = SetupData.create_compensation_charge_functions(self, lmax, r_g,
                                                              dr_g)
        return g_lg
        
    def plot(self):
        """Plot localized functions of HGH setup."""
        import pylab as pl
        rgd = self.rgd
        gcut = self.get_max_projector_cutoff()
        rcut = rgd.r_g[gcut]
        r_g = rgd.r_g[:gcut]
        dr_g = rgd.dr_g[:gcut]

        pl.subplot(211) # vbar, compensation charge
        rloc = self.hghdata.rloc
        gloc = self.rgd.r2g_ceil(rloc)
        pl.plot([rloc, rloc], [0.0, self.vloc_g[gloc]], '--k')
        extraspace = 5.0
        gplotmax = self.rgd.r2g_ceil(extraspace * rloc)
        pl.plot(r_g, self.vloc_g[:gcut], 'r', label='vloc', linewidth=3)
        g_lg = self.create_compensation_charge_functions(0, r_g, dr_g)
        g_g = g_lg[0]
        if self.vloc_g[0] != 0 and g_g[0] != 0:
            g_g *= self.vloc_g[0] / g_g[0]
        
        pl.plot(r_g, g_g, 'b--', label='Comp charge [arb. unit]', linewidth=3)

        pl.legend(loc='center right')

        pl.subplot(212) # projectors
        for j, (n, l, pt_g) in enumerate(zip(self.n_j, self.l_j, self.pt_jg)):
            label = 'n=%d, l=%d' % (n, l)
            pl.ylabel('$r p_n^l(r) / r^l$')
            pl.plot(r_g, r_g * divrl(pt_g[:gcut], l, r_g), label=label)
                                        
            r0 = self.hghdata.v_l[self.l_j[j]].r0
            g0 = self.rgd.r2g_ceil(r0)
            pl.plot([r0, r0], [0.0, r0 * pt_g[g0] / r0**l], 'k--')
        pl.legend()
        

def create_local_shortrange_potential(r_g, rloc, c_n):
    rr_g = r_g / rloc # "Relative r"
    rr2_g = rr_g**2
    rr4_g = rr2_g**2
    rr6_g = rr4_g * rr2_g

    gaussianpart = np.exp(-.5 * rr2_g)
    polypart = np.zeros(r_g.shape)
    for c, rrn_g in zip(c_n, [1, rr2_g, rr4_g, rr6_g]):
        polypart += c * rrn_g

    vloc_g = gaussianpart * polypart
    return vloc_g


def create_hgh_projector(r_g, l, n, r0):
    poly_g = r_g**(l + 2 * (n - 1))
    gauss_g = np.exp(-.5 * r_g**2 / r0**2)
    A = r0**(l + (4 * n - 1)/2.)
    assert (4 * n - 1) % 2 == 1
    B = half_integer_gamma[l + (4 * n - 1) // 2] ** .5
    #print l, n, B, r0
    pt_g = 2.**.5 / A / B * poly_g * gauss_g
    return pt_g
    

# Coefficients determining off-diagonal elements of h_nn for l = 0...2
# given the diagonal elements
hcoefs_l = [
    [-.5 * (3. / 5.)**.5, .5 * (5. / 21.)**.5, -.5 * (100. / 63.)**.5],
    [-.5 * (5. / 7.)**.5, 1./6. * (35. / 11.)**.5, -1./6. * 14./11.**.5],
    [-.5 * (7. / 9.)**.5, .5 * (63. / 143)**.5, -.5 * 18. / 143.**.5]
    ]


class VNonLocal:
    """Wrapper class for one nonlocal term of an HGH potential."""
    def __init__(self, l, r0, h_n, k_n):
        self.l = l
        self.r0 = r0
        assert (l == 0 and len(k_n) == 0) or (len(h_n) == len(k_n))
        nn = len(h_n)
        self.nn = nn
        h_nn = np.zeros((nn, nn))
        self.h_n = h_n
        self.h_nn = h_nn
        for n, h in enumerate(h_n):
            h_nn[n, n] = h
        if l > 2:
            #print 'Warning: no diagonal elements for l=%d' % l
            # Some elements have projectors corresponding to l=3, but
            # the HGH article only specifies how to calculate the
            # diagonal elements of the atomic hamiltonian for l = 0, 1, 2 !
            return
        coefs = hcoefs_l[l]
        if nn > 2:
            h_nn[0, 2] = h_nn[2, 0] = coefs[1] * h_n[2]
            h_nn[1, 2] = h_nn[2, 1] = coefs[2] * h_n[2]
        if nn > 1:
            h_nn[0, 1] = h_nn[1, 0] = coefs[0] * h_n[1]


class HGHData:
    """Wrapper class for HGH-specific data corresponding to one element."""
    def __init__(self, symbol, Z, Nv, rloc, c_n):
        self.symbol = symbol # Identifier, e.g. 'Na', 'Na.sc', ...
        self.Z = Z # Actual atomic number
        self.Nv = Nv # Valence electron count
        self.rloc = rloc # Characteristic radius of local part
        self.c_n = c_n # Polynomial coefficients for local part
        self.v_l = [] # Non-local parts
        
    def print_info(self, txt):
        txt('HGH setup for %s\n' % self.symbol)
        txt('  Valence Z=%d, rloc=%.05f\n' % (self.Nv, self.rloc))
        txt('  Local part coeffs: ' +
            ', '.join(['%.05f' % c for c in self.c_n]))
        txt('\n')
        txt('  Projectors:\n')
        for v in self.v_l:
            txt('    l=%d, rc=%.05f\n' % (v.l, v.r0))
        txt('  Diagonal coefficients of nonlocal parts\n')
        for v in self.v_l:
            txt('    l=%d: ' % v.l + ', '.join(['%8.05f' % h for h in v.h_n]))
        txt('\n')

    def nl_iter(self):
        for n in range(4):
            for l, v in enumerate(self.v_l):
                if n < v.nn:
                    yield n, l
        
def parse_local_part(string):
    """Create HGHData object with local part initialized."""
    tokens = iter(string.split())
    symbol = tokens.next()
    actual_chemical_symbol = symbol.split('.')[0]
    Z = atomic_numbers[actual_chemical_symbol]
    Nv = int(tokens.next())
    rloc = float(tokens.next())
    c_n = [float(token) for token in tokens]
    hgh = HGHData(symbol, Z, Nv, rloc, c_n)
    return hgh


def parse_hgh_setup(lines):
    """Initialize HGHData object from text representation."""
    lines = iter(lines)
    hgh = parse_local_part(lines.next())

    def pair_up_nonlocal_lines(lines):
        yield lines.next(), ''
        while True:
            yield lines.next(), lines.next()

    for l, (nonlocal, spinorbit) in enumerate(pair_up_nonlocal_lines(lines)):
        nltokens = nonlocal.split()
        sotokens = spinorbit.split()
        r0 = float(nltokens[0])
        h_n = [float(token) for token in nltokens[1:]]
        k_n = [float(token) for token in sotokens]
        vnl = VNonLocal(l, r0, h_n, k_n)
        hgh.v_l.append(vnl)
    return hgh


def parse(filename=None):
    """Read HGH data from file."""
    if filename is None:
        from hgh_parameters import parameters
        all_lines = parameters.splitlines()
    else:
        src = open(filename, 'r')
        all_lines = src.readlines()
        src.close()
    entry_lines = [i for i in xrange(len(all_lines)) 
                   if all_lines[i][0].isalpha()]
    lines_by_element = [all_lines[entry_lines[i]:entry_lines[i + 1]]
                        for i in xrange(len(entry_lines) - 1)]
    lines_by_element.append(all_lines[entry_lines[-1]:])

    for lines in lines_by_element:
        hgh = parse_hgh_setup(lines)
        symbol_sc = hgh.symbol.split('.')
        symbol = symbol_sc[0]
        if len(symbol_sc) > 1:
            assert symbol_sc[1] == 'sc'
            sc_setups[symbol] = hgh
        else:
            setups[symbol] = hgh

def plot(symbol, extension=None):
    import pylab as pl
    s = HGHSetup(symbol)
    s.plot()
    if extension is not None:
        pl.savefig('hgh.%s.%s' % (symbol, extension))

def plot_many(*symbols):
    import pylab as pl
    if not symbols:
        symbols = setups.keys() + [key + '.sc' for key in sc_setups.keys()]
    for symbol in symbols:
        pl.figure(1)
        plot(symbol, extension='png')
        pl.clf()

parse()
