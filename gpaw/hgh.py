from math import sqrt, pi

import numpy as np
from ase.data import atomic_numbers

from gpaw.utilities import pack2, divrl
from gpaw.setup import BaseSetup
from gpaw.spline import Spline
from gpaw.grid_descriptor import AERadialGridDescriptor
from gpaw.grid_descriptor import EquidistantRadialGridDescriptor
from gpaw.atom.atompaw import AtomPAW
from gpaw.atom.configurations import configurations
from gpaw.basis_data import Basis, BasisFunction

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


class HGHSetup(BaseSetup):
    def __init__(self, data, nspins, basis):
        self.data = data

        self.natoms = 0
        self.R_sii = None
        self.HubU = None
        self.lq = None

        self.filename = None
        self.fingerprint = None
        self.symbol = data.symbol
        self.type = data.name
        self.nspins = nspins

        self.Z = data.Z
        self.Nv = data.Nv
        self.Nc = data.Nc
        
        self.ni = sum([2 * l + 1 for l in data.l_j])
        self.pt_j = data.get_projectors()
        self.phit_j = basis.tosplines()
        self.niAO = sum([2 * phit.get_angular_momentum_number() + 1
                         for phit in self.phit_j])

        self.Nct = 0.0
        self.nct = Spline(0, 0.5, [0., 0., 0.])

        self.lmax = 0
        
        self.xc_correction = null_xc_correction

        r, g = data.get_compensation_charge_function()
        self.ghat_l = [Spline(0, r[-1], g)]

        # accuracy is rather sensitive to this
        self.vbar = data.get_local_potential()

        _np = self.ni * (self.ni + 1) // 2
        self.Delta0 = data.Delta0
        self.Delta_pL = np.zeros((_np, 1))
        
        self.E = 0.0
        self.Kc = 0.0
        self.M = 0.0
        self.M_p = np.zeros(_np)
        self.M_pp = np.zeros((_np, _np))
        self.K_p = data.expand_hamiltonian_matrix()
        self.MB = 0.0
        self.MB_p = np.zeros(_np)
        self.O_ii = np.zeros((self.ni, self.ni))

        self.f_j = data.f_j
        self.n_j = data.n_j
        self.l_j = data.l_j
        self.nj = len(data.l_j)

        # We don't really care about these variables
        self.rcutfilter = None
        self.rcore = None

        self.N0_p = np.zeros(_np) # not really implemented
        self.Delta1_jj = None
        self.phicorehole_g = None
        self.beta = None
        self.rcut_j = data.rcut_j
        self.ng = None
        self.tauct = None
        self.Delta_Lii = None
        self.B_ii = None
        self.C_ii = None
        self.X_p = None
        self.ExxC = None
        self.dEH0 = 0.0
        self.dEH_p = np.zeros(_np)
        self.extra_xc_data = {}

        self.wg_lg = None
        self.g_lg = None

class HGHSetupData:
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
        self.symbol = chemsymbol
        self.type = hghdata.symbol
        self.name = 'LDA'
        self.initialize_setup_data()
        
    def initialize_setup_data(self):
        hghdata = self.hghdata
        rgd = AERadialGridDescriptor(0.1, 450, default_spline_points=100)
        #rgd = EquidistantRadialGridDescriptor(0.001, 10000)
        self.rgd = rgd
        
        self.Z = hghdata.Z
        self.Nc = hghdata.Z -  hghdata.Nv
        self.Nv = hghdata.Nv
        
        threshold = 1e-8
        if hghdata.c_n:
            vloc_g = create_local_shortrange_potential(rgd.r_g, hghdata.rloc,
                                                       hghdata.c_n)
            gcutvbar, rcutvbar = self.find_cutoff(rgd.r_g, rgd.dr_g, vloc_g,
                                                  threshold)
            self.vbar_g = sqrt(4.0 * pi) * vloc_g[:gcutvbar]
        else:
            rcutvbar = 0.5
            gcutvbar = rgd.r2g_ceil(rcutvbar)
            self.vbar_g = np.zeros(gcutvbar)
        
        nj = sum([v.nn for v in hghdata.v_l])
        if nj == 0:
            nj = 1 # Code assumes nj > 0 elsewhere, we fill out with zeroes

        if not hghdata.v_l:
            # No projectors.  But the remaining code assumes that everything
            # has projectors!  We'll just add the zero function then
            hghdata.v_l = [VNonLocal(0, 0.01, np.array([[0.]]), [])]

        n_j = []
        l_j = []

        # j ordering is significant, must be nl rather than ln
        for n, l in self.hghdata.nl_iter():
            n_j.append(n + 1) # Note: actual n must be positive!
            l_j.append(l)
        assert nj == len(n_j)
        self.nj = nj
        self.l_j = l_j
        self.n_j = n_j
        
        self.rcut_j = []
        self.pt_jg = []
        
        for n, l in zip(n_j, l_j):
            # Note: even pseudopotentials without projectors will get one
            # projector, but the coefficients h_ij should be zero so it
            # doesn't matter
            pt_g = create_hgh_projector(rgd.r_g, l, n, hghdata.v_l[l].r0)
            norm = sqrt(np.dot(rgd.dr_g, pt_g**2 * rgd.r_g**2))
            assert np.abs(1 - norm) < 1e-5, str(1 - norm)
            gcut, rcut = self.find_cutoff(rgd.r_g, rgd.dr_g, pt_g, threshold)
            if rcut < 0.5:
                rcut = 0.5
                gcut = rgd.r2g_ceil(rcut)
            pt_g = pt_g[:gcut].copy()
            rcut = max(rcut, 0.5)
            self.rcut_j.append(rcut)
            self.pt_jg.append(pt_g)

        # This is the correct magnitude of the otherwise normalized
        # compensation charge
        self.Delta0 = -self.Nv / sqrt(4.0 * pi)

        f_ln = self.hghdata.get_occupation_numbers()
        f_j = [0] * nj
        for j, (n, l) in enumerate(self.hghdata.nl_iter()):
            try:
                f_j[j] = f_ln[l][n]
            except IndexError:
                pass
        self.f_ln = f_ln
        self.f_j = f_j

    def find_cutoff(self, r_g, dr_g, f_g, sqrtailnorm=1e-5):
        g = len(r_g)
        acc_sqrnorm = 0.0
        while acc_sqrnorm <= sqrtailnorm:
            g -= 1
            acc_sqrnorm += (r_g[g] * f_g[g])**2.0 * dr_g[g]
            if r_g[g] < 0.5: # XXX
                return g, r_g[g]
        return g, r_g[g]

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
                    H_mm = np.identity(M2end - M2start) * v.h_nn[n1, n2]
                    H_ii[M1start:M1end, M2start:M2end] += H_mm
                M2start = M2end
            M1start = M1end
        K_p = pack2(H_ii)
        return K_p

    def __str__(self):
        return "HGHSetup('%s')" % self.type

    def __repr__(self):
        return self.__str__()

    def print_info(self, text, _setup):
        self.hghdata.print_info(text)
        
    def plot(self):
        """Plot localized functions of HGH setup."""
        import pylab as pl
        rgd = self.rgd
        
        pl.subplot(211) # vbar, compensation charge
        rloc = self.hghdata.rloc
        gloc = self.rgd.r2g_ceil(rloc)
        gcutvbar = len(self.vbar_g)
        pl.plot(rgd.r_g[:gcutvbar], self.vbar_g, 'r', label='vloc',
                linewidth=3)
        rcc, gcc = self.get_compensation_charge_function()
        
        pl.plot(rcc, gcc * self.Delta0, 'b--', label='Comp charge [arb. unit]',
                linewidth=3)
        pl.legend(loc='best')

        pl.subplot(212) # projectors
        for j, (n, l, pt_g) in enumerate(zip(self.n_j, self.l_j, self.pt_jg)):
            label = 'n=%d, l=%d' % (n, l)
            pl.ylabel('$p_n^l(r)$')
            ng = len(pt_g)
            r_g = rgd.r_g[:ng]
            pl.plot(r_g, pt_g, label=label)
            r0 = self.hghdata.v_l[self.l_j[j]].r0
            g0 = self.rgd.r2g_ceil(r0)
        pl.legend()

    def get_projectors(self):
        # XXX equal-range projectors still required for some reason
        #from gpaw import extra_parameters
        #if extra_parameters.get('usenewlfc', True):
        #    pt_jg = self.pt_jg
        #else: # give projectors equal range
        maxlen = max([len(pt_g) for pt_g in self.pt_jg])
        pt_jg = []        
        for pt1_g in self.pt_jg:
            pt2_g = np.zeros(maxlen)
            pt2_g[:len(pt1_g)] = pt1_g
            pt_jg.append(pt2_g)
        
        pt_j = [self.rgd.reducedspline(l, pt_g)
                for l, pt_g, in zip(self.l_j, pt_jg)]
        return pt_j

    def create_basis_functions(self):
        class SimpleBasis(Basis):
            def __init__(self, symbol, l_j):
                Basis.__init__(self, symbol, 'simple', readxml=False)
                self.generatordata = 'simple'
                self.d = 0.02
                self.ng = 160
                rgd = self.get_grid_descriptor()
                bf_j = self.bf_j
                rcgauss = rgd.rcut / 3.0
                gauss_g = np.exp(-(rgd.r_g / rcgauss)**2.0)
                for l in l_j:
                    phit_g = rgd.r_g**l * gauss_g
                    norm = np.dot((rgd.r_g * phit_g)**2, rgd.dr_g)**.5
                    phit_g /= norm
                    bf = BasisFunction(l, rgd.rcut, phit_g, 'gaussian')
                    bf_j.append(bf)
        b1 = SimpleBasis(self.symbol, range(max(self.l_j) + 1))
        apaw = AtomPAW(self.symbol, [self.f_ln], h=0.05, rcut=9.0,
                       basis={self.symbol: b1},
                       setups={self.symbol : self},
                       lmax=0, txt=None)
        basis = apaw.extract_basis_functions()
        return basis

    def get_compensation_charge_function(self):
        rcgauss = sqrt(2.0) * self.hghdata.rloc
        alpha = rcgauss**-2
        rcutgauss = rcgauss * 5.0 # smaller values break charge conservation
        r = np.linspace(0.0, rcutgauss, 100)
        g = alpha**1.5 * np.exp(-alpha * r**2) * 4.0 / sqrt(pi)
        g[-1] = 0.0
        return r, g

    def get_local_potential(self):
        return self.rgd.reducedspline(0, self.vbar_g)

    def build(self, xcfunc, lmax, nspins, basis):
        if xcfunc.get_setup_name() != 'LDA':
            raise ValueError('HGH setups support only LDA')
        if lmax != 0:
            raise ValueError('HGH setups support only lmax=0')
        if basis is None:
            basis = self.create_basis_functions()
        elif isinstance(basis, str):
            basis = Basis(self.symbol, basis)
        setup = HGHSetup(self, nspins, basis)
        return setup

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
    A = r0**(l + (4 * n - 1) / 2.0)
    assert (4 * n - 1) % 2 == 1
    B = half_integer_gamma[l + (4 * n - 1) // 2]**.5
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
    def __init__(self, l, r0, h_n, k_n=None):
        # We don't deal with spin-orbit coupling so ignore k_n
        self.l = l
        self.r0 = r0
        #assert (l == 0 and len(k_n) == 0) or (len(h_n) == len(k_n))
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


class HGHParameterSet:
    """Wrapper class for HGH-specific data corresponding to one element."""
    def __init__(self, symbol, Z, Nv, rloc, c_n):
        self.symbol = symbol # Identifier, e.g. 'Na', 'Na.sc', ...
        self.Z = Z # Actual atomic number
        self.Nv = Nv # Valence electron count
        self.rloc = rloc # Characteristic radius of local part
        self.c_n = c_n # Polynomial coefficients for local part
        self.v_l = [] # Non-local parts

    def __str__(self):
        strings = ['HGH setup for %s\n' % self.symbol,
                   '    Valence Z=%d, rloc=%.05f\n' % (self.Nv, self.rloc)]

        if self.c_n:
            coef_string = ', '.join(['%.05f' % c for c in self.c_n])
        else:
            coef_string = 'zeros'
        strings.append('    Local part coeffs: %s\n' % coef_string)
        #strings.extend(['\n',
        strings.append('    Projectors:\n')
        for v in self.v_l:
            strings.append('        l=%d, rc=%.05f\n' % (v.l, v.r0))
        strings.append('    Diagonal coefficients of nonlocal parts')
        for v in self.v_l:
            strings.append('\n')
            strings.append('        l=%d: ' % v.l +
                           ', '.join(['%8.05f' % h for h in v.h_n]))
        return ''.join(strings)
        
    def print_info(self, txt):
        txt(str(self))
        txt()

    def nl_iter(self):
        for n in range(4):
            for l, v in enumerate(self.v_l):
                if n < v.nn:
                    yield n, l

    def get_occupation_numbers(self):
        Z, nlfe_j = configurations[self.symbol.split('.')[0]]
        nlfe_j = list(nlfe_j)
        nlfe_j.reverse()
        f_ln = [[], [], []] # [[s], [p], [d]]
        Nv = 0
        for n, l, f, e in nlfe_j:
            Nv += f
            f_n = f_ln[l]
            assert f_n == [] or self.symbol.endswith('.sc')
            f_n.append(f)
            if Nv >= self.Nv:
                break
        assert Nv == self.Nv
        return f_ln

    def zeropad(self):
        """Return a new HGHParameterSet with all arrays zero padded so they
        have the same (max) length for all such HGH setups.  Makes
        plotting multiple HGH setups easier because they have compatible
        arrays."""
        c_n = np.zeros(4)
        for n, c in enumerate(self.c_n):
            c_n[n] = c
        copy = HGHParameterSet(self.symbol, self.Z, self.Nv, self.rloc, c_n)
        v_l = copy.v_l
        for l, v in enumerate(self.v_l):
            h_n = np.zeros(3)
            k_n = np.zeros(3)
            h_n[:len(v.h_n)] = list(v.h_n)
            v2 = VNonLocal(l, v.r0, h_n)
            v_l.append(v2)
        for l in range(len(self.v_l), 3):
            v_l.append(VNonLocal(l, 0.5, np.zeros(3)))
        return copy
        
def parse_local_part(string):
    """Create HGHParameterSet object with local part initialized."""
    tokens = iter(string.split())
    symbol = tokens.next()
    actual_chemical_symbol = symbol.split('.')[0]
    Z = atomic_numbers[actual_chemical_symbol]
    Nv = int(tokens.next())
    rloc = float(tokens.next())
    c_n = [float(token) for token in tokens]
    hgh = HGHParameterSet(symbol, Z, Nv, rloc, c_n)
    return hgh

class HGHBogusNumbersError(ValueError):
    """Error which is raised when the HGH parameters contain f-type
    or higher projectors.  The HGH article only defines atomic Hamiltonian
    matrices up to l=2, so these are meaningless."""
    pass

def parse_hgh_setup(lines):
    """Initialize HGHParameterSet object from text representation."""
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
        if l > 2:
            raise HGHBogusNumbersError
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
        try:
            hgh = parse_hgh_setup(lines)
        except HGHBogusNumbersError:
            continue
        symbol_sc = hgh.symbol.split('.')
        symbol = symbol_sc[0]
        if len(symbol_sc) > 1:
            assert symbol_sc[1] == 'sc'
            sc_setups[symbol] = hgh
        else:
            setups[symbol] = hgh

def plot(symbol, extension=None):
    import pylab as pl
    try:
        s = HGHSetupData(symbol)
    except IndexError:
        print 'Nooooo'
        return
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
