from math import sqrt, pi

import numpy as npy
from ase.data import atomic_numbers

from gpaw.utilities import pack2
from gpaw.setup import Setup
from gpaw.setup_data import SetupData
from gpaw.basis_data import Basis

setups = {} # Filled out during parsing below

# Tabulated values of Gamma(m + 1/2)
half_integer_gamma = [sqrt(pi)]
for m in range(20):
    half_integer_gamma.append(half_integer_gamma[m] * (m + 0.5))


class HGHSetup(Setup):
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
    
    def __init__(self, symbol, xcfunc, nspins, basis=None,
                 hghdata=None):
        """Creates a HGH setup."""
        assert xcfunc.get_name() == 'LDA' # HGH setups are designed for LDA
        if hghdata is None:
            if not setups.has_key(symbol):
                raise KeyError('No HGH setup for %s' % symbol)
            hghdata = setups[symbol]
        data = self.initialize_setup_data(symbol, xcfunc, hghdata)

        Setup.__init__(self, symbol, xcfunc, lmax=0, nspins=nspins,
                       basis=basis, setupdata=data)
        assert len(data.phit_jg) > 0
        assert len(self.phit_j) > 0

        self.expand_hamiltonian_matrix()

    def find_core_density_cutoff(self, r_g, dr_g, nc_g):
        return 0.5

    def create_basis_functions(self, phit_jg, beta, ng, rcut2,
                               gcut2, r_g):
        raise NotImplementedError('No partial waves for HGH setups from '
                                  'which to create basis functions!  '
                                  'Please specify basis manually.')

    def read_basis_functions(self, basis):
        if isinstance(basis, str):
            basis = Basis(self.symbol, basis)
        rc = basis.d * (basis.ng - 1)
        r_g = npy.linspace(0., rc, basis.ng)
        for bf in basis.bf_j:
            rb_g = r_g[:bf.ng]
            normsqr = npy.sum((bf.phit_g * rb_g)**2) * basis.d
            bf.phit_g /= normsqr ** .5
        return Setup.read_basis_functions(self, basis)

    def initialize_setup_data(self, symbol, xcfunc, hghdata):
        data = SetupData(symbol, xcfunc.get_setup_name(), 'hgh', readxml=False)
        self.hghdata = hghdata

        ng = 450
        beta = .4        
        g = npy.arange(ng, dtype=float)
        self.r_g = r_g = beta * g / (ng - g)
        self.dr_g = dr_g = beta * ng / (ng - g)**2

        self.vloc_g = create_local_shortrange_potential(r_g, hghdata.Nv,
                                                        hghdata.rloc,
                                                        hghdata.c_n)

        # Code probably breaks if we use different radial grids for
        # projectors/partial waves, so we'll use this as filler
        zerofunction = npy.zeros(ng)

        data.tauc_g = zerofunction
        data.tauct_g = zerofunction

        data.Z = hghdata.Z
        data.Nc = hghdata.Z -  hghdata.Nv
        data.Nv = hghdata.Nv
        data.beta = beta
        data.ng = ng
        data.rcgauss = sqrt(2.0) * hghdata.rloc # this is actually correct
        data.e_kinetic = 0.
        data.e_xc = 0.
        data.e_electrostatic = 0.
        data.e_total = 0.
        data.e_kinetic_core = 0.
        
        data.nc_g = zerofunction.copy()
        data.nct_g = zerofunction

        # We use nc_g to emulate that the nucleus does not have charge -Z
        # but rather -Zion = -Z + integral(core density)
        if data.Nc > 0:
            imax = 400
            for i in range(imax):
                x = float(i) / imax
                data.nc_g[i] = (1. - x**2)**2

            amount = npy.dot(data.nc_g, self.r_g**2 * self.dr_g)
            data.nc_g *= data.Nc / amount / sqrt(4 * pi)

        data.nvt_g = zerofunction # nvt is not actually used!
        data.vbar_g = sqrt(4 * pi) * self.vloc_g # !

        nj = sum([v.nn for v in hghdata.v_l])
        if nj == 0:
            nj = 1 # Code assumes nj > 0 elsewhere, we fill out with zeroes
        data.e_kin_jj = npy.zeros((nj, nj)) # XXXX
        data.generatordata = 'HGH'

        if not hghdata.v_l:
            # No projectors.  But the remaining code assumes that everything
            # has projectors!  We'll just add the zero function then
            hghdata.v_l = [VNonLocal(0, 1., npy.array([[0.]]), [])]

        v_l = hghdata.v_l

        # Construct projectors
        electroncount = 0 # Occupations are used during initialization,
        # for this reason we'll have to at least specify the right number

        v_j = []
        n_j = []
        l_j = []

        # j ordering is significant, must be nl rather than ln
        for n in range(4):
            for l, v in enumerate(v_l):
                if n < v.nn:
                    v_j.append(v)
                    n_j.append(n + 1) # Note: actual n must be positive!
                    l_j.append(l)
        assert nj == len(v_j)

        data.l_j = l_j
        data.n_j = n_j

        for n, l, v in zip(n_j, l_j, v_j):
            pt_g = create_hgh_projector(r_g, l, n, v.r0)
            norm = sqrt(npy.dot(dr_g, pt_g**2 * r_g**2))
            assert npy.abs(1 - norm) < 1e-5

            degeneracy = (2 * l + 1) * 2
            f = min(data.Nv - electroncount, degeneracy)
            electroncount += f
            data.f_j.append(f)
            data.eps_j.append(-1.) # probably doesn't matter
            data.rcut_j.append(3.) # XXX I have no idea
            data.id_j.append('%s-%s%d' % (symbol, 'spdf'[l], n))
            
            # Must force projectors to be small so they don't extend
            # past cells and so on.  Of course this is wasteful, but
            # so is the stuff going on in setup.py
            rc = 3.0
            gcut = int(rc * ng / (beta + rc))
            pt_g[gcut:] = 0.0

            data.pt_jg.append(pt_g)
            data.phi_jg.append(zerofunction)
            data.phit_jg.append(zerofunction)

        data.stdfilename = '%s.hgh.LDA' % symbol
        return data

    def print_info(self, text):
        self.hghdata.print_info(text)

    def expand_hamiltonian_matrix(self):
        """Construct H_p from individual h_nn for each l."""
        icount = 0
        Mcount_l = []
        H_lMM = []
        for v in self.hghdata.v_l:
            l = v.l
            mcount = 2 * l + 1
            ncount = v.nn
            Mcount = mcount * ncount
            H_MM = v.h_nn.repeat(mcount, 0).repeat(mcount, 1)
            H_lMM.append(H_MM)
            Mcount_l.append(Mcount)
            icount += Mcount

        assert icount == sum([H_MM.shape[0] for H_MM in H_lMM])

        H_ii = npy.zeros((icount, icount))

        Mstart = 0
        for Mcount, H_MM in zip(Mcount_l, H_lMM):
            Mend = Mstart + Mcount
            H_ii[Mstart:Mend, Mstart:Mend] = H_MM
            Mstart = Mend

        K_p = pack2(H_ii)
        self.K_p[:] = K_p # Maybe simple assignment is enough
        # But this will complain if anything goes wrong with shape etc.


def create_local_shortrange_potential(r_g, Z, rloc, c_n):
    rr_g = r_g / rloc # "Relative r"
    rr2_g = rr_g**2
    rr4_g = rr2_g**2
    rr6_g = rr4_g * rr2_g

    gaussianpart = npy.exp(-.5 * rr2_g)
    polypart = npy.zeros(r_g.shape)
    for c, rrn_g in zip(c_n, [1, rr2_g, rr4_g, rr6_g]):
        polypart += c * rrn_g

    vloc_g = gaussianpart * polypart
    return vloc_g


def create_hgh_projector(r_g, l, n, r0):
    poly_g = r_g**(l + 2 * (n - 1))
    gauss_g = npy.exp(-.5 * r_g**2 / r0**2)
    A = r0**(l + (4 * n - 1)/2.)
    assert (4 * n - 1) % 2 == 1
    B = half_integer_gamma[l + (4 * n - 1) // 2] ** .5
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
        h_nn = npy.zeros((nn, nn))
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
        txt('HGH setup for %s' % self.symbol)
        txt('Valence Z=%.3f, rloc=%.3f' % (self.Nv, self.rloc))
        txt('Local part coeffs')
        txt('  ' + ' '.join(['%5.3f' % c for c in self.c_n]))
        for v in self.v_l:
            txt('Projector  l=%d, rc=%.3f' % (v.l, v.r0))
            for n in range(v.nn):
                txt('  ' + ' '.join(['%7.3f' % h_n[n] for h_n in v.h_nn]))
        
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
        lines = iter(lines)
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


def plot_hgh_setup(setup):
    """Plot short-ranged local part and projectors of HGH setup."""
    import pylab as pl
    pl.plot(setup.r_g, setup.vloc_g, label='Local part')
    for j, pt_g in enumerate(setup.data.pt_jg):
        pl.plot(setup.r_g, setup.r_g * pt_g, label='r p[%d](r)' % j)
    pl.legend()
    pl.show()


def mksetup(symbol):
    from gpaw.xc_functional import XCFunctional
    xcfunc = XCFunctional('LDA')
    setup = HGHSetup(symbol, xcfunc, 1)
    return setup


def test_hgh():
    from gpaw import Calculator
    from ase.data.molecules import molecule

    calc = Calculator(setups='hgh',
                      h=.2,
                      idiotproof=False,
                      basis='sz')

    system = molecule('N', cell=(8.,8.,8.))
    system.center()
    system.set_calculator(calc)
    
    E = system.get_potential_energy()

    sys2 = molecule('N2', cell=(8.,8.,8.))
    sys2.center()
    sys2.set_calculator(calc)
    E2 = sys2.get_potential_energy()

    DE = E2 - 2 * E
    print 'atomization energy', DE

    return calc


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
        setups[hgh.symbol] = hgh


parse()


if __name__ == '__main__':
    test_hgh()
