import sys
import math

import numpy as npy
import pylab
from ase import Atom, Atoms
from ase.data import chemical_symbols

from gpaw import Calculator
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
from gpaw.localized_functions import create_localized_functions
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import configurations
from gpaw.testing import g2
from gpaw.testing.amoeba import Amoeba

class QuasiGaussian:
    """Implements f(r) = A [G(r) - P(r)] where:

    G(r) = exp{- alpha r^2}
    P(r) = a - b r^2

    with (a, b) such that f(rcut) == f'(rcut) == 0.
    """
    def __init__(self, alpha, rcut, A=1.):
        self.alpha = alpha
        self.rcut = rcut
        a, b = get_polynomial_coefficients(alpha, rcut)
        self.a = a
        self.b = b
        self.A = A
        
    def __call__(self, r):
        """Evaluate function values at r, which is a numpy array"""
        condition = (r < self.rcut) & (self.alpha * r**2 < 700.)
        r2 = npy.where(condition, r**2., 0.) # prevent overflow
        g = npy.exp(-self.alpha * r2)
        p = (self.a - self.b * r2)
        y = npy.where(condition, g - p, 0.)
        return self.A * y

    def renormalize(self, norm):
        self.A /= norm

class LinearCombination:
    def __init__(self, coefs, functions):
        self.coefs = coefs
        self.functions = functions

    def __call__(self, r):
        """Evaluate function values at r, which is a numpy array"""
        return sum([coef * function(r) for coef, function
                    in zip(self.coefs, self.functions)])

    def renormalize(self, norm):
        self.coefs = [coef/norm for coef in self.coefs]

def get_polynomial_coefficients(alpha, rcut):
    """Returns the coefficients (a, b) of the polynomial p(r) = a - b r^2,
    such that the polynomial joins exp(-alpha r**2) differentiably at r=rcut.
    """
    expmar2 = math.exp(-alpha * rcut**2)
    a = (1 + alpha * rcut**2) * expmar2
    b = alpha * expmar2
    return a, b

def gramschmidt(gd, psit_k):
    """Modifies the elements of psit_k such that each scalar product
    < psit_k[i] | psit_k[j] > = delta[ij], where psit_k are on the grid gd"""
    for k in range(len(psit_k)):
        psi = psit_k[k]
        for l in range(k):
            phi = psit_k[l]
            psit_k[k] = psi - gd.integrate(psi*phi) * phi
        psi = psit_k[k]
        psit_k[k] = psi / gd.integrate(psi*psi)**.5

def make_dimer_reference_calculation(formula, a):
    # this function not migrated to npy
    system = g2.get_g2(formula, (a, a, a,))
    calc = Calculator(h=.25)
    system.set_calculator(calc)
    system.get_potential_energy()
    psit = calc.kpt_u[0].psit_nG
    return calc.gd, psit, (system.positions/system.get_cell().diagonal())[0]

def rotation_test():
    molecule = 'N2'
    a = 8.
    rcut = 6.
    l = 1

    rotationvector = npy.array([1.,1.,1.])
    angle_increment = .4
    
    system = g2.get_g2(molecule, (a,a,a))
    calc = Calculator(h=.25, txt=None)
    system.set_calculator(calc)

    pog = PolarizationOrbitalGenerator(rcut)

    r = npy.linspace(0., rcut, 300)

    for i in range(0, int(6.28/angle_increment)):
        print 'angle=%.03f' % (angle_increment * i)
        energy = system.get_potential_energy()
        center = (system.positions / system.get_cell().diagonal())[0]
        orbital = pog.generate(l, calc.gd, calc.kpt_u[0].psit_nG, center)
        pylab.plot(r, orbital(r), label='%.02f' % (i * angle_increment))
        print 'Quality by orbital', pretty(pog.optimizer.lastterms)
        system.rotate(rotationvector, angle_increment)
        system.center()

    pylab.legend()
    pylab.show()

    #orig = system.copy()
    #orig.center(vacuum=a/2.)
    #
    #orig.set_calculator(calc)
    #print 'Calculating energy in g2 system'
    #orig.get_potential_energy()

    #center = (orig.positions / orig.get_cell().diagonal())[0]
    #gd, psit_k, center = calc.gd, calc.kpt_u[0].psit_nG, orig.positions[0]


    #r = npy.linspace(0., rcut, 300)
    #pylab.plot(r, orbital(r))

    #x = system.positions[0][2] / (3.**.5)
    #positions = npy.array([[x,x,x],
    #                       [-x,-x,-x]])
    #system.set_positions(positions)
    #system.center(vacuum=a/2.)

    #system.set_calculator(calc)
    #system.get_potential_energy()

    #center = (system.positions / system.get_cell().diagonal())[0]
    #orbital = pog.generate(l, calc.gd, calc.kpt_u[0].psit_nG, center)
    #print 'Quality by orbital', pretty(pog.optimizer.lastterms)

def make_dummy_calculation(l, rcut, alpha):
    print 'Dummy reference: l=%d, rcut=%.02f, alpha=%.02f' % (l, rcut, alpha)
    g = QuasiGaussian(alpha, rcut)
    r = npy.arange(0., rcut, .01)
    norm = get_norm(r, g(r), l)
    g.renormalize(norm)
    mcount = 2*l + 1
    fcount = 1
    a = 12.
    n = 60
    domain = Domain((a, a, a), (False, False, False))
    gd = GridDescriptor(domain, (n, n, n))
    spline = Spline(l, r[-1], g(r))
    center = (.5, .5, .5)
    lf = create_localized_functions([spline], gd, center)
    psit_k = gd.zeros(mcount)
    coef_xi = npy.identity(mcount * fcount)
    lf.add(psit_k, coef_xi)
    return gd, psit_k, center, g

class CoefficientOptimizer:
    """Given matrices of scalar products s and S as returned by get_matrices,
    finds the optimal set of coefficients resulting in the largest overlap.

    ccount is the number of coefficients.
    if fix is True, the first coefficient will be set to 1. and only the
    remaining coefficients will be subject to optimization.
    """
    def __init__(self, s, S, ccount, fix=False):
        self.s = s
        self.S = S
        self.fix = fix
        function = self.evaluate
        self.lastterms = None
        if fix:
            function = self.evaluate_fixed
            ccount -= 1
        ones = npy.ones((ccount, ccount))
        diag = npy.identity(ccount)
        simplex = npy.concatenate((npy.ones((ccount,1)),
                                   ones + .5 * diag), axis=1)
        simplex = npy.transpose(simplex)
        self.amoeba = Amoeba(simplex, function=function, tolerance=1e-10)
        
    def find_coefficients(self):
        self.amoeba.optimize()
        coefficients = self.amoeba.simplex[0]
        if self.fix:
            coefficients = [1.] + list(coefficients)
        return coefficients

    def evaluate_fixed(self, coef):
        return self.evaluate([1.] + list(coef))

    def evaluate(self, coef):
        coef_trans = npy.array([coef])
        coef = npy.transpose(coef_trans)
        numerator = npy.array([npy.dot(coef_trans, npy.dot(S, coef))
                               for S in self.S])
        denominator = npy.array([npy.dot(coef_trans, npy.dot(s, coef))
                                 for s in self.s])
        terms = numerator / denominator
        assert terms.shape == (len(self.S), 1, 1)
        self.lastterms = terms[:, 0, 0]
        quality = terms.sum()
        badness = - quality
        return badness

def pretty(floatlist):
    return ' '.join(['%.03f' % f for f in floatlist])

def norm_squared(r, f, l):
    dr = r[1]
    frl = f * r**l
    assert abs(r[1] - (r[-1] - r[-2])) < 1e-10 # error if not equidistant
    return sum(frl * frl * r * r * dr)

def get_norm(r, f, l=0):
    return norm_squared(r, f, l) ** .5

class PolarizationOrbitalGenerator:
    def __init__(self, rcut):
        self.rcut = rcut
        amount = int(rcut/.3) # lots!
        r_alphas = npy.linspace(1., rcut, amount)
        self.alphas = 1. / r_alphas**2
        self.s = None
        self.S = None
        self.optimizer = None

    def generate(self, l, gd, psit_k, center):
        rcut = self.rcut
        #print 'Generating polarization function'
        #print 'l=%d, rcut=%.02f Bohr. Gaussians: %d.' % (l, rcut,
        #                                                 len(self.alphas))
        phi_i = [QuasiGaussian(alpha, rcut) for alpha in self.alphas]
        r = npy.arange(0, rcut, .01)
        dr = r[1] # equidistant
        integration_multiplier = r**(2*(l+1))
        for phi in phi_i:
            y = phi(r)
            norm = (dr * sum(y * y * integration_multiplier))**.5
            phi.renormalize(norm)
        splines = [Spline(l, r[-1], phi(r)) for phi in phi_i]
        #print 'Calculating basis/reference overlap integrals'
        self.s, self.S = get_matrices(l, gd, splines, psit_k, center)
        #print 'Optimizing expansion coefficients'
        self.optimizer = CoefficientOptimizer(self.s, self.S, len(phi_i))
        coefs = self.optimizer.find_coefficients()
        self.quality = - self.optimizer.amoeba.y[0]
        self.qualities = self.optimizer.lastterms
        #print 'Quality: %.03f [%s]' % (self.quality,
        #                               pretty(partial_qualities))
        orbital = LinearCombination(coefs, phi_i)
        orbital.renormalize(get_norm(r, orbital(r), l))
        return orbital

def get_matrices(l, gd, splines, psit_k, center=(.5, .5, .5)):
    """Returns the triple-indexed matrices s and S, where
    
      s    = < phi   | phi   > ,
       mij        mi      mj
    
            -----
             \     /        |  ~   \   /  ~   |        \
      S    =  )   (  phi    | psi   ) (  psi  | phi     )
       mij   /     \    mi  |    k /   \    k |     mj /
            -----
              k

    The functions phi are taken from the given splines, whereas psit
    must be on the grid represented by the GridDescriptor gd.
    Integrals are evaluated at the relative location given by center.
    """
    mcounts = [spline.get_angular_momentum_number() for spline in splines]
    mcount = mcounts[0]
    for mcount_i in mcounts:
        assert mcount == mcount_i
    mcount = 2*l + 1
    fcount = len(splines)
    phi_lf = create_localized_functions(splines, gd, center)
    phi_mi = gd.zeros(fcount*mcount) # one set for each phi
    coef_xi = npy.identity(fcount*mcount)
    phi_lf.add(phi_mi, coef_xi)
    integrals = npy.zeros((fcount*mcount, fcount*mcount))
    phi_lf.integrate(phi_mi, integrals)
    """Integral matrix contents (assuming l==1 so there are three m-values)

            --phi1--  --phi2--  --phi3-- ...
            m1 m2 m3  m1 m2 m3  m1 m2 m3 ...
           +---------------------------------
           |
     |   m1| x 0 0     x 0 0
    phi1 m2| 0 x 0     0 x 0   ...
     |   m3| 0 0 x     0 0 x 
           |
     |   m1|   .
    phi2 m2|   .
     |   m3|   .
         . |
         .

    We want < phi_mi | phi_mj >, and thus only the diagonal elements of
    each submatrix!  For l=1 the diagonal elements are all equal, but this
    is not true in general"""

    # phiproducts: for each m, < phi_mi | phi_mj >
    phiproducts_mij = npy.zeros((mcount, fcount, fcount))
    for i in range(fcount):
        for j in range(fcount):
            ioff = mcount*i
            joff = mcount*j
            submatrix_ij = integrals[ioff:ioff+mcount,joff:joff+mcount]
            phiproducts_mij[:, i, j] = submatrix_ij.diagonal()
    # This should be ones in submatrix diagonals and zero elsewhere

    # Now calculate scalar products < phi_mi | psit_k >, where psit_k are
    # solutions from reference calculation
    psitcount = len(psit_k)
    psi_phi_integrals = npy.zeros((psitcount, fcount*mcount))
    phi_lf.integrate(psit_k, psi_phi_integrals)

    # Now psiproducts[k] is a flat list, but we want it to be a matrix with
    # dimensions corresponding to f and m.
    # The first three elements correspond to the same localized function
    # and so on.
    # What we want is one whole matrix for each m-value.
    psiproducts_mik = npy.zeros((mcount, fcount, psitcount))
    for m in range(mcount):
        for i in range(fcount):
            for k in range(psitcount):
                psiproducts_mik[m, i, k] = psi_phi_integrals[k, mcount * i + m]

    # s[ij] = < phi_mi | phi_mj >
    s = phiproducts_mij

    # S[ij] = sum over k: < phi_mi | psit_k > < psit_k | phi_mj >
    S = npy.array([npy.dot(psiproducts_ik, npy.transpose(psiproducts_ik))
                   for psiproducts_ik in psiproducts_mik])

    return s, S

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        args = g2.atoms.keys()
    rcut = 6.
    generator = PolarizationOrbitalGenerator(rcut)
    for symbol in args:
        gd, psit_k, center = Reference(symbol, txt=None).get_reference_data()
        psitcount = len(psit_k)
        gramschmidt(gd, psit_k)
        print 'Wave function count', psitcount
        psit_k_norms = gd.integrate(psit_k * psit_k)

        Z, states = configurations[symbol]
        highest_state = states[-1]
        n, l_atom, occupation, energy = highest_state
        l = l_atom + 1
        
        phi = generator.generate(l, gd, psit_k, center)
        print 'Quality by orbital', pretty(generator.optimizer.lastterms)
        
        r = npy.arange(0., rcut, .01)
        norm = get_norm(r, phi(r), l)

        quality = generator.quality
        orbital = 'spdf'[l]
        style = ['-.', '--','-',':'][l]
        pylab.plot(r, phi(r) * r**l, style,
                   label='%s [type=%s][q=%.03f]' % (symbol, orbital, quality))
    pylab.legend()
    pylab.show()

def dummy_test(lmax=4, rcut=6.):
    generator = PolarizationOrbitalGenerator(rcut)
    r = npy.arange(0., rcut, .01)
    alpha = 1. / (rcut/2.) ** 2.
    for l in range(lmax + 1):
        gd, psit_k, center, ref = make_dummy_calculation(l, rcut, alpha)
        phi = generator.generate(l, gd, psit_k, center)
        pylab.figure(l)
        pylab.plot(r, ref(r)*r**l, 'g', label='ref')
        pylab.plot(r, phi(r)*r**l, 'r', label='pol')
        pylab.title('l=%d' % l)
        pylab.legend()
    pylab.show()

restart_filename = 'ref.%s.gpw'
output_filename = 'ref.%s.txt'

# Systems for non-dimer-forming or troublesome atoms
# 'symbol' : (g2 key, index of desired atom)

special_systems = {'H' : ('HCl', 1), # Better results with more states
                   'Li' : ('LiF', 0), # More states
                   'Na' : ('NaCl', 0), # More states
                   'B' : ('BCl3', 0), # No boron dimer
                   'C' : ('CH4', 0), # No carbon dimer
                   'N' : ('NH3', 0), # double/triple bonds tend to be bad
                   'O' : ('H2O', 0), # O2 requires spin polarization
                   'Al' : ('AlCl3', 0),
                   'Si' : ('SiO', 0), # No reason really.
                   'S' : ('SO2', 0), # S2 requires spin polarization
                   'P' : ('PH3', 0)}

# Calculator parameters for particularly troublesome systems
#special_parameters = {('Be2', 'h') : .2, # malloc thing?
#                      ('Na2', 'h') : .25,
#                      ('Na2', 'a') : 20., # Na is RIDICULOUSLY large
#                      ('LiF', 'h') : .22, # this is probably bad for F
#                      ('LiF', 'a') : 18.} # also large


# Errors with standard parameters
#
# Be2, AlCl3:
# python: c/extensions.h:29: gpaw_malloc: Assertion `p != ((void *)0)' failed.
#
# NaCl, Li:
# Box too close to boundary

def check_magmoms():
    systems = get_systems()
    for formula, index in systems:
        atoms = g2.get_g2(formula, (0,0,0))
        magmom = atoms.get_magnetic_moments()
        if magmom is not None:
            print formula,'has nonzero magnetic moment!!'
    
def get_system(symbol):
    system = special_systems.get(symbol)
    if system is None:
        system = (symbol + '2', 0)
    return system

def get_systems(symbols=None):
    if symbols is None:
        symbols = g2.atoms.keys()
    systems = []
    for symbol in symbols:
        systems.append(get_system(symbol))
    return systems

def multiple_calculations(systems=None, a=None, h=None):
    if systems is None:
        systems = zip(*get_systems())[0]
    formulas = [] # We want a list of unique formulas in case some elements use
    # the same system
    for system in systems:
        if not system in formulas:
            formulas.append(system)
        
    print 'All:', formulas
    for formula in formulas:
        try:
            print formula,
            
            if h is None:
                h = special_parameters.get((formula, 'h'), .17)
            if a is None:
                a = special_parameters.get((formula, 'a'), 14.)

            print '[a=%.03f, h=%.03f] ... ' % (a, h),

            sys.stdout.flush()
                
            make_reference_calculation(formula, a, h)
            print 'OK!'
        except KeyboardInterrupt:
            raise
        except:
            print 'FAILED!'
            traceback.print_exc()

def make_reference_calculation(formula, a, h):
    calc = Calculator(h=h, xc='PBE', txt=output_filename % formula)
    system = g2.get_g2(formula, (a,a,a))
    assert system.get_magnetic_moments() is None
    system.set_calculator(calc)
    energy = system.get_potential_energy()
    calc.write(restart_filename % formula, mode='all')

class Reference:
    def __init__(self, symbol, filename=None, index=None, txt=None):
        if filename is None:
            formula, index = get_system(symbol)
            filename = restart_filename % formula
        calc = Calculator(filename, txt=txt)
        atoms = calc.get_atoms()
        symbols = atoms.get_chemical_symbols()
        if index is None:
            index = symbols.index(symbol)
        else:
            if not symbols[index] == symbol:
                raise ValueError(('Atom (%s) at specified index (%d) not of '+
                                  'requested type (%s)') % (symbols[index],
                                                            index, symbol))

        self.calc = calc
        self.filename = filename
        self.atoms = atoms
        self.symbol = symbol
        self.symbols = symbols
        self.index = index
        self.center = atoms.positions[index]
        self.cell = atoms.get_cell().diagonal() # cubic cell
        self.gpts = calc.gd.N_c
        # NOTE: for spin-polarized calculations, BAD THINGS HAPPEN
        if len(calc.kpt_u) > 1:
            raise RuntimeError('Multiple kpts/spin-pol not supported!!')
        k0 = calc.kpt_u[0]
        if k0.psit_nG is None:
            raise RuntimeError('No wave functions found in .gpw file')

    def get_reference_data(self):
        c = self.calc
        return c.gd, c.kpt_u[0].psit_nG[:], self.center / self.cell

#def load_reference(symbol, filename=None, index=None, txt=None):
    #print 'Loading reference for %s from disk.' % symbol
