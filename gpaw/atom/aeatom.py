#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, log

import numpy as np
from scipy.integrate import cumtrapz
from scipy.special import gamma
from ase.data import atomic_numbers, atomic_names
import ase.units as units

from gpaw.atom.configurations import configurations
from gpaw.xc_functional import XCFunctional
from gpaw.utilities.progressbar import ProgressBar


class GaussianBasis:
    def __init__(self, l, r_g, alpha_B, eps=1.0e-7):
        """Guassian basis set for spherically symmetric atom.

        l: int
            angular momentum quantum number.
        r_g: ndarray
            Radial grid points.
        alpha_B: ndarray
            Exponents.
        eps: float
            Cutoff for eigenvalues of overlap matrix."""
        
        self.l = l
        self.alpha_B = alpha_B

        A_BB = np.add.outer(alpha_B, alpha_B)
        M_BB = np.multiply.outer(alpha_B, alpha_B)

        # Overlap matrix:
        S_BB = (2 * M_BB**0.5 / A_BB)**(l + 1.5)

        # Find set of liearly independent functions.
        # We have len(alpha_B) gaussians (index B) and self.nbasis
        # linearly independent functions (index b).
        s_B, U_BB = np.linalg.eigh(S_BB)
        self.nbasis = (s_B > eps).sum()

        self.Q_Bb = np.dot(U_BB[:, -self.nbasis:],
                           np.diag(s_B[-self.nbasis:]**-0.5))

        # Kinetic energy matrix:
        self.T_BB = 2**(l + 2.5) * M_BB**(0.5 * l + 0.75) / gamma(l + 1.5) * (
            gamma(l + 2.5) * M_BB / A_BB**(l + 2.5) -
            0.5 * (l + 1) * gamma(l + 1.5) / A_BB**(l + 0.5) +
            0.25 * (l + 1) * (2 * l + 1) * gamma(l + 0.5) / A_BB**(l + 0.5))
        self.T_bb = np.dot(np.dot(self.Q_Bb.T, self.T_BB), self.Q_Bb)

        self.basis_bg = (np.dot(
                self.Q_Bb.T,
                (2 * (2 * alpha_B[:, np.newaxis])**(l + 1.5) /
                 gamma(l + 1.5))**0.5 *
                np.exp(-np.multiply.outer(alpha_B, r_g**2))) * r_g**l)

    def diagonalize(self, vr2dr_g):
        """Diagonalize Schrödinger equation in basis set.

        Returns eigenvalues and eigenvectors."""
        
        H_bb = np.inner(self.basis_bg, self.basis_bg * vr2dr_g)
        H_bb += self.T_bb
        eps_n, C_bn = np.linalg.eigh(H_bb)
        return eps_n, C_bn
    
    def calculate_density(self, f_n, C_bn):
        """Calculate density and density matrix."""
        Cf_bn = C_bn[:, :len(f_n)] * f_n**0.5
        n_g = (np.dot(Cf_bn.T, self.basis_bg)**2).sum(0) / (4 * pi)
        Cf_Bn = np.dot(self.Q_Bb, Cf_bn)
        rho_BB = np.inner(Cf_Bn, Cf_Bn)
        return n_g, rho_BB
        

class AllElectronAtom:
    def __init__(self, symbol, xc='LDA', spinpol=False, log=sys.stdout):
        """All-electron calculation for spherically symmetric atom.

        symbol: str
            Chemical symbol.
        xc: str
            Name of XC-functional.
        spinpol: bool
            If true, do spin-polarized calculation.  Default is spin-paired.
        log: stream
            Text output."""
        
        self.symbol = symbol
        self.Z = atomic_numbers[symbol]

        self.nspins = 1 + int(bool(spinpol))

        if isinstance(xc, str):
            self.xc = XCFunctional(xc, nspins=self.nspins)
        else:
            self.xc = xc
            
        self.fd = log

        self.basis_l = None  # basis functions
        self.vr_sg = None    # potential * r
        self.n_sg = 0.0      # density
        self.C_lsbn = None   # eigenvectors
        self.e_lsn = None    # eigenvalues
        self.f_lsn = None    # occupation numbers
        
        self.initialize_configuration()

        self.log('Atom:            ', atomic_names[self.Z])
        self.log('XC-functional:   ', self.xc.xcname)
        
    def log(self, *args, **kwargs):
        self.fd.write(kwargs.get('sep', ' ').join([str(arg) for arg in args]) +
                      kwargs.get('end', '\n'))

    def initialize_configuration(self):
        self.f_lsn = {}
        for n, l, f, e in configurations[self.symbol][1]:
            if l not in self.f_lsn:
                self.f_lsn[l] = [[] for s in range(self.nspins)]
            if self.nspins == 1:
                self.f_lsn[l][0].append(f)
            else:
                # Use Hund's rule:
                f0 = min(f, 2 * l + 1)
                self.f_lsn[l][0].append(f0)
                self.f_lsn[l][1].append(f - f0)
    def add(self, n, l, df=+1, s=None):
        """Add (remove) electrons."""
        if s is None:
            if self.nspins == 1:
                s = 0
            else:
                self.add(n, l, 0.5 * df, 0)
                self.add(n, l, 0.6 * df, 1)
                return
            
        if l not in self.f_lsn:
            self.f_lsn[l] = [[] for x in range(self.nspins)]
            
        f_n = self.f_lsn[l][s]
        if len(f_n) < n - l:
            f_n.extend([0] * (n - l - len(f_n)))
        f_n[n - l - 1] += df

    def initialize(self, ngpts=1000, alpha1=0.01, alpha2=None, ngauss=50,
                   eps=1.0e-7):
        """Initialize basis sets and radial grid.

        ngpts: int
            Number of grid points for radial grid.
        alpha1: float
            Smallest exponent for gaussian.
        alpha2: float
            Largest exponent for gaussian.
        ngauss: int
            Number of gaussians.
        eps: float
            Cutoff for eigenvalues of overlap matrix.

        The radial grid is::

                beta x                         g
            r = -------,  0 <= x <= 1, x = ---------, g = 0, ..., ngpts - 1
                 1 - x                     ngpts - 1

        """

        self.ngauss = ngauss
        if alpha2 is None:
            alpha2 = 50.0 * self.Z**2

        # Distribute exponents between alpha1 and alpha2:
        self.alpha_B = alpha1 * (alpha2 / alpha1)**np.linspace(0, 1, ngauss)
        self.log('Exponents:        %d (%.3f, %.3f, ..., %.3f, %.3f)' %
                 ((ngauss,) + tuple(self.alpha_B[[0, 1, -2, -1]])))

        # Radial grid:
        self.ngpts = ngpts
        beta = ngpts / alpha2**0.5 / 10
        self.x_g = np.linspace(0, 1, ngpts)
        self.r_g = beta * self.x_g / (1 - self.x_g)
        self.r_g[-1] = 1.0e9
        dx = 1.0 / (ngpts - 1)
        self.dr_g = dx / beta * (self.r_g / self.x_g)**2
        self.dr_g[0] = dx * beta
        self.dr_g[-1] = 1.0e9

        # Maximum l value:
        self.lmax = max(self.f_lsn.keys())

        self.basis_l = []
        for l in range(self.lmax + 1):
            basis = GaussianBasis(l, self.r_g, self.alpha_B, eps)
            self.basis_l.append(basis)

        self.log('Basis functions:  %s (%s)' %
                 (', '.join([str(basis.nbasis) for basis in self.basis_l]),
                  ', '.join('spdf'[:self.lmax + 1])))

        self.log('Grid points:      r=b*x/(1-x),',
                 'b=%.3f and x=i/N for i=0,...,N and N=%d' %
                 (beta, ngpts - 1))

        self.vr_sg = np.zeros((self.nspins, ngpts))
        self.vr_sg[:] = -self.Z
        
    def diagonalize(self):
        """Diagonalize Schrödinger equation."""
        self.eps_lsn = []
        self.C_lsbn = []
        for basis in self.basis_l:
            self.eps_lsn.append([])
            self.C_lsbn.append([])
            for vr_g in self.vr_sg:
                eps_n, C_bn = basis.diagonalize(self.r_g * vr_g * self.dr_g)
                self.eps_lsn[basis.l].append(eps_n)
                self.C_lsbn[basis.l].append(C_bn)

    def calculate_density(self):
        """Calculate elctron density and kinetic energy."""
        self.n_sg = np.zeros((self.nspins, self.ngpts))
        self.ekin = 0.0
        for l, basis in enumerate(self.basis_l):
            for s, f_n in enumerate(self.f_lsn[l]):
                n_g, rho_BB = basis.calculate_density(np.array(f_n),
                                                      self.C_lsbn[l][s])
                self.n_sg[s] += n_g
                self.ekin += (basis.T_BB * rho_BB).sum()

    def calculate_electrostatic_potential(self):
        """Calculate electrostatic potential and energy."""
        self.vHr_g = vHr_g = np.zeros(self.ngpts)
        n_g = self.n_sg.sum(0)
        vHr_g[1:] = -cumtrapz(n_g * self.r_g * self.dr_g) * (4 * pi)
        vHr_g -= vHr_g[-1]
        vHr_g *= self.r_g
        vHr_g[1:] += cumtrapz(n_g * self.r_g**2 * self.dr_g) * (4 * pi)
        
        self.eH = np.trapz(n_g * self.vHr_g * self.r_g * self.dr_g) * 2 * pi
        self.eZ = -self.Z * np.trapz(n_g * self.r_g * self.dr_g) * 4 * pi
        
    def calculate_xc_potential(self):
        self.vxc_sg = np.zeros((self.nspins, self.ngpts))
        exc_g = np.zeros(self.ngpts)
        if self.nspins == 1:
            self.xc.calculate_spinpaired(exc_g, self.n_sg[0], self.vxc_sg[0])
        else:
            self.xc.calculate_spinpolarized(exc_g,
                                            self.n_sg[0], self.vxc_sg[0],
                                            self.n_sg[1], self.vxc_sg[1])
        exc_g[-1] = 0.0
        self.exc = np.trapz(exc_g * self.r_g**2 * self.dr_g) * 4 * pi

    def step(self):
        self.diagonalize()
        self.calculate_density()
        self.calculate_electrostatic_potential()
        self.calculate_xc_potential()
        self.vr_sg = self.vxc_sg * self.r_g
        self.vr_sg += self.vHr_g
        self.vr_sg -= self.Z

    def run(self, mix=0.4, maxiter=117, dnmax=1e-9):
        if self.basis_l is None:
            self.initialize()

        dn = self.Z
        pb = ProgressBar(log(dnmax / dn), 0, 52, self.fd)
        self.log()
        
        for iter in range(maxiter):
            if iter > 1:
                self.vr_sg *= mix
                self.vr_sg += (1 - mix) * vr_old_sg
                dn = np.trapz(abs(self.n_sg - n_old_sg) *
                              self.r_g**2 * self.dr_g).sum() * 4 * pi
                if dn <= dnmax:
                    break

            pb(log(dnmax / dn))

            vr_old_sg = self.vr_sg
            n_old_sg = self.n_sg
            self.step()

        self.summary()
        if dn > dnmax:
            raise RuntimeError('Did not converge!')

    def summary(self):
        states = []
        for l, f_sn in self.f_lsn.items():
            for n, f in enumerate(f_sn[0]):
                states.append((self.eps_lsn[l][0][n], n, l, f))
        states.sort()
        
        self.log('\nstate  occupation             eigenvalue')
        self.log('====================================================')
        for e, n, l, f in states:
            self.log(' %d%s      %6.3f   %13.6f Ha  %13.5f eV' %
                     (n + l + 1, 'spdfg'[l], f, e, e * units.Hartree))
            if self.nspins == 2:
                f = self.f_lsn[l][1][n]
                e = self.eps_lsn[l][1][n]
                self.log('         %6.3f   %13.6f Ha  %13.5f eV' %
                     (f, e, e * units.Hartree))
        self.log('====================================================')
        self.log('\nEnergies:')
        self.log('==============================')
        self.log('kinetic       %+13.6f Ha' % self.ekin)
        self.log('coulomb (e-e) %+13.6f Ha' % self.eH)
        self.log('coulomb (e-n) %+13.6f Ha' % self.eZ)
        self.log('xc            %+13.6f Ha' % self.exc)
        self.log('total         %+13.6f Ha' %
                 (self.ekin + self.eH + self.eZ + self.exc))
        self.log('==============================')

    def g(self):
        import matplotlib.pyplot as plt
        for a in self.basis_l[0].alpha_B:
            plt.plot(self.x_g, np.exp(-a * self.r_g**2))
        plt.show()

    def plot(self, a_g, rc=4.0, show=False):
        import matplotlib.pyplot as plt
        mask_g = self.r_g < rc
        plt.plot(self.r_g[mask_g], a_g[mask_g])
        if show:
            plt.show()


def build_parser(): 
    from optparse import OptionParser

    parser = OptionParser(usage='%prog [options] element',
                          version='%prog 0.1')
    parser.add_option('-f', '--xc-functional', type='string', default='LDA',
                      help='Exchange-Correlation functional ' +
                      '(default value LDA)',
                      metavar='<XC>')
    parser.add_option('--add', metavar='states',
                      help='Add electron(s). Use "1s0.5a" to add 0.5 1s ' +
                      'electrons to the alpha-spin channel (use "b" for ' +
                      'beta-spin).  The number of electrons defaults to ' +
                      'one. Examples: "1s", "2p2b", "4f0.1b,3d-0.1a".')
    parser.add_option('-s', '--spin-polarized', action='store_true')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-e', '--exponents',
                      help='Exponents a: exp(-a*r^2).  Use "-e 0.1:20.0:30" ' +
                      'to get 30 exponents from 0.1 to 20.0.')
    return parser


def main():
    parser = build_parser()
    opt, args = parser.parse_args()

    if len(args) != 1:
        parser.error("incorrect number of arguments")
    symbol = args[0]

    nlfs = []
    if opt.add:
        for x in opt.add.split(','):
            n = int(x[0])
            l = 'spdfg'.find(x[1])
            x = x[2:]
            if x[-1] in 'ab':
                s = int(x[-1] == 'b')
                opt.spin_polarized = True
                x = x[:-1]
            else:
                s = None
            if x:
                f = float(x)
            else:
                f = 1
            nlfs.append((n, l, f, s))
            
    aea = AllElectronAtom(symbol,
                          xc=opt.xc_functional,
                          spinpol=opt.spin_polarized)

    if opt.exponents:
        parts = opt.exponents.split(':')
        kwargs['alpha1'] = float(parts[0])
        if len(parts) > 1:
            kwargs['alpha2'] = float(parts[1])
            if len(parts) > 2:
                kwargs['ngauss'] = int(parts[2])
        aea.initialize(**kwargs)

    for n, l, f, s in nlfs:
        aea.add(n, l, f, s)

    aea.run()
    

if __name__ == '__main__':
    main()
