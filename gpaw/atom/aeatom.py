#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import pi, log

import numpy as np
from scipy.special import gamma
from ase.data import atomic_numbers, atomic_names, chemical_symbols
import ase.units as units
from ase.utils import devnull

from gpaw.atom.configurations import configurations
from gpaw.xc_functional import XCFunctional
from gpaw.utilities.progressbar import ProgressBar

# Velocity of light in atomic units:
c = 2 * units._hplanck / (units._mu0 * units._c * units._e**2)


class GridDescriptor:
    def __init__(self, r1, rN=50.0, N=1000):
        """Grid descriptor for radial grid.

        The radial grid is::

                     a g
            r(g) = -------,  g = 0, 1, ..., N - 1
                   1 - b g
        
        so that r(0)=0, r(1)=r1 and r(N)=rN."""

        self.N = N
        self.a = (1 - 1.0 / N) / (1.0 / r1 - 1.0 / rN)
        self.b = 1.0 - self.a / r1
        g_g = np.arange(N)
        self.r_g = self.a * g_g / (1 - self.b * g_g)
        self.dr_g = (self.b * self.r_g + self.a)**2 / self.a

    def get_index(self, r):
        return int(1 / (self.b + self.a / r) + 0.5)

    def zeros(self, x=()):
        if isinstance(x, int):
            x = (x,)
        return np.zeros(x + (self.N,))

    def integrate(self, a_xg, n=0):
        assert n > -2
        return np.dot(a_xg[..., 1:],
                      (self.r_g**(2 + n) * self.dr_g)[1:]) * (4 * pi)

    def poisson(self, n_g):
        a_g = -4 * pi * n_g * self.r_g * self.dr_g
        A_g = np.add.accumulate(a_g)
        vr_g = self.zeros()
        vr_g[1:] = A_g[:-1] + 0.5 * a_g[1:]
        vr_g -= A_g[-1]
        vr_g *= self.r_g
        a_g *= self.r_g
        A_g = np.add.accumulate(a_g)
        vr_g[1:] -= A_g[:-1] + 0.5 * a_g[1:]
        return vr_g

    def plot(self, a_g, n=0, rc=4.0, show=False):
        import matplotlib.pyplot as plt
        gc = self.get_index(rc)
        plt.plot(self.r_g, a_g * self.r_g**n)
        plt.axis(xmax=rc)
        if show:
            plt.show()


class GaussianBasis:
    def __init__(self, l, alpha_B, gd, eps=1.0e-7):
        """Guassian basis set for spherically symmetric atom.

        l: int
            Angular momentum quantum number.
        alpha_B: ndarray
            Exponents.
        gd: GridDescriptor
            Grid descriptor.
        eps: float
            Cutoff for eigenvalues of overlap matrix."""
        
        self.l = l
        self.alpha_B = alpha_B
        self.gd = gd

        A_BB = np.add.outer(alpha_B, alpha_B)
        M_BB = np.multiply.outer(alpha_B, alpha_B)

        # Overlap matrix:
        S_BB = (2 * M_BB**0.5 / A_BB)**(l + 1.5)

        # Kinetic energy matrix:
        T_BB = 2**(l + 2.5) * M_BB**(0.5 * l + 0.75) / gamma(l + 1.5) * (
            gamma(l + 2.5) * M_BB / A_BB**(l + 2.5) -
            0.5 * (l + 1) * gamma(l + 1.5) / A_BB**(l + 0.5) +
            0.25 * (l + 1) * (2 * l + 1) * gamma(l + 0.5) / A_BB**(l + 0.5))

        # Derivative matrix:
        D_BB = 2**(l + 2.5) * M_BB**(0.5 * l + 0.75) / gamma(l + 1.5) * (
            0.5 * (l + 1) * gamma(l + 1) / A_BB**(l + 1) -
            gamma(l + 2) * alpha_B / A_BB**(l + 2))

        # 1/r matrix:
        K_BB = 2**(l + 2.5) * M_BB**(0.5 * l + 0.75) / gamma(l + 1.5) * (
            0.5 * gamma(l + 1) / A_BB**(l + 1))

        # Find set of liearly independent functions.
        # We have len(alpha_B) gaussians (index B) and self.nbasis
        # linearly independent functions (index b).
        s_B, U_BB = np.linalg.eigh(S_BB)
        self.nbasis = int((s_B > eps).sum())

        Q_Bb = np.dot(U_BB[:, -self.nbasis:],
                      np.diag(s_B[-self.nbasis:]**-0.5))

        self.T_bb = np.dot(np.dot(Q_Bb.T, T_BB), Q_Bb)
        self.D_bb = np.dot(np.dot(Q_Bb.T, D_BB), Q_Bb)
        self.K_bb = np.dot(np.dot(Q_Bb.T, K_BB), Q_Bb)

        r_g = gd.r_g
        self.basis_bg = (np.dot(
                Q_Bb.T,
                (2 * (2 * alpha_B[:, np.newaxis])**(l + 1.5) /
                 gamma(l + 1.5))**0.5 *
                np.exp(-np.multiply.outer(alpha_B, r_g**2))) * r_g**l)

    def __len__(self):
        return self.nbasis

    def expand(self, C_xb):
        return np.dot(C_xb, self.basis_bg)

    def calculate_potential_matrix(self, vr_g):
        vr2dr_g = vr_g * self.gd.r_g * self.gd.dr_g
        V_bb = np.inner(self.basis_bg[:, 1:],
                        self.basis_bg[:, 1:] * vr2dr_g[1:])
        return V_bb


class Channel:
    def __init__(self, l, s, f_n, basis):
        self.l = l
        self.s = s
        self.basis = basis

        self.C_nb = None                       # eigenvectors
        self.e_n = None                        # eigenvalues
        self.f_n = np.array(f_n, dtype=float)  # occupation numbers
        
        self.name = 'spdf'[l]

    def solve(self, vr_g):
        """Diagonalize Schrödinger equation in basis set."""
        H_bb = self.basis.calculate_potential_matrix(vr_g)
        H_bb += self.basis.T_bb
        self.e_n, C_bn = np.linalg.eigh(H_bb)
        self.C_nb = C_bn.T
    
    def calculate_density(self, n=None):
        """Calculate density."""
        if n is None:
            n_g = 0.0
            for n, f in enumerate(self.f_n):
                n_g += f * self.calculate_density(n)
        else:
            n_g = self.basis.expand(self.C_nb[n])**2 / (4 * pi)
        return n_g

    def get_eigenvalue_sum(self):
        f_n = self.f_n
        return np.dot(f_n, self.e_n[:len(f_n)])


class DiracChannel(Channel):
    def __init__(self, k, f_n, basis):
        l = (abs(2 * k + 1) - 1) // 2
        Channel.__init__(self, l, 0, f_n, basis)
        self.k = k
        self.j = abs(k) - 0.5
        self.c_nb = None  # eigenvectors (small component)

        self.name += '(%d/2)' % (2 * self.j)

    def solve(self, vr_g):
        """Solve Dirac equation in basis set."""
        nb = len(self.basis)
        V_bb = self.basis.calculate_potential_matrix(vr_g)
        H_bb = np.zeros((2 * nb, 2 * nb))
        H_bb[:nb, :nb] = V_bb
        H_bb[nb:, nb:] = V_bb - 2 * c**2 * np.eye(nb)
        H_bb[nb:, :nb] = -c * (-self.basis.D_bb.T + self.k * self.basis.K_bb)
        e_n, C_bn = np.linalg.eigh(H_bb)
        if self.k < 0:
            n0 = nb
        else:
            n0 = nb + 1
        self.e_n = e_n[n0:].copy()
        self.C_nb = C_bn[:nb, n0:].T.copy()  # large component
        self.c_nb = C_bn[nb:, n0:].T.copy()  # small component

    def calculate_density(self, n=None):
        """Calculate density."""
        if n is None:
            n_g = Channel.calculate_density(self)
        else:
            n_g = (self.basis.expand(self.C_nb[n])**2 +
                   self.basis.expand(self.c_nb[n])**2) / (4 * pi)
            if self.basis.l < 0:
                n_g[0] = n_g[1]
        return n_g

        
class AllElectronAtom:
    def __init__(self, symbol, xc='LDA', spinpol=False, dirac=False,
                 log=sys.stdout):
        """All-electron calculation for spherically symmetric atom.

        symbol: str (or int)
            Chemical symbol (or atomic number).
        xc: str
            Name of XC-functional.
        spinpol: bool
            If true, do spin-polarized calculation.  Default is spin-paired.
        dirac: bool
            Solve Dirac equation instead of Schrödinger equation.
        log: stream
            Text output."""

        if isinstance(symbol, int):
            symbol = chemical_symbols[symbol]
        self.symbol = symbol
        self.Z = atomic_numbers[symbol]

        self.nspins = 1 + int(bool(spinpol))

        self.dirac = bool(dirac)

        if isinstance(xc, str):
            self.xc = XCFunctional(xc, nspins=self.nspins)
        else:
            self.xc = xc

        if log is None:
            log = devnull
        self.fd = log

        self.vr_sg = None  # potential * r
        self.n_sg = 0.0    # density
        self.gd = None     # radial grid descriptor

        # Energies:
        self.ekin = None
        self.eeig = None
        self.eH = None
        self.eZ = None

        self.channels = None

        self.initialize_configuration()

        self.log('Z:              ', self.Z)
        self.log('Name:           ', atomic_names[self.Z])
        self.log('Symbol:         ', symbol)
        self.log('XC-functional:  ', self.xc.xcname)
        self.log('Equation:       ', ['Schrödinger', 'Dirac'][self.dirac])

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

    def initialize(self, ngpts=1000, rcut=50.0,
                   alpha1=0.01, alpha2=None, ngauss=50,
                   eps=1.0e-7):
        """Initialize basis sets and radial grid.

        ngpts: int
            Number of grid points for radial grid.
        rcut: float
            Cutoff for radial grid.
        alpha1: float
            Smallest exponent for gaussian.
        alpha2: float
            Largest exponent for gaussian.
        ngauss: int
            Number of gaussians.
        eps: float
            Cutoff for eigenvalues of overlap matrix."""

        if alpha2 is None:
            alpha2 = 50.0 * self.Z**2

        self.gd = GridDescriptor(r1=1 / alpha2**0.5 / 50, rN=rcut, N=ngpts)
        self.log('Grid points:     %d (%.5f, %.5f, %.5f, ..., %.3f, %.3f)' %
                 ((self.gd.N,) + tuple(self.gd.r_g[[0, 1, 2, -2, -1]])))

        # Distribute exponents between alpha1 and alpha2:
        alpha_B = alpha1 * (alpha2 / alpha1)**np.linspace(0, 1, ngauss)
        self.log('Exponents:       %d (%.3f, %.3f, ..., %.3f, %.3f)' %
                 ((ngauss,) + tuple(alpha_B[[0, 1, -2, -1]])))

        # Maximum l value:
        lmax = max(self.f_lsn.keys())

        self.channels = []
        nb_l = []
        if not self.dirac:
            for l in range(lmax + 1):
                basis = GaussianBasis(l, alpha_B, self.gd, eps)
                nb_l.append(len(basis))
                for s in range(self.nspins):
                    self.channels.append(Channel(l, s, self.f_lsn[l][s], basis))
        else:
            for K in range(1, lmax + 2):
                leff = (K**2 - (self.Z / c)**2)**0.5 - 1
                basis = GaussianBasis(leff, alpha_B, self.gd, eps)
                nb_l.append(len(basis))
                for k, l in [(-K, K - 1), (K, K)]:
                    if l > lmax:
                        continue
                    f_n = self.f_lsn[l][0]
                    j = abs(k) - 0.5
                    f_n = (2 * j + 1) / (4 * l + 2) * np.array(f_n)
                    self.channels.append(DiracChannel(k, f_n, basis))

        self.log('Basis functions: %s (%s)' %
                 (', '.join([str(nb) for nb in nb_l]),
                  ', '.join('spdf'[:lmax + 1])))

        self.vr_sg = self.gd.zeros(self.nspins)
        self.vr_sg[:] = -self.Z

    def solve(self):
        """Diagonalize Schrödinger equation."""
        self.eeig = 0.0
        for channel in self.channels:
            channel.solve(self.vr_sg[channel.s])
            self.eeig += channel.get_eigenvalue_sum()

    def calculate_density(self):
        """Calculate elctron density and kinetic energy."""
        self.n_sg = self.gd.zeros(self.nspins)
        for channel in self.channels:
            self.n_sg[channel.s] += channel.calculate_density()

    def calculate_electrostatic_potential(self):
        """Calculate electrostatic potential and energy."""
        n_g = self.n_sg.sum(0)
        self.vHr_g = self.gd.poisson(n_g)        
        self.eH = 0.5 * self.gd.integrate(n_g * self.vHr_g, -1)
        self.eZ = -self.Z * self.gd.integrate(n_g, -1)
        
    def calculate_xc_potential(self):
        self.vxc_sg = self.gd.zeros(self.nspins)
        exc_g = self.gd.zeros()
        if self.nspins == 1:
            self.xc.calculate_spinpaired(exc_g, self.n_sg[0], self.vxc_sg[0])
        else:
            self.xc.calculate_spinpolarized(exc_g,
                                            self.n_sg[0], self.vxc_sg[0],
                                            self.n_sg[1], self.vxc_sg[1])
        exc_g[-1] = 0.0
        self.exc = self.gd.integrate(exc_g)
        self.exc_g = exc_g

    def step(self):
        self.solve()
        self.calculate_density()
        self.calculate_electrostatic_potential()
        self.calculate_xc_potential()
        self.vr_sg = self.vxc_sg * self.gd.r_g
        self.vr_sg += self.vHr_g
        self.vr_sg -= self.Z
        self.ekin = (self.eeig -
                     self.gd.integrate((self.vr_sg * self.n_sg).sum(0), -1))
        
    def run(self, mix=0.4, maxiter=117, dnmax=1e-9):
        if self.channels is None:
            self.initialize()

        dn = self.Z
        pb = ProgressBar(log(dnmax / dn), 0, 53, self.fd)
        self.log()
        
        for iter in range(maxiter):
            if iter > 1:
                self.vr_sg *= mix
                self.vr_sg += (1 - mix) * vr_old_sg
                dn = self.gd.integrate(abs(self.n_sg - n_old_sg).sum(0))
                pb(log(dnmax / dn))
                if dn <= dnmax:
                    break

            vr_old_sg = self.vr_sg
            n_old_sg = self.n_sg
            self.step()

        self.summary()
        if dn > dnmax:
            raise RuntimeError('Did not converge!')

    def summary(self):
        self.write_states()
        self.write_energies()

    def write_states(self):
        self.log('\n state  occupation         eigenvalue          <r>')
        if self.dirac:
            self.log(' nl(j)               [Hartree]        [eV]    [Bohr]')
        else:
            self.log(' nl                  [Hartree]        [eV]    [Bohr]')
        self.log('=====================================================')
        states = []
        for ch in self.channels:
            for n, f in enumerate(ch.f_n):
                states.append((ch.e_n[n], ch, n))
        states.sort()
        for e, ch, n in states:
            name = str(n + ch.l + 1) + ch.name
            if self.nspins == 2:
                name += '(%s)' % '+-'[ch.s]    
            n_g = ch.calculate_density(n)
            rave = self.gd.integrate(n_g, 1)
            self.log(' %-7s  %6.3f %13.6f  %13.5f %6.3f' %
                     (name, ch.f_n[n], e, e * units.Hartree, rave))
        self.log('=====================================================')

    def write_energies(self):
        self.log('\nEnergies [Hartree]:')
        self.log('=============================')
        self.log(' kinetic       %+13.6f' % self.ekin)
        self.log(' coulomb (e-e) %+13.6f' % self.eH)
        self.log(' coulomb (e-n) %+13.6f' % self.eZ)
        self.log(' xc            %+13.6f' % self.exc)
        self.log(' total         %+13.6f' %
                 (self.ekin + self.eH + self.eZ + self.exc))
        self.log('=============================')

    def get_channel(self, l=None, s=0, k=None):
        if self.dirac:
            for channel in self.channels:
                if channel.k == k:
                    return channel
        else:
            for channel in self.channels:
                if channel.l == l and channel.s == s:
                    return channel
        raise ValueError

    def get_orbital(self, n, l=None, s=0, k=None):
        channel = self.get_channel(l, s, k)
        return channel.basis.expand(channel.C_nb[n])

    def plot_wave_functions(self, rc=4.0):
        import matplotlib.pyplot as plt
        colors = 'krgbycm'
        for ch in self.channels:
            for n in range(len(ch.f_n)):
                fr_g = ch.basis.expand(ch.C_nb[n]) * self.gd.r_g
                name = str(n + ch.l + 1) + ch.name
                lw = 2
                if self.nspins == 2:
                    name += '(%s)' % '+-'[ch.s]    
                    if ch.s == 1:
                        lw = 1
                if self.dirac and ch.k > 0:
                    lw = 1
                ls = ['-', '--', '-.', ':'][ch.l]
                n_g = ch.calculate_density(n)
                rave = self.gd.integrate(n_g, 1)
                gave = self.gd.get_index(rave)
                fr_g *= cmp(fr_g[gave], 0)
                plt.plot(self.gd.r_g, fr_g,
                         ls=ls, lw=lw, color=colors[n + ch.l], label=name)
        plt.legend(loc='best')
        plt.axis(xmax=rc)
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
    parser.add_option('-d', '--dirac', action='store_true')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-e', '--exponents',
                      help='Exponents a: exp(-a*r^2).  Use "-e 0.1:20.0:30" ' +
                      'to get 30 exponents from 0.1 to 20.0.')
    return parser


def main():
    parser = build_parser()
    opt, args = parser.parse_args()

    if len(args) != 1:
        parser.error('Incorrect number of arguments')
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
                          spinpol=opt.spin_polarized,
                          dirac=opt.dirac)

    if opt.exponents:
        parts = opt.exponents.split(':')
        kwargs = {}
        kwargs['alpha1'] = float(parts[0])
        if len(parts) > 1:
            kwargs['alpha2'] = float(parts[1])
            if len(parts) > 2:
                kwargs['ngauss'] = int(parts[2])
        aea.initialize(**kwargs)

    for n, l, f, s in nlfs:
        aea.add(n, l, f, s)

    aea.run()

    if opt.plot:
        aea.plot_wave_functions()


if __name__ == '__main__':
    main()
