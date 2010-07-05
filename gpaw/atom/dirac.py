#!/usr/bin/env python
import sys
from math import pi

import numpy as np
from scipy.special import gamma
import ase.units as units

from gpaw.atom.aeatom import GaussianBasis, AllElectronAtom

# Velocity of light in atomic units:
c = 2 * units._hplanck / (units._mu0 * units._c * units._e**2)


class GaussianDiracBasis:
    def __init__(self, basis):
        """Guassian basis set for atomic Dirac equation."""
        self.basis = basis
        self.nbasis = basis.nbasis
        self.T_BB = basis.T_BB
        
        alpha_B = basis.alpha_B
        l = self.l = basis.l
        
        A_BB = np.add.outer(alpha_B, alpha_B)
        M_BB = np.multiply.outer(alpha_B, alpha_B)

        # Derivative matrix:
        D_BB = 2**(l + 2.5) * M_BB**(0.5 * l + 0.75) / gamma(l + 1.5) * (
            0.5 * (l + 1) * gamma(l + 1) / A_BB**(l + 1) -
            gamma(l + 2) * alpha_B / A_BB**(l + 2))
        self.D_bb = np.dot(np.dot(basis.Q_Bb.T, D_BB), basis.Q_Bb)

        # 1/r matrix:
        K_BB = 2**(l + 2.5) * M_BB**(0.5 * l + 0.75) / gamma(l + 1.5) * (
            0.5 * gamma(l + 1) / A_BB**(l + 1))
        self.K_bb = np.dot(np.dot(basis.Q_Bb.T, K_BB), basis.Q_Bb)

    def diagonalize(self, k, vr2dr_g):
        """Solve Dirac equation in basis set.

        Returns eigenvalues and eigenvectors."""

        basis = self.basis
        nbasis = self.nbasis
        l = basis.l

        V_bb = np.inner(basis.basis_bg[:, 1:],
                        basis.basis_bg[:, 1:] * vr2dr_g[1:])
        
        H_bb = np.zeros((2 * nbasis, 2 * nbasis))
        H_bb[:nbasis, :nbasis] = V_bb
        H_bb[nbasis:, nbasis:] = V_bb - 2 * c**2 * np.eye(nbasis)
        H_bb[nbasis:, :nbasis] = -c * (-self.D_bb.T + k * self.K_bb)
        eps_n, C_bn = np.linalg.eigh(H_bb)
        if k < 0:
            return eps_n[nbasis:], C_bn[:, nbasis:].copy()
        else:
            return eps_n[nbasis + 1:], C_bn[:, nbasis + 1:].copy()
    
    def calculate_density(self, f_n, C_bn):
        """Calculate density and density matrix."""
        return (self.basis.calculate_density(f_n, C_bn[:self.nbasis]) +
                self.basis.calculate_density(f_n, C_bn[self.nbasis:]))


class AllElectronDiracAtom(AllElectronAtom):
    def __init__(self, symbol, xc='LDA', log=sys.stdout):
        """All-electron calculation for spherically symmetric atom.

        symbol: str
            Chemical symbol.
        xc: str
            Name of XC-functional.
        log: stream
            Text output."""

        AllElectronAtom.__init__(self, symbol, xc, spinpol=False, log=log)

    def get_basis(self, l, eps):
        l = ((l + 1)**2 - (self.Z / c)**2)**0.5 - 1  # relativistic correction
        return GaussianDiracBasis(AllElectronAtom.get_basis(self, l, eps))

    def initialize_configuration(self):
        AllElectronAtom.initialize_configuration(self)
        self.f_kn = {}
        for l, f_sn in self.f_lsn.items():
            for k in [-l - 1, l]:
                if k == 0:
                    continue
                j = abs(k) - 0.5 
                self.f_kn[k] = (2 * j + 1) / (4 * l + 2) * np.array(f_sn[0])
        
    def diagonalize(self):
        """Diagonalize Dirac equation."""
        self.eps_kn = {}
        self.C_kbn = {}
        self.eeig = 0.0
        vr2dr_g = self.r_g * self.vr_sg[0] * self.dr_g
        for k, f_n in self.f_kn.items():
            basis = self.basis_l[abs(k) - 1]
            self.eps_kn[k], self.C_kbn[k] = basis.diagonalize(k, vr2dr_g)
            self.eeig += np.dot(f_n, self.eps_kn[k][:len(f_n)])
        
    def calculate_density(self):
        """Calculate elctron density and kinetic energy."""
        self.n_sg = np.zeros((1, self.ngpts))
        for k, C_bn in self.C_kbn.items():
            basis = self.basis_l[abs(k) - 1]
            self.n_sg[0] += basis.calculate_density(self.f_kn[k], C_bn)
        self.n_sg[:, 0] = self.n_sg[:, 1]

    def sort_states(self):
        states = []
        for k, eps_n in self.eps_kn.items():
            l = (abs(2 * k + 1) - 1) // 2
            j = abs(k) - 0.5
            for n, f in enumerate(self.f_kn[k]):
                states.append((eps_n[n], n, l, j, k, f))
        states.sort()
        return states
    
    def write_states(self):
        self.log('\n  state   occupation             eigenvalue')
        self.log('======================================================')
        for e, n, l, j, k, f in self.sort_states():
            self.log(' %d%s(%d/2)   %6.3f   %13.6f Ha  %13.5f eV' %
                     (n + l + 1, 'spdfg'[l], 2 * j, f, e, e * units.Hartree))
        self.log('======================================================')

    def get_wave_functions(self, k, n, force_positive_tail=True):
        l = (abs(2 * k + 1) - 1) // 2
        C_pb = self.C_kbn[k][:, n - l - 1].reshape((2, -1))
        basis_bg = self.basis_l[abs(k) - 1].basis.basis_bg
        f_g = np.dot(C_pb[0], basis_bg)
        g_g = np.dot(C_pb[1], basis_bg)        
        if force_positive_tail:
            g = int(self.ngpts * 5.0 / (self.beta + 5.0))
            if f_g[g] < 0:
                f_g *= -1.0
            if g_g[g] < 0:
                g_g *= -1.0
        return f_g, g_g
        
    def plot_wave_functions(self, rc=4.0):
        import matplotlib.pyplot as plt
        if rc is None:
            r_g = self.x_g
        else:
            r_g = self.r_g
        colors = 'krgbycm' * 5
        for e, n, l, j, k, f in self.sort_states():
            n += l + 1
            f_g, g_g = self.get_wave_functions(k, n)
            if j > l:
                ls = '-'
            else:
                ls = '--'
            plt.plot(r_g, f_g * self.r_g,
                     lw=2, color=colors[0], ls=ls,
                     label='%d%s(%d/2)' % (n, 'spdfg'[l], 2 * j))
            plt.plot(r_g, g_g * self.r_g,
                     lw=1, color=colors[0], ls=ls)
            colors = colors[1:]
        plt.legend()
        if rc is not None:
            plt.axis(xmax=rc)
        plt.show()


if __name__ == '__main__':
    from gpaw.atom.aeatom import main
    main(AEA=AllElectronDiracAtom)
    
