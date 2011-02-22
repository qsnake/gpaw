#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, log

import numpy as np
from scipy.special import gamma
from scipy.interpolate import splrep, splev

from gpaw.atom.configurations import configurations
from gpaw.atom.aeatom import AllElectronAtom, Channel
from gpaw.setup import BaseSetup
from gpaw.spline import Spline
from gpaw.basis_data import Basis
from gpaw.hgh import null_xc_correction


class PAWWaves:
    def __init__(self, l, basis):
        self.l = l
        self.basis = basis
        self.phi_ng = []
        self.e_n = []
        self.f_n = []
        self.rc_n = []
        self.phit_ng = []
        self.pt_ng = []
        
    def add(self, phi_g, e, f, rc):
        self.phi_ng.append(phi_g)
        self.e_n.append(e)
        self.f_n.append(f)
        self.rc_n.append(rc)

        self.phit_ng.append(None)
        self.pt_ng.append(None)

    def solve(self, vtr_g, pt_g):
        A_bb = self.basis.T_bb + self.basis.calculate_potential_matrix(vtr_g)
        e = self.e_n[0]
        A_bb -= e * np.eye(len(A_bb))
        gd = self.basis.gd
        b_b = gd.integrate(self.basis.basis_bg * pt_g)
        c_b = np.linalg.solve(A_bb, b_b)
        phit_g = self.basis.expand(c_b)
        gc = gd.get_index(self.rc_n[0] + 0.1)
        a = self.phi_ng[0][gc] / phit_g[gc]
        b = 1 / a / np.dot(c_b, b_b) * 4 * pi
        phit_g *= a
        self.phit_ng[0] = phit_g
        self.pt_ng[0] = pt_g * b
        #gd.plot(phit_g)
        #gd.plot(self.phi_ng[0],show=1)
        self.nt_g = phit_g**2 / (4 * pi)
        self.ds = 1 - gd.integrate(self.nt_g)
        self.dh = e * self.ds - a / b
        self.Q = self.ds


class PAWSetupGenerator:
    def __init__(self, aea, waves, l0, rcmax=0):
        self.aea = aea
        self.l0 = l0
        self.gd = aea.gd
        
        if len(waves) == 0:
            lmax = -1
        else:
            lmax = max([l for n, l, rc in waves])
        self.waves_l = [PAWWaves(l, aea.channels[l].basis)
                       for l in range(lmax + 1)]

        self.rcmax = rcmax
        for n, l, rc in waves:
            ch = aea.channels[l]
            e = ch.e_n[n - l - 1]
            f = ch.f_n[n - l - 1]
            print n,l,rc,e,f
            phi_g = ch.basis.expand(ch.C_nb[0])
            self.waves_l[l].add(phi_g, e, f, rc)
            self.rcmax = max(rc, self.rcmax)
            
        self.gcmax = self.gd.get_index(self.rcmax)

        self.alpha = log(1.0e4) / self.rcmax**2  # exp(-alpha*rcmax^2)=1.0e-4
        self.alpha = round(self.alpha, 2)
        self.ghat_g = (np.exp(-self.alpha * self.gd.r_g**2) *
                       (self.alpha / pi)**1.5)

        f0 = 0.0
        if l0 == 0:
            f0 = 1.0
        self.zeropot = Channel(l0, 0, [f0], aea.channels[l0].basis)

        self.vtr_g = None

    def pseudize(self, a_g, gc, n=1):
        assert isinstance(gc, int) and gc > 10 and n == 1
        r_g = self.gd.r_g
        poly = np.polyfit([0, r_g[gc - 1], r_g[gc], r_g[gc + 1]],
                          [0, a_g[gc - 1], a_g[gc], a_g[gc + 1]], 3)
        at_g = a_g.copy()
        at_g[:gc] = np.polyval(poly, r_g[:gc])
        return at_g

    def generate(self):
        gd = self.gd
        
        self.vtr_g = self.pseudize(self.aea.vr_sg[0], self.gcmax)
        self.v0_g = 0.0

        ntold_g = 0.0
        while True:
            self.update()
            dn = self.gd.integrate(abs(self.nt_g - ntold_g))
            print dn
            if dn < 1.0e-7:
                break
            ntold_g = self.nt_g

    def update(self):
        self.nt_g = 0.0
        self.Q = -1.0

        self.find_zero_potential()

        for waves in self.waves_l:
            waves.solve(self.vtr_g, self.ghat_g)
            self.nt_g += waves.nt_g
            self.Q += waves.Q
            
        self.rhot_g = self.nt_g + self.Q * self.ghat_g
        self.vHtr_g = self.gd.poisson(self.rhot_g)
        self.vxct_g = self.gd.zeros()
        exct_g = self.gd.zeros()
        self.exct = self.aea.xc.calculate_spherical(
            self.gd, self.nt_g.reshape((1, -1)), self.vxct_g.reshape((1, -1)))
        self.vtr_g = self.vHtr_g + (self.vxct_g + self.v0_g) * self.gd.r_g
        
    def find_zero_potential(self):
        e0 = self.aea.channels[self.l0].e_n[0]
        dv0_g = self.ghat_g
        r_g = self.gd.r_g
        pot0 = self.zeropot
        V_bb = pot0.basis.calculate_potential_matrix(dv0_g * r_g)
        while True:
            pot0.solve(self.vtr_g)
            e = pot0.e_n[0]
            c_b = pot0.C_nb[0]
            if abs(e - e0) < 1.0e-8:
                break

            v = np.dot(np.dot(c_b, V_bb), c_b)
            a = (e0 - e) / v
            self.vtr_g += a * dv0_g * r_g
            self.v0_g += a * dv0_g

        if self.l0 == 0:
            self.nt_g += pot0.calculate_density()

    def plot(self):
        import matplotlib.pyplot as plt
        r_g = self.gd.r_g
        plt.plot(r_g, self.vxc_g, label='xc')
        plt.plot(r_g, self.v0_g, label='0')
        plt.plot(r_g[1:], self.vHtr_g[1:] / r_g[1:], label='H')
        plt.plot(r_g[1:], self.vtr_g[1:] / r_g[1:], label='ps')
        plt.plot(r_g[1:], self.aea.vr_sg[0, 1:] / r_g[1:], label='ae')
        plt.axis(xmax=2 * self.rcmax,
                 ymin=self.vtr_g[1] / r_g[1],
                 ymax=max(0, self.v0_g[0]))
        plt.legend()
        
        plt.figure()

        if self.l0 == 0:
            phit_g = self.zeropot.basis.expand(self.zeropot.C_nb[0])
            ch = self.aea.channels[0]
            phi_g = ch.basis.expand(ch.C_nb[0])
            plt.plot(r_g, phi_g * r_g)
            plt.plot(r_g, phit_g * r_g)

        for waves in self.waves_l:
            plt.plot(r_g, waves.phi_ng[0] * r_g)
            plt.plot(r_g, waves.phit_ng[0] * r_g)
            #plt.plot(r_g, waves.pt_ng[0] * r_g)
        plt.axis(xmax=2 * self.rcmax)
        plt.show()
        
    def make_paw_setup(self):
        phit_g = self.zeropot.basis.expand(self.zeropot.C_nb[0])
        return PAWSetup(self.alpha, self.gd.r_g, phit_g, self.v0_g)


class PAWSetup:
    def __init__(self, alpha, r_g, phit_g, v0_g):
        self.natoms = 0
        self.E = 0.0
        self.Z = 1
        self.Nc = 0
        self.Nv = 1
        self.niAO = 1
        self.pt_j = []
        self.ni = 0
        self.l_j = []
        self.nct = None
        self.Nct = 0.0

        rc = 1.0
        r2_g = np.linspace(0, rc, 100)**2
        x_g = np.exp(-alpha * r2_g)
        x_g[-1] = 0 

        self.ghat_l = [Spline(0, rc,
                              (4 * pi)**0.5 * (alpha / pi)**1.5 * x_g)]

        self.vbar = Spline(0, rc, (4 * pi)**0.5 * v0_g[0] * x_g)

        r = np.linspace(0, 4.0, 100)
        phit = splev(r, splrep(r_g, phit_g))
        poly = np.polyfit(r[[-30,-29,-2,-1]], [0, 0, phit[-2], phit[-1]], 3)
        phit[-30:] -= np.polyval(poly, r[-30:])
        self.phit_j = [Spline(0, 4.0, phit)]
                              
        self.Delta_pL = np.zeros((0, 1))
        self.Delta0 = -1 / (4 * pi)**0.5
        self.lmax = 0
        self.K_p = self.M_p = self.MB_p = np.zeros(0)
        self.M_pp = np.zeros((0, 0))
        self.Kc = 0.0
        self.MB = 0.0
        self.M = 0.0
        self.xc_correction = null_xc_correction
        self.HubU = None
        self.dO_ii = np.zeros((0, 0))
        self.type = 'local'
        self.fingerprint = None
        
    def get_basis_description(self):
        return '1s basis cut off at 4 Bohr'

    def print_info(self, text):
        text('Local pseudo potential')
        
    def calculate_initial_occupation_numbers(self, magmom, hund, charge,
                                             nspins):
        return np.array([(1.0,)])

    def initialize_density_matrix(self, f_si):
        return np.zeros((1, 0))

    def calculate_rotations(self, R_slmm):
        self.R_sii = np.zeros((1, 0, 0))

     
def build_parser(): 
    from optparse import OptionParser

    parser = OptionParser(usage='%prog [options] element',
                          version='%prog 0.1')
    parser.add_option('-f', '--xc-functional', type='string', default='LDA',
                      help='Exchange-Correlation functional ' +
                      '(default value LDA)',
                      metavar='<XC>')
    parser.add_option('-p', '--plot', action='store_true')
    return parser


def main(AEA=AllElectronAtom):
    parser = build_parser()
    opt, args = parser.parse_args()

    if len(args) != 1:
        parser.error('Incorrect number of arguments')
    symbol = args[0]

    kwargs = {'xc': opt.xc_functional}
        
    aea = AEA(symbol, **kwargs)
    aea.run()

    gen = Generator(aea)
    gen.generate()
    
    if opt.plot:
        gen.plot()

if __name__ == '__main__':
    #main()
    aea = AllElectronAtom('H')
    #aea.add(2, 1, 0)
    #aea.initialize()
    aea.run()
    #g = PAWSetupGenerator(aea, [(1, 0, 0.8)], 1)
    g = PAWSetupGenerator(aea, [], 0, 0.8)
    g.generate()
    #g.plot()
    setup = g.make_paw_setup()
    from ase.data.molecules import molecule
    from gpaw import GPAW

    a = molecule('H', pbc=1, magmoms=[0])
    a.center(vacuum=2)
    a.set_calculator(
        GPAW(setups={0: setup}))
    a.get_potential_energy()
