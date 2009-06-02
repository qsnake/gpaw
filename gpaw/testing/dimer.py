#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Emacs: treat this as -*- python -*-

from optparse import OptionParser


parser = OptionParser(usage='%prog [options] element[s]',
                      version='%prog 0.1')
parser.add_option('-s', '--summary', action='store_true',
                  default=False,
                  help='Do a summary.')
parser.add_option('--lcao', type='string', metavar='basis',
                  help='Do a LCAO calculation.')

opt, args = parser.parse_args()

import os
import sys
import pickle
import tempfile
from math import sqrt

import numpy as npy
from ase.atoms import Atoms
from ase.data import atomic_names, covalent_radii, atomic_numbers

from gpaw import GPAW, ConvergenceError

b0 = {'Ni': 2.150, 'Pd': 2.496,
      'Pt': 2.35,
      'Ru': 2.122, 'Na': 3.300,
      'Nb': 1.99,
      'Mg': 5.0, 'Li': 3.270, 'Pb': 2.880, 'Rb': 4.375, 'Ti': 2.05, 'Rh': 2.232, 'Be': 2.55, 'Ba': 5.0, 'La': 2.968, 'Si': 2.220, 'As': 2.124, 'Fe': 1.843, 'Sr': 5.0, 'Mo': 1.8, 'C': 1.290, 'B': 1.710, 'F': 1.413, 'H': 0.753, 'K': 4.107, 'Mn': 1.6, 'O': 1.240, 'P': 1.880, 'S': 1.900, 'V': 1.6, 'N': 1.110, 'Zn': 3.5, 'Ag': 2.650, 'Cl': 1.990, 'Ca': 4.5, 'Ir': 2.2, 'Al': 2.824, 'Cd': 4.0, 'Ge': 2.390, 'Au': 2.570, 'Zr': 2.3, 'Ga': 2.840, 'Cs': 4.850, 'Cr': 1.5, 'Cu': 2.277, 'Se': 2.1}
b0 = {'Ni': 2.143, 'Pd': 2.485, 'Pt': 2.373, 'Ru': 2.125, 'Na': 3.289, 'Nb': 2.005, 'Mg': 4.0, 'Li': 2.8, 'Pb': 2.873, 'Rb': 4.360, 'Ti': 2.055, 'Rh': 2.231, 'Ta': 2.2, 'Be': 2.618, 'Ba': 4.871, 'La': 2.872, 'Si': 2.218, 'As': 2.071, 'Fe': 1.837, 'Br': 2.281, 'He': 1.972, 'C': 1.279, 'B': 1.694, 'F': 1.413, 'H': 0.753, 'K': 4.108, 'Mn': 1.665, 'O': 1.234, 'Ne': 1.976, 'P': 1.878, 'S': 1.893, 'Kr': 4.3, 'W': 2.1, 'V': 1.672, 'N': 1.102, 'Se': 2.154, 'Zn': 2.8, 'Co': 2.0, 'Ag': 2.626, 'Cl': 1.989, 'Ca': 2.805, 'Ir': 2.227, 'Al': 2.868, 'Cd': 3.0, 'Ge': 2.319, 'Ar': 2.589, 'Au': 2.555, 'Zr': 2.385, 'Ga': 2.837, 'Cs': 4.819, 'Cu': 2.281, 'Cr': 1.8, 'Mo': 1.9, 'Sr': 2.7}
colors = ['black', 'brown', 'red', 'orange',
          'yellow', 'green', 'blue', 'violet', 'gray', 'gray', 'gray', 'gray',
          'gray', 'gray', 'gray', 'gray']

class TestAtom:
    def __init__(self, symbol):
        self.parameters = {}
        self.name = symbol

        if '.' in symbol:
            symbol, setupname = symbol.split('.')
            self.parameters['setups'] = setupname
            
        self.symbol = symbol

        if not os.path.isdir(self.name):
            os.mkdir(self.name)

        self.Z = atomic_numbers[symbol]
        r = covalent_radii[self.Z]
        
        if symbol in b0:
            self.d0 = b0[symbol]
        else:
            self.d0 = 2 * r

        if symbol in ['He', 'Ne', 'Ar', 'Kr']:
            self.a = round(2 / sqrt(3) * self.d0 / 0.2 / 4) * 4 * 0.2
        else:
            self.a = round(max(2.5 * self.d0, 5.5) / 0.2 / 4) * 4 * 0.2

        gmin = 4 * int(self.a / 0.30 / 4 + 0.5)
        gmax = 4 * int(self.a / 0.14 / 4 + 0.5)
        
        self.ng = (gmax + 4 - gmin) // 4

        self.h = self.a / npy.arange(gmin, gmax + 4, 4)

    def run(self, summary, lcao):
        self.lcao = lcao
        tasks = ['eggbox', 'dimer']
        self.ready = True
        for task in tasks:
            filename = self.name + '/' + task + '.pckl'
            if os.path.isfile(filename):
                data = pickle.load(open(filename))
                if data:
                    for name, value in data.items():
                        setattr(self, name, value)
                else:
                    print 'Skipping', filename
                    self.ready = False
            else:
                self.ready = False
                if not summary:
                    self.pickle(task)
                    print 'Running', task, 'part for', self.name
                    getattr(self, task)()

    def pickle(self, task, attrs=None):
        if attrs is None:
            attrs = []
        else:
            attrs += ['d0', 'a', 'ng', 'h']
        data = {}
        for attr in attrs:
            data[attr] = getattr(self, attr)
        pickle.dump(data, open(self.name + '/' + task + '.pckl', 'w'))
        
    def eggbox(self):
        atom = Atoms(self.symbol, pbc=True, cell=(self.a, self.a, self.a))

        negg = 25
        self.Eegg = npy.zeros((self.ng, negg))
        self.Fegg = npy.zeros((self.ng, negg))
        
        for i in range(self.ng):
            h = self.h[i]
            print '%.3f:' % h,
            calc = GPAW(h=h, width=0.1, xc='PBE',
                        txt='%s/eggbox-%.3f.txt' % (self.name, h),
                        **self.parameters)
            if self.lcao:
                calc.set(mode='lcao', basis=self.lcao)

            atom.set_calculator(calc)

            for j in range(negg):
                x = h * j / (2 * negg - 2)
                atom[0].x = x
                try:
                    self.Eegg[i, j] = calc.get_potential_energy(atom,
                        force_consistent=True)
                except ConvergenceError:
                    sys.stdout.write('E')
                    break
                self.Fegg[i, j] = atom.get_forces()[0, 0]
                sys.stdout.write('.')
                sys.stdout.flush()
            print

        self.pickle('eggbox', ['Eegg', 'Fegg'])

    def dimer(self):
        dimer = Atoms([self.symbol, self.symbol],
                      pbc=True, cell=(self.a, self.a, self.a))

        self.Edimer = npy.zeros((self.ng, 7))
        self.Fdimer = npy.zeros((self.ng, 7, 2))
        
        q0 = self.d0 / sqrt(3)
        for i in range(self.ng):
            h = self.h[i]
            print '%.3f:' % h,
            calc = GPAW(h=h, width=0.1, xc='PBE',
                        txt='%s/dimer-%.3f.txt' % (self.name, h),
                        **self.parameters)
            if self.lcao:
                calc.set(mode='lcao', basis=self.lcao)

            dimer.set_calculator(calc)

            y = []
            for j in range(-3, 4):
                q = q0 * (1 + j * 0.02)
                dimer.positions[1] = (q, q, q)
                try:
                    self.Edimer[i, j + 3] = calc.get_potential_energy(dimer,
                        force_consistent=True)
                except ConvergenceError:
                    sys.stdout.write('E')
                    break
                self.Fdimer[i, j + 3] = dimer.get_forces()[:, 0]
                sys.stdout.write('.')
                sys.stdout.flush()
            print

        self.pickle('dimer', ['Edimer', 'Fdimer'])

    def summary(self, show=True):
        dd = 0.02 * self.d0
        d = self.d0 + dd * npy.arange(-3, 4)
        M = npy.array([(1, 0, 0, 0),
                      (1, 1, 1, 1),
                      (0, 1, 0, 0),
                      (0, 1, 2, 3)])

        Edimer0 = npy.empty(self.ng)
        ddimer0 = npy.empty(self.ng)
        Eegg = self.Eegg
        Edimer = self.Edimer
        h = self.h
        for i in range(self.ng):
            E = Edimer[i]
            ia = E.argsort()[0]
            E0 = E[ia]
            d0 = d[ia]
            energy = npy.polyfit(d**-1, E, 3)
            der = npy.polyder(energy, 1)
            roots = npy.roots(der)
            if isinstance(roots[0], float):
                der2 = npy.polyder(der, 1)
                if npy.polyval(der2, roots[0]) > 0:
                    root = roots[0]
                else:
                    root = roots[1]
                d0 = 1.0 / root
                E0 = npy.polyval(energy, root)
            Edimer0[i] = E0
            ddimer0[i] = d0
        
        for j in range(self.ng):
            if abs(h[j] - 0.2) < 0.00001:
                break
        #print '%r: %.3f,' % (self.symbol, d0),
        Ediss = 2 * Eegg[:, 0] - Edimer0
        
        B, C = fit(h[j:], Eegg[j:, 0])
        
        if 1:
            print ('%-2s %8.5f %8.5f %8.5f %8.6f %8.5f %8.5f %6.3f %6.4f %7.3f %10.2f %7.2f' %
                   (self.name,
                    Edimer0[j:].ptp(),
                    Ediss[j:].ptp(),
                    Eegg.ptp(axis=1)[j:].max(),
                    ddimer0[j:].ptp(),
                    npy.abs(self.Fegg[j:]).max(1).max(),
                    npy.abs(self.Fdimer[j:].sum(2)).max(1).max(),
                    Ediss[-1], d0,
                    d0 - self.d0,B,C
                   ))
        if 1:
            import pylab as plt
            dpi = 80
            fig = plt.figure(figsize=(8, 12), dpi=dpi)
            fig.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.04,
                                wspace=0.25, hspace=0.25)

            plt.subplot(321)
            for i in range(self.ng):
                x, y = f(d, Edimer[i] - 2 * Eegg[i, 0],
                         -self.Fdimer[i, :, 1] * sqrt(3))
                plt.plot(x, y, color=colors[i], label=u'h=%.2f Å' % h[i])
            plt.title('Dimer')
            plt.xlabel(u'bond length [Å]')
            plt.ylabel(r'$\Delta E=E_d-2E_a\ (\rm{eV})$')
            plt.legend(loc='best')

            plt.subplot(322)
            plt.plot(h, 100 * ddimer0 / ddimer0[-1] - 100, '-o')
            plt.title('Bond length')
            plt.xlabel(u'h [Å]')
            plt.ylabel(u'bond length error [%]')
                
            plt.subplot(323)
            for i in range(self.ng):
                x, y = f(npy.linspace(0, 0.5 * h[i], 25),
                         Eegg[i] - Eegg[i, 0],
                         -self.Fegg[i])
                plt.plot(x, 1000 * y, color=colors[i])
            plt.axis('tight')
            plt.title('Eggbox error')
            plt.xlabel(u'displacement [Å]')
            plt.ylabel(r'$E_{egg}\ (\rm{meV})$')
            
            plt.subplot(324)
            plt.plot(h, 1000 * (Ediss - Ediss[-1]), '-o', label=r'$\Delta E$')
            plt.plot(h, 1000 * Eegg.ptp(axis=1), '-o', label=r'$E_{egg}$')
            plt.title('Energy differences')
            plt.xlabel(u'h [Å]')
            plt.ylabel('energy [meV]')
            plt.legend(loc='best')

            plt.subplot(325)
            plt.plot(h, npy.abs(self.Fegg).max(axis=1), '-o', label='eggbox')
            plt.plot(h, npy.abs(self.Fdimer.sum(axis=2)).max(axis=1), '-o',
                     label='dimer')
            plt.title('Forces')
            plt.xlabel(u'h [Å]')
            plt.ylabel(u'force [eV/Å]')
            plt.legend(loc='best')

            plt.subplot(326)
            E = 2 * B * h**C
            plt.plot(h[j:], Edimer0[j:] - Edimer0[-1], '-o', label='$E_d$')
            plt.plot(h[j:], 2 * Eegg[j:, 0] - 2 * Eegg[-1, 0], '-o',
                     label='$2E_a$')
            plt.plot(h[j:], E[j:] - E[-1], '-o', label='$2A(h/h_0)^n$')
            plt.title('Absolute energies')
            plt.xlabel(u'h [Å]')
            plt.ylabel('energy [eV]')
            plt.legend(loc='best')

            plt.savefig(self.name + '-dimer-eggbox.png', dpi=dpi)
            if show:
                plt.show()
        return self.h[-1], B * 0.2**C, C


def f(x, y, dydx):
    dx = x[1] - x[0]
    x2 = npy.empty(2 * len(x))
    x2[::2] = x - 0.5 * dx
    x2[1::2] = x + 0.5 * dx
    y2 = npy.empty_like(x2)
    y2[::2] = y - dydx * 0.5 * dx
    y2[1::2] = y + dydx * 0.5 * dx
    return x2, y2

def fit(h, e):
    a = e[-1]
    c = 5.0
    b = (e[0] - a) / h[0]**c
    from gpaw.testing.amoeba import Amoeba
    a = Amoeba(lambda x: ((x[0] + x[1] * h**x[2] - e)**2).sum(), [a, b, c],
               tolerance=1e-12)
    x,d = a.optimize(1000)
    return x[1], x[2]

if __name__ == '__main__':
    if len(args) == 0:
        from gpaw.atom.generator import parameters
        args = parameters.keys()
        
    if opt.summary:
        print '    deltaE   deltaEa  Eegg     deltad   Fegg     Ftot'

    for symbol in args:
        ta = TestAtom(symbol)
        ta.run(opt.summary, opt.lcao)
        if opt.summary and ta.ready:
            ta.summary()

