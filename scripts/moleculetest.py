#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser


parser = OptionParser(usage='%prog options',
                      version='%prog 0.1')
parser.add_option('-s', '--summary', action='store_true',
                  default=False,
                  help='Do a summary.')

opt, tests = parser.parse_args()

import os
import sys
import pickle
from math import sqrt

import Numeric as npy
from LinearAlgebra import inverse
from LinearAlgebra import solve_linear_equations as solve
from ASE.Units import Convert

from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import molecules, Molecule
from gpaw.utilities import locked, fix, fix2
from gpaw.paw import ConvergenceError
from atomization_data import atomization_vasp


X = {}

Ea12 = [('H2',   104.6, 104.5),
        ('LiH',   53.5,  53.5),
        ('CH4',  419.8, 419.2),
        ('NH3',  301.7, 301.0),
        ('OH',   109.8, 109.5),
        ('H2O',  234.2, 233.8),
        ('HF',   142.0, 141.7),
        ('Li2',   19.9,  19.7),
        ('LiF',  138.6, 139.5),
        ('Be2',    9.8,   9.5),
        ('C2H2', 414.9, 412.9),
        ('C2H4', 571.5, 570.2),
        ('HCN',  326.1, 324.5),
        ('CO',   268.8, 267.6),
        ('N2',   243.2, 241.2),
        ('NO',   171.9, 169.7),
        ('O2',   143.7, 141.7),
        ('F2',    53.4,  51.9),
        ('P2',   121.1, 117.2),
        ('Cl2',   65.1,  63.1)]

for formula, Ea1, Ea2 in Ea12:
    X[formula] = {'dref': []}
    X[formula]['Earef'] = [(Ea1 / 23.0605, 2), (Ea2 / 23.0605, 3)]

for formula, d in [('Cl2', 1.999),
                   ('CO',  1.136),
                   ('F2',  1.414),
                   ('Li2', 2.728),
                   ('LiF', 1.583),
                   ('LiH', 1.604),
                   ('N2',  1.103),
                   ('O2',  1.218)]:
    X[formula]['dref'].append((d, 1))

for formula, d in [('H2', 1.418),
                   ('N2', 2.084),
                   ('NO', 2.189),
                   ('O2', 2.306),
                   ('F2', 2.672)]:
    X[formula]['dref'].append((d * 0.529177, 4))

parameters = {'xc': 'PBE', 'lmax': 2}

dd = npy.array([(i - 2) * 0.015 for i in range(5)])
a = 12.0
n = 76
h = a / n
atoms = {}
for formula in molecules:
    filename = '%s.pckl' % formula
    x = X[formula]
    if opt.summary:
        try:
            e0, e_i = pickle.load(open(filename))
        except (EOFError, IOError, ValueError):
            e0 = e_i = None
        x['Em0'] = e0
        x['Em'] = npy.array(e_i)
        if len(molecules[formula]) == 2:
            pos = molecules[formula].GetCartesianPositions()
            x['d0'] = pos[1, 0] - pos[0, 0]
    elif not locked(filename):
        file = open(filename, 'w')
        parameters['out'] = formula + '.txt'
        try:
            molecule = Molecule(formula, a=a + 4 * h, b=a, c=a - 4 * h, h=h,
                                parameters=parameters)
            e0 = molecule.energy()
            e_i = []
            if len(molecule.atoms) == 2:
                pos = molecule.atoms[1].GetCartesianPosition()
                for i in range(5):
                    molecule.atoms[1].SetCartesianPosition(pos + [dd[i], 0, 0])
                    e_i.append(molecule.energy())
        except ConvergenceError:
            print >> file, 'FAILED'
        else:
            pickle.dump((e0, e_i), file)
    
    for atom in molecules[formula]:
        atoms[atom.GetChemicalSymbol()] = 1

Ea = {}
for symbol in atoms:
    filename = '%s.pckl' % symbol
    if opt.summary:
        try:
            e0 = pickle.load(open(filename))
        except (EOFError, IOError, ValueError):
            e0 = None
        Ea[symbol] = e0
    elif not locked(filename):
        file = open(filename, 'w')
        parameters['out'] = symbol + '.txt'
        try:
            e0 = SingleAtom(symbol, a=a + 4 * h, b=a, c=a - 4 * h,
                            spinpaired=False,
                            h=h, parameters=parameters, forcesymm=1).energy()
        except ConvergenceError:
            print >> file, 'FAILED'
        else:
            pickle.dump(e0, file)

if opt.summary:
    import pylab
    for formula, molecule in molecules.items():
        x = X[formula]
        if x['Em0'] is None:
            continue
        E0 = 0.0
        ok = True
        for atom in molecule:
            symbol = atom.GetChemicalSymbol()
            if Ea[symbol] is None:
                ok = False
                break
            E0 += Ea[symbol]
        if ok:
            x['Ea'] = E0 - x['Em0']
        if len(molecule) == 2:
            d = x['d0'] + dd
            M = npy.zeros((4, 5), npy.Float)
            for n in range(4):
                M[n] = d**-n
            a = solve(npy.innerproduct(M, M), npy.dot(M, x['Em'] - E0))

            dmin = 1 / ((-2 * a[2] +
                         sqrt(4 * a[2]**2 - 12 * a[1] * a[3])) / (6 * a[3]))
            #B = xmin**2 / 9 / vmin * (2 * a[2] + 6 * a[3] * xmin)

            dfit = npy.arange(d[0] * 0.95, d[4] * 1.05, d[2] * 0.005)

            emin = a[0]
            efit = a[0]
            for n in range(1, 4):
                efit += a[n] * dfit**-n
                emin += a[n] * dmin**-n

            x['d'] = dmin
            x['Eamin'] = -emin
            
            pylab.plot(dfit, efit, '-', color='0.7')
            
            if ok:
                pylab.plot(d, x['Em'] - E0, 'g.')
            else:
                pylab.plot(d, x['Em'] - E0, 'ro')

            pylab.text(dfit[0], efit[0], fix(formula))

    pylab.xlabel(u'Bond length [Å]')
    pylab.ylabel('Energy [eV]')
    pylab.savefig('molecules.png')

    o = open('molecules.txt', 'w')
    print >> o, """\
.. contents::

==============
Molecule tests
==============

Atomization energies (*E*\ `a`:sub:) and bond lengths (*d*) for 20
small molecules calculated with the PBE functional.  All calculations
are done in a box of size 12.6 x 12.0 x 11.4 Å with a grid spacing of
*h*\ =0.16 Å and zero-boundary conditions.  Compensation charges are
expanded with correct multipole moments up to *l*\ `max`:sub:\ =2.
Open-shell atoms are treated as non-spherical with integer occupation
numbers, and zero-point energy is not included in the atomization
energies. The numbers are compared to very accurate, state-of-the-art,
PBE calculations (*ref* subscripts).

.. figure:: molecules.png
   

Bond lengths and atomization energies at relaxed geometries
===========================================================

(*rlx* subscript)

.. list-table::
   :widths: 2 3 8 5 6 8

   * -
     - *d* [Å]
     - *d*-*d*\ `ref`:sub: [Å]
     - *E*\ `a,rlx`:sub: [eV]
     - *E*\ `a,rlx`:sub:-*E*\ `a`:sub: [eV]
     - *E*\ `a,rlx`:sub:-*E*\ `a,rlx,ref`:sub: [eV] [1]_"""
    for formula, Ea1, Ea2 in Ea12:
        x = X[formula]
        if 'Eamin' in x:
            print >> o, '   * -', fix2(formula)
            print >> o, '     - %5.3f' % x['d']
            if 'dref' in x:
                print >> o, ('     - ' +
                             ', '.join(['%+5.3f [%d]_' % (x['d'] - dref, ref)
                                        for dref, ref in x['dref']]))
            else:
                print >> o, '     -'
            print >> o, '     - %6.3f' % x['Eamin']
            print >> o, '     - %6.3f' % (x['Eamin'] - x['Ea'])
            if formula in atomization_vasp:
                print >> o, '     - %6.3f' % (x['Eamin'] -
                                              atomization_vasp[formula][1] /
                                              23.0605)
            else:
                print >> o, '     -'

    print >> o, """\

Atomization energies at experimental geometries
===============================================

.. list-table::
   :widths: 6 6 12

   * -
     - *E*\ `a`:sub: [eV]
     - *E*\ `a`:sub:-*E*\ `a,ref`:sub: [eV]"""
    for formula, Ea1, Ea2 in Ea12:
        x = X[formula]
        print >> o, '   * -', fix2(formula)
        if 'Ea' in x:
            print >> o, '     - %6.3f' % x['Ea']
            if 'Earef' in x:
                print >> o, ('     - ' +
                             ', '.join(['%+5.3f [%d]_' % (x['Ea'] - Ecref, ref)
                                        for Ecref, ref in x['Earef']]))
            else:
                print >> o, '     -'
        else:
            print >> o, '     -'
            print >> o, '     -'
        
    print >> o, """

References
==========

.. [1] "The Perdew-Burke-Ernzerhof exchange-correlation functional
       applied to the G2-1 test set using a plane-wave basis set",
       J. Paier, R. Hirschl, M. Marsman and G. Kresse,
       J. Chem. Phys. 122, 234102 (2005)

.. [2] "Molecular and Solid State Tests of Density Functional
       Approximations: LSD, GGAs, and Meta-GGAs", S. Kurth,
       J. P. Perdew and P. Blaha, Int. J. Quant. Chem. 75, 889-909
       (1999)

.. [3] "Comment on 'Generalized Gradient Approximation Made Simple'",
       Y. Zhang and W. Yang, Phys. Rev. Lett.

.. [4] Reply to [3]_, J. P. Perdew, K. Burke and M. Ernzerhof

"""

    o.close()
    
    os.system('rst2html.py ' +
              '--no-footnote-backlinks ' +
              '--trim-footnote-reference-space ' +
              '--footnote-references=superscript molecules.txt molecules.html')
