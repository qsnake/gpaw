#!/usr/bin/env python

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
from ASE.Units import Convert
from ASE import Atom, ListOfAtoms

from gpaw.utilities.bulk import data
from gpaw.paw import ConvergenceError
from gpaw.utilities import locked
from gpaw import Calculator


inf = 1e4000
nan = inf - inf

def cbrt(x):
    return x**(1.0 / 3.0)

scale = {'sc':      1,
         'bcc':     cbrt(2) * sqrt(3) / 2,
         'fcc':     cbrt(4) / sqrt(2),
         'hcp':     cbrt(2 / sqrt(3)),
         'diamond': cbrt(8) * sqrt(3) / 4}

sigma = {}
Eegg = {}
Fegg = {}
H = {}
for symbol in data:
    X = data[symbol]
    structure = X['structure']
    V = X['volume']
    coa = X.get('c/a', 1)
    magmom = X.get('magmom', 0)

    # Find nearest neighbor distance:
    d = cbrt(V / coa) * scale[structure]
    a = d * sqrt(2)
    z = a / 2
    bulk = ListOfAtoms([Atom(symbol, (0, 0, 0), magmom=magmom),
                        Atom(symbol, (z, z, 0), magmom=magmom),
                        Atom(symbol, (0, z, z), magmom=magmom),
                        Atom(symbol, (z, 0, z), magmom=magmom)],
                       periodic=True,
                       cell=(a, a, a))
    slab = bulk.Copy()
    slab.SetUnitCell([a, a, 2 * a], fix=True)
    
    sigma[symbol] = []
    Eegg[symbol] = []
    Fegg[symbol] = []
    H[symbol] = []
    gmin = max(8, 4 * int(a / 0.30 / 4 + 0.5))
    gmax = 4 * int(a / 0.14 / 4 + 0.5)
    for g in range(gmin, gmax + 4, 4):
        h = a / g
        filename = '%s-surface-%d.pickle' % (symbol, g)
        if opt.summary:
            try:
                e0, f0, e1, f1, e_i, f_i = pickle.load(open(filename))
            except (EOFError, IOError, ValueError):
                e0 = None
            #e[formula] = e0
        elif not locked(filename):
            file = open(filename, 'w')
            try:
                calc = Calculator(h=h, kpts=(2, 2, 1),
                                  out='%s-surface-%d.txt' % (symbol, g))
                slab.SetCalculator(calc)
                e0 = slab.GetPotentialEnergy()
                F = slab.GetCartesianForces()
                f0 = F[2, 2]
                slab[2].SetCartesianPosition([0, z, 0.98 * z])
                slab[3].SetCartesianPosition([z, 0, 0.98 * z])
                e1 = slab.GetPotentialEnergy()
                f1 = slab.GetCartesianForces()[2, 2]
                e_i = []
                f_i = []
                dd = h / 2 / 13
                for i in range(13):
                    slab.SetCartesianPositions(
                        slab.GetCartesianPositions() + (0, 0, dd))
                    e_i.append(slab.GetPotentialEnergy())
                    F = slab.GetCartesianForces()
                    f_i.append(F[0, 2] + F[2, 2])
            except ConvergenceError:
                print >> file, 'FAILED'
            else:
                pickle.dump((e0, f0, e1, f1, e_i, f_i), file)

        filename = '%s-bulk-%d.pickle' % (symbol, g)
        if opt.summary:
            try:
                eb = pickle.load(open(filename))
            except (EOFError, IOError, ValueError):
                eb = None
        elif not locked(filename):
            file = open(filename, 'w')
            calc = Calculator(h=h, kpts=(2, 2, 2),
                              out='%s-bulk-%d.txt' % (symbol, g))
            bulk.SetCalculator(calc)
            try:
                e = bulk.GetPotentialEnergy()
            except ConvergenceError:
                print >> file, 'FAILED'
            else:
                pickle.dump(e, file)

        if opt.summary:
            print symbol, h, e0, eb
            if e0 is not None and eb is not None:
                es = (e0 - eb) / 4
                fmax = max(npy.fabs(f_i))
                emax = max(e_i) - min(e_i)
                H[symbol].append(h)
                sigma[symbol].append(es)
                Eegg[symbol].append(emax)
                Fegg[symbol].append(fmax)


if opt.summary:
    from pylab import plot, show, text, subplot
    subplot(221)
    for symbol in data:
        if len(H[symbol]) > 1:
            plot(H[symbol], sigma[symbol])
            text(H[symbol][0], sigma[symbol][0], symbol)
    subplot(222)
    for symbol in data:
        if len(H[symbol]) > 1:
            plot(H[symbol], Eegg[symbol])
            text(H[symbol][0], Eegg[symbol][0], symbol)
    subplot(223)
    for symbol in data:
        if len(H[symbol]) > 1:
            plot(H[symbol], Fegg[symbol])
            text(H[symbol][0], Fegg[symbol][0], symbol)
    show()
