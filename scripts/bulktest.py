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
from LinearAlgebra import inverse
from ASE.Units import Convert

from gridpaw.utilities.singleatom import SingleAtom
from gridpaw.utilities.bulk import Bulk, data
from gridpaw.utilities import locked
from gridpaw.paw import ConvergenceError


parameters = {'lmax': 2, 'xc': 'PBE'}

inf = 1e4000

scale = npy.array([(1 + (i - 2) * 0.015) for i in range(5)])
a = 12.0
h = 0.15
e = {}   # 5 bulk energies
m = {}   # 5 magnetic moments
e1 = {}  # atomic energy
V = {}   # volume
N = {}   # number of atoms
for symbol in data:
    bulk = Bulk(symbol)
    cell = npy.diagonal(bulk.atoms.GetUnitCell())
    gpts = [4 * int(L / h / 4 + 0.5) for L in cell]
    kpts = [2 * int(15 / L) for L in cell]
    e[symbol] = npy.zeros(5, npy.Float)
    m[symbol] = npy.zeros(5, npy.Float)
    V[symbol] = npy.product(cell)
    N[symbol] = len(bulk.atoms)
    for i in range(5):
        filename = '%s-bulk%d.pickle' % (symbol, i)
        if opt.summary:
            try:
                e0, m0 = pickle.load(open(filename))
            except (EOFError, IOError):
                e0 = m0 = inf
            e[symbol][i] = e0
            m[symbol][i] = m0
        elif not locked(filename):
            file = open(filename, 'w')
            parameters['out'] = '%s-bulk%d.txt' % (symbol, i)
            try:
                bulk.atoms.SetUnitCell(cell * scale[i])
                e0, m0 = bulk.energy(gpts=gpts, kpts=kpts,
                                     parameters=parameters)
            except ConvergenceError:
                print >> file, 'FAILED'
            else:
                pickle.dump((e0, m0), file)

    filename = '%s.pickle' % symbol
    if opt.summary:
        try:
            e0 = pickle.load(open(filename))
        except (EOFError, IOError):
            e0 = inf
        e1[symbol] = e0
    elif not locked(filename):
        file = open(filename, 'w')
        parameters['out'] = symbol + '.txt'
        try:
            e0 = SingleAtom(symbol, a=a, b=a + 4 * h, c=a - 4 * h,
                            spinpaired=False,
                            h=h, parameters=parameters, forcesymm=1).energy()
        except ConvergenceError:
            print >> file, 'FAILED'
        else:
            pickle.dump(e0, file)

if opt.summary:
    from pylab import plot, text, show
    M = npy.zeros((4, 5), npy.Float)
    for n in range(4):
        M[n] = scale**-n
    M = npy.dot(inverse(npy.innerproduct(M, M)), M)
    for symbol in data:
        print symbol
        n0 = N[symbol]
        eb = e[symbol] / n0
        ea = e1[symbol]
        v0 = V[symbol] / n0
        if ea != inf:
            eb -= ea
        vb = v0 * scale**3
        if inf in eb:
            continue
        vfit = npy.arange(vb[0] * 0.9, vb[-1] * 1.1, v0 * 0.02)
        a = npy.dot(M, eb)
        xmin = (-2 * a[2] + sqrt(4 * a[2]**2 - 12 * a[1] * a[3])) / (6 * a[3])
        vmin = v0 / xmin**3
        B = xmin**2 / 9 / vmin * (2 * a[2] + 6 * a[3] * xmin)
        ec = -a[0]
        efit = a[0]
        for n in range(1, 4):
            efit += a[n] * (v0 / vfit)**(n / 3.0)
            ec -= a[n] * (v0 / vmin)**(n / 3.0)
        print vmin, B, ec, ea
        plot(vb, eb, 'ro')
        plot(vfit, efit, 'k-')
        text(vmin, -ec, symbol)
    show()

Vexp = {'Na': 255.4,
        }
V1 = {'Na': 249.8,
      'Al': 111.2,
      'Si': 276.3,
      'Ge': 322.1,
      'Cu': 80.6,
      'W':  108.9,
      'Fe': 76.7,
      'Pd': 103.2,
      'Pt': 105.7,
      'Au': 121.1,
      }
B1 = {'Na': 7.6,
      'Al': 77.3,
      'Si': 89.0,
      'Ge': 59.9,
      'Cu': 139,
      'W':  298,
      'Fe': 198,
      'Pd': 174,
      'Pt': 247,
      'Au': 142
      }
