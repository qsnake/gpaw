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

import Numeric as npy
from LinearAlgebra import inverse
from ASE.Units import Convert

from gridpaw.setuptests.singleatom import SingleAtom
from gridpaw.setuptests.bulk import Bulk, data
from gridpaw.paw import ConvergenceError


parameters = {'lmax': 2}

inf = 1e4000
nan = inf - inf

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
        elif not os.path.isfile(filename):
            file = open(filename, 'w')
            parameters['out'] = '%s-bulk%d.txt' % (symbol, i)
            try:
                bulk.atoms.SetUnitCell(cell * scale[i])
                e0, m0 = bulk.energy(gpts=gpts, kpts=kpts,
                                     parameters=parameters)
            except ConvergenceError:
                pass
            else:
                pickle.dump((e0, m0), file)

    filename = '%s.pickle' % symbol
    if opt.summary:
        try:
            e0 = pickle.load(open(filename))
        except (EOFError, IOError):
            e0 = inf
        e1[symbol] = e0
    elif not os.path.isfile(filename):
        file = open(filename, 'w')
        parameters['out'] = symbol + '.txt'
        try:
            e0 = SingleAtom(symbol, a=a, spinpaired=False,
                            h=h, parameters=parameters, forcesymm=1).energy()
        except ConvergenceError:
            pass
        else:
            pickle.dump(e0, file)

from pylab import plot, show
if opt.summary:
    M = npy.zeros((4, 5), npy.Float)
    for n in range(4):
        M[n] = scale**-n
    M = npy.dot(inverse(npy.innerproduct(M, M)), M)
    for symbol in data:
        print symbol
        ea = e[symbol] / N[symbol]
        va = V[symbol] * scale**3 / N[symbol]
        print ea
        if inf in ea:
            continue
        vfit = npy.arange(va[0] * 0.9, va[-1] * 1.1, va[2] * 0.02)
        a = npy.dot(M, ea)
        efit = a[0]
        for n in range(1, 4):
            efit += a[n] * (va[2] / vfit)**(n / 3.0)
        plot(va, ea, 'o', vfit, efit, '-')
    show()
