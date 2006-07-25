#!/usr/bin/env python

import Numeric as num

import os
import sys
import pickle
from optparse import OptionParser


parser = OptionParser(usage='%prog options',
                      version='%prog 0.1')
parser.add_option('-s', '--summary', action='store_true',
                  default=False,
                  help='Do a summary.')

opt, tests = parser.parse_args()

from ASE.Units import Convert

from gridpaw.setuptests.singleatom import SingleAtom
from gridpaw.setuptests.bulk import Bulk, data
from gridpaw.paw import ConvergenceError


parameters = {'lmax': 2}

inf = 1e4000
nan = inf - inf

a = 12.0
h = 0.15
e = {}
for symbol in data:
    bulk = Bulk(symbol)
    cell = num.diagonal(bulk.atoms.GetUnitCell())
    gpts = [4 * int(L / h / 4 + 0.5) for L in cell]
    kpts = [2 * int(15 / L) for L in cell]
    for i in range(5):
        filename = '%s-bulk%d.pickle' % (symbol, i)
        if opt.summary:
            try:
                e0, m0 = pickle.load(open(filename))
            except EOFError:
                e0 = m0 = inf
            except IOError:
                e0 = m0 = nan
            #e[formula] = e0
        elif not os.path.isfile(filename):
            file = open(filename, 'w')
            parameters['out'] = '%s-bulk%d.txt' % (symbol, i)
            try:
                bulk.atoms.SetUnitCell(cell * (1 + (i - 2) * 0.015))
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
        except EOFError:
            e0 = inf
        except IOError:
            e0 = nan
        #e[symbol] = e0
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

if opt.summary:
    print '\nAtomic energies:'
    print '-----------------------'
    for symbol in atoms:
        print '%-4s %8.3f' % (symbol, e[symbol])
    print '-----------------------'
    print '\nAtomization energies:'
    print '-----------------------------------------'
    for formula, molecule in molecules.items():
        ea = -e[formula]
        for atom in molecule:
            ea += e[atom.GetChemicalSymbol()]
        earef = x * atomization[formula][index]
        print ('%-4s %10.3f %8.3f %8.3f %8.3f' %
               (formula, -e[formula], earef, ea, ea - earef))
    print '-----------------------------------------'
                                          
