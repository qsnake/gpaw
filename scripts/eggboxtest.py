#!/usr/bin/env python

# Emacs: treat this as -*- python -*-

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

from gpaw.utilities.singleatom import SingleAtom
from gpaw.paw import ConvergenceError

symbols = 'H He Li Be C N O F Al Si P Cl'.split()

a = 5.0
data = {}
for symbol in symbols:
    d = data[symbol] = []
    for n in range(16, 36, 4):
        h = a / n
        filename = '%s-%.3f.egg' % (symbol, h)
        if opt.summary:
            try:
                de, f = pickle.load(open(filename))
            except EOFError:
                de, f = '?', '?'
            except IOError:
                de, f = ' ', ' '
            d.append((de, f))
        elif not os.path.isfile(filename):
            file = open(filename, 'w')
            parameters['out']=filename+'.txt'
            atom = SingleAtom(symbol, a=a, spinpaired=True,
                              eggboxtest=True,
                              h=h, parameters=parameters)
            try:
                x, e, dedx = atom.eggboxtest(9)
            except ConvergenceError:
                pass
            else:
                de = max(e) - min(e)
                f = max(abs(dedx))
                pickle.dump((de, f), file)

if opt.summary:
    print '\nEggbox-test (maximum energy variation)\nh:',
    for n in range(16, 36, 4):
        h = a / n
        print '%9.6f' % h,
    print '\n-------------------------------------------------',
    for symbol in symbols:
        d = data[symbol]
        print '\n%-2s' % symbol,
        for de, f in d:
            if type(de) is float:
                print '%9.6f' % de,
            else:
                print '    %s    ' % de,
    print '\n\nEggbox-tex (maximum force)\nh:',
    for n in range(16, 36, 4):
        h = a / n
        print '%9.6f' % h,
    print '\n---------------------------------',
    for symbol in symbols:
        d = data[symbol]
        print '\n%-2s' % symbol,
        for de, f in d:
            if type(de) is float:
                print '%9.6f' % f,
            else:
                print '    %s    ' % f,
