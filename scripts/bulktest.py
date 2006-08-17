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
from ASE.Units import Convert
from ASE.ChemicalElements.symbol import symbols

from gridpaw.utilities.singleatom import SingleAtom
from gridpaw.utilities.bulk import Bulk
from gridpaw.utilities.bulk import data as bulkdata
from gridpaw.utilities import locked
from gridpaw.paw import ConvergenceError

X = {}
for symbol in symbols:
    X[symbol] = {'aref': [], 'Bref': [], 'Ecref': []}

for symbol, Vref in [('Na', 249.8*2),
                     ('Al', 111.2*4),
                     ('Si', 276.3*4),
                     ('Ge', 322.1*4),
                     ('Cu',  80.6*4),
                     ('W',  108.9*2),
                     ('Fe',  76.7*2),
                     ('Pd', 103.2*4),
                     ('Pt', 105.7*4),
                     ('Au', 121.1*4)]:
    X[symbol]['aref'].append((Vref**(1/3.)*0.529177, 1))

for symbol, aref in [('Li', 3.438),
                     ('Na', 4.200),
                     ('Al', 4.040),
                     ('C',  3.574),
                     ('Si', 5.469),
                     ('Cu', 3.635),
                     ('Rh', 3.830),
                     ('Pd', 3.943),
                     ('Ag', 4.147)]:
    X[symbol]['aref'].append((aref, 2))
    
for symbol, Bref in [('Na',   7.6),
                     ('Al',  77.3),
                     ('Si',  89.0),
                     ('Ge',  59.9),
                     ('Cu', 139),
                     ('W',  298),
                     ('Fe', 198),
                     ('Pd', 174),
                     ('Pt', 247),
                     ('Au', 142)]:
    X[symbol]['Bref'].append((Bref, 1))

for symbol, Bref in [('Li',  13.7),
                     ('Na',   7.80),
                     ('Al',  76.6),
                     ('C',  431),
                     ('Si',  87.8),
                     ('Cu', 136),
                     ('Rh', 254),
                     ('Pd', 166),
                     ('Ag',  89.1)]:
    X[symbol]['Bref'].append((Bref, 2))
    
for symbol, Ecref in [('Li', 1.605),
                      ('Na', 1.079),
                      ('Al', 3.434),
                      ('C',  7.706),
                      ('Si', 4.556),
                      ('Cu', 3.484),
                      ('Rh', 5.724),
                      ('Pd', 3.706),
                      ('Ag', 2.518)]:
    X[symbol]['Ecref'].append((Ecref, 2))

X['Fe']['Mref'] = [(2.2, 7)]

parameters = {'lmax': 2, 'xc': 'PBE'}

scale = npy.array([(1 + (i - 2) * 0.015) for i in range(5)])
a = 12.0
h = 0.17
for symbol in bulkdata:
    bulk = Bulk(symbol)
    cell = npy.diagonal(bulk.atoms.GetUnitCell())
    gpts = [4 * int(L / h / 4 + 0.5) for L in cell]
    kpts = [2 * int(15 / L) for L in cell]
    x = X[symbol]
    x['Eb'] = npy.zeros(5, npy.Float)
    x['M'] = npy.zeros(5, npy.Float)
    x['V0'] = npy.product(cell) / len(bulk.atoms)
    x['N'] = len(bulk.atoms)
    for i in range(5):
        filename = '%s-bulk%d.pickle' % (symbol, i)
        if opt.summary:
            try:
                e0, m0 = pickle.load(open(filename))
            except (EOFError, IOError, ValueError):
                e0 = m0 = 117
            x['Eb'][i] = e0
            x['M'][i] = m0
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
        except (EOFError, IOError, ValueError):
            e0 = None
        x['Ea'] = e0
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
    import pylab
    A = npy.zeros((4, 5), npy.Float)
    for n in range(4):
        A[n] = scale**-n
    A = npy.dot(inverse(npy.innerproduct(A, A)), A)
    for symbol in bulkdata:
        x = X[symbol]
        if 117 in x['Eb']:
            continue
        N = x['N']
        Eb = x['Eb'] / N
        Ea = x['Ea']
        V0 = x['V0']
        if Ea is not None:
            Eb -= Ea
        Vb = V0 * scale**3
        Vfit = npy.arange(Vb[0] * 0.9, Vb[-1] * 1.1, V0 * 0.02)
        a = npy.dot(A, Eb)
        s = (-2 * a[2] + sqrt(4 * a[2]**2 - 12 * a[1] * a[3])) / (6 * a[3])
        V = V0 / s**3
        x['B'] = s**2 / 9 / V * (2 * a[2] + 6 * a[3] * s) * 160.21765

        M = x['M'] / N
        aM = npy.dot(A, M)

        Mfit = aM[0]
        Ec = -a[0]
        Efit = a[0]
        for n in range(1, 4):
            Efit += a[n] * (V0 / Vfit)**(n / 3.0)
            Ec -= a[n] * (V0 / V)**(n / 3.0)
            Mfit += aM[n] * (V0 / V)**(n / 3.0)

        if Ea is not None:
            x['Ec'] = Ec
        x['Mfit'] = Mfit
        
        if bulkdata[symbol]['structure'] == 'fcc':
            x['a'] = (V * 2 * N)**(1 / 3.0)
        else:
            x['a'] = (V * N)**(1 / 3.0)

        pylab.plot(Vfit, Efit, '-', color=0.7)

        if Ea is None:
            pylab.plot(Vb, Eb, 'ro')
        else:
            pylab.plot(Vb, Eb, 'go')
            
        pylab.text(Vfit[0], Efit[0], symbol)
        
    #pylab.xlabel(r'$a + \rm{Volume per atom [Å}^3]$')
    pylab.xlabel(u'Volume per atom [Å^3]')
    pylab.ylabel('Energy per atom [eV]')
    pylab.savefig('bulk.png')

    o = open('bulk.txt', 'w')
    print >> o, """\
==========
Bulk tests
==========

Lattice constants (*a*), bulk muduli (*B*) and cohesive energies
(*E*\ `c`:sub:)
are calculated with the PBE functional.  All calculations
are done with a grid spacing of approximately *h*\ =0.1? Å and the
number of **k**-points is given by 30 Å / *a* (rounded to nearest even
number) in all three directions.  Compensation charges are expanded
with correct multipole moments up to *l*\ `max`:sub:\ =2.  Open-shell
atoms are treated as non-spherical with integer occupation numbers,
and zero-point energy is not included in the cohesive energies. The
numbers are compared to very accurate, state-of-the-art, PBE
calculations (*ref* subscripts).


.. figure:: bulk.png

Results
=======

.. list-table::
   :widths: 2 4 9 4 6 5 8

   * -
     - *a* [Å]
     - *a*-*a*\ `ref`:sub: [Å]
     - *B* [GPa]
     - *B*-*B*\ `ref`:sub: [GPa]
     - *E*\ `c`:sub: [eV]
     - *E*\ `c`:sub:-*E*\ `c,ref`:sub:[eV]"""
    for symbol in symbols:
        if symbol not in bulkdata:
            continue
        x = X[symbol]
        print >> o, '   * -', symbol
        for y, f in [('a', '5.3'), ('B', '5.1'), ('Ec', '5.3')]:
            if y in x:
                print >> o, ('     - %%%sf' % f) % x[y]
                if y + 'ref' in x:
                    print >> o, ('     - ' +
                           ', '.join([('%%+%sf [%d]_' % (f, ref))
                                      % (x[y] - Ecref)
                                      for Ecref, ref in x[y + 'ref']]))
                else:
                    print >> o, '     -'
            else:
                print >> o, '     -'
                print >> o, '     -'
        
    print >> o, """

Magnetic moments
================

.. list-table::
   :widths: 2 4 9

   * -
     - *M*
     - *M*-*M*\ `ref`:sub:"""
    for symbol in symbols:
        if symbol not in bulkdata:
            continue
        x = X[symbol]
        if x['M'][0] == 0 or 'Mfit' not in x:
            continue
        print >> o, '   * -', symbol
        print symbol, x
        print >> o, '     - %5.3f' % x['Mfit']
        if 'Mref' in x:
            print >> o, ('     - ' +
                         ', '.join(['%5.2f [%d]_' % (M, ref)
                                    for M, ref in x['Mref']]))
        else:
            print >> o, '     -'
        
    print >> o, """

.. [1] Kurth, Perdew, Blaha
.. [2] JCP 124, 154709"""

    o.close()
    
    os.system('rst2html.py ' +
              '--no-footnote-backlinks ' +
              '--trim-footnote-reference-space ' +
              '--footnote-references=superscript bulk.txt bulk.html')
