#!/usr/bin/env python

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
from gridpaw.setuptests.molecule import molecules, Molecule
from gridpaw.paw import ConvergenceError


# atomization energies in kcal / mol:
#                       Exp    LDA    PBE    PBEZY  RPBE   PBEVASP
atomization = {'H2':   (109.5, 113.2, 104.6, 104.5, 105.5,   0  ),
               'LiH':  ( 57.8,  61.0,  53.5,  53.5,  53.4,  53.5),
               'CH4':  (419.3, 462.3, 419.8, 419.2, 410.6, 419.6),
               'NH3':  (297.4, 337.3, 301.7, 301.0, 293.2, 301.7),
               'OH':   (106.4, 124.1, 109.8, 109.5, 106.3, 109.7),
               'H2O':  (232.2, 266.5, 234.2, 233.8, 226.6, 233.7),
               'HF':   (140.8, 162.2, 142.0, 141.7, 137.5, 141.5),
               'Li2':  ( 24.4,  23.9,  19.9,  19.7,  20.2,  19.9),
               'LiF':  (138.9, 156.1, 138.6, 139.5, 132.9, 138.4),
               'Be2':  (  3.0,  12.8,   9.8,   9.5,   7.9,   0  ),
               'C2H2': (405.4, 460.3, 414.9, 412.9, 400.4, 414.5),
               'C2H4': (562.6, 632.6, 571.5, 570.2, 554.5, 571.0),
               'HCN':  (311.9, 361.0, 326.1, 324.5, 313.6, 326.3),
               'CO':   (259.3, 299.1, 268.8, 267.6, 257.9, 268.6),
               'N2':   (228.5, 267.4, 243.2, 241.2, 232.7, 243.7),
               'NO':   (152.9, 198.7, 171.9, 169.7, 161.6, 172.0),
               'O2':   (120.5, 175.0, 143.7, 141.7, 133.3, 143.3),
               'F2':   ( 38.5,  78.2,  53.4,  51.9,  45.6,  52.6),
               'P2':   (117.3, 143.8, 121.1, 117.2, 114.1, 121.5),
               'Cl2':  ( 58.0,  83.0,  65.1,  63.1,  58.9,  65.8)}

x = Convert(1, 'kcal/mol/Nav', 'eV')

index = 2

parameters = {'xc': 'PBE', 'lmax': 2}

inf = 1e4000
nan = inf - inf

a = 12.0
n = 72
h = a / n
e = {}
atoms = {}
for formula in molecules:
    filename = '%s.pickle' % formula
    if opt.summary:
        try:
            e0 = pickle.load(open(filename))
        except EOFError:
            e0 = inf
        except IOError:
            e0 = nan
        e[formula] = e0
    elif not os.path.isfile(filename):
        file = open(filename, 'w')
        parameters['out'] = formula + '.txt'
        try:
            molecule = Molecule(formula, a=a + 1, b=a, c=a - 1, h=h,
                                parameters=parameters)
            e0 = molecule.energy()
            
            e_i = []
            if len(molecule.atoms) == 2:
                x = molecule.atoms[1].GetCartesianPosition()[0]
                for i in range(5):
                    molecule.atoms[1].SetCartesianPosition(
                        [x + (i - 2) * 0.015, 0, 0])
                    e_i.append(molecule.energy())
        except ConvergenceError:
            pass
        else:
            pickle.dump((e0, e_i), file)
    
    for atom in molecules[formula]:
        atoms[atom.GetChemicalSymbol()] = 1

for symbol in atoms:
    filename = '%s.pickle' % symbol
    if opt.summary:
        try:
            e0 = pickle.load(open(filename))
        except EOFError:
            e0 = inf
        except IOError:
            e0 = nan
        e[symbol] = e0
    elif not os.path.isfile(filename):
        file = open(filename, 'w')
        parameters['out'] = symbol + '.txt'
        try:
            e0 = SingleAtom(symbol, a=a + 1, b=a, c=a - 1, spinpaired=False,
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
                                          
