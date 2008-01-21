#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser

import os
import sys
import pickle

import numpy as npy
from ASE.Units import Convert

from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import molecules, Molecule
from gpaw.utilities import locked
from gpaw.paw import ConvergenceError

from gpaw.testing.data import moleculedata

parser = OptionParser(usage='%prog options',
                      version='%prog 0.1')
parser.add_option('-s', '--summary', action='store_true',
                  default=False,
                  help='Do a summary.')

opt, setups = parser.parse_args()

if len(setups) > 1:
    raise RuntimeException('Please specify only one setup')
elif len(setups) == 1:
    setup = setups[0]
    fileprefix = setup+'.'
else:
    fileprefix = ''
    setup = None

parameters = {'xc': 'PBE'}
if setup is not None:
    parameters['setups'] = setup

# This is a set of distances which is considered during calculation of
# bond lengths
dd = npy.array([(i - 2) * 0.015 for i in range(5)])
a = 12.0
n = 76
h = a / n
atoms = {}

results = {}

def calc_molecule_energy(formula):
    filename = '%s%s.pckl' % (fileprefix, formula)
    #reference = moleculedata[formula]
    result = {}
    results[formula] = result
    if opt.summary:
        try:
            e0, e_i = pickle.load(open(filename))
        except (EOFError, IOError, ValueError):
            e0 = e_i = None
        result['Em0'] = e0
        result['Em'] = npy.array(e_i)
        if len(molecules[formula]) == 2:
            pos = molecules[formula].GetCartesianPositions()
            result['d0'] = pos[1, 0] - pos[0, 0]
    elif not locked(filename):
        file = open(filename, 'w')
        parameters['txt'] = formula + '.txt'
        try:
            molecule = Molecule(formula, a=a + 4 * h, b=a, c=a - 4 * h, h=h,
                                parameters=parameters)
            if formula in ['OH', 'NO']:
                print 'This is %s, so break symmetry' %formula
                #if formula == 'NO':
                #    print 'lets do some experiments'
                #    molecule=Molecule(formula, a=12., b=12., c=12., h=.2,
                #                      parameters=parameters)
                # Displace from molecular (the x-)axis to break symmetry
                # This is necessary because these molecules are asymmetric
                for atom in molecule.atoms:
                    pos = atom.GetCartesianPosition()
                    atom.SetCartesianPosition(pos + [0, .2, .0])
                print molecule.atoms
            try:
                e0 = molecule.energy()
            except RuntimeError: # In case the setup does not exist
                print 'Could not find setup for %s' % formula
                return
            e_i = []
            if len(molecule.atoms) == 2:
                pos = molecule.atoms[1].GetCartesianPosition()
                for displ in dd:
                    molecule.atoms[1].SetCartesianPosition(pos + [displ, 0, 0])
                    e_i.append(molecule.energy())
        except ConvergenceError:
            print >> file, 'FAILED'
        else:
            pickle.dump((e0, e_i), file)

    # This will ensure that the energy of that particular atom
    # will be calculated later. In fact, the set of atoms that will be
    # considered later is exactly the entries of this hashtable.
    # The '1' is just for show.
    # Actually, this should have been done with the builtin 'set'
    for atom in molecules[formula]:
        atoms[atom.GetChemicalSymbol()] = 1

def calc_atomization_energy(symbol, Ea):

    filename = '%s%s.pckl' % (fileprefix, symbol)
    if opt.summary:
        try:
            e0 = pickle.load(open(filename))
        except (EOFError, IOError, ValueError):
            e0 = None
        Ea[symbol] = e0
    elif not locked(filename):
        file = open(filename, 'w')
        parameters['txt'] = symbol + '.txt'
        try:
            spinpaired = False
            if symbol == 'Be':
                # Special treatment for troublesome elements
                spinpaired = True
            e0 = SingleAtom(symbol, a=a + 4 * h, b=a, c=a - 4 * h,
                            spinpaired=spinpaired,
                            h=h, parameters=parameters, forcesymm=1).energy()
        except ConvergenceError:
            print >> file, 'FAILED'
        else:
            pickle.dump(e0, file)

def main():
    for formula in molecules:
        print 'Calculating or getting molecule energy:',formula
        
        """This will calculate molecule energies and dump them to
        files.  This will ignore files that already exist, presuming
        that the energies in question are already calculated"""
        calc_molecule_energy(formula)

    Ea = {}
    for symbol in atoms:
        print 'Calculating or getting atomic energy:',symbol
        """ This will calculate energies of single atoms, storing them
        in files as appropriate.  """

        calc_atomization_energy(symbol, Ea)
        
    if opt.summary:
        import data2restructured
        print 'Generating restructured text'
        data2restructured.main(molecules, moleculedata, results, Ea)
    

if __name__ == '__main__':
    main()
