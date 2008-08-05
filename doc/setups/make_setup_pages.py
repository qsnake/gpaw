# -*- coding: utf-8 -*-
import os
import pickle
from sys import argv

from ase.atoms import string2symbols
from ase.data import atomic_numbers, atomic_names
from ase.data.molecules import extra, g2, rest
from ase.data.molecules import data as molecule_data

from gpaw.atom.generator import Generator, parameters
from gpaw.atom.analyse_setup import analyse
from gpaw.testing.dimer import TestAtom
from gpaw.testing.atomization_data import atomization_vasp

systems = atomization_vasp.keys()

bulk = {'Ti': ()}

def make(symbol, show):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]

    if not os.path.isfile(symbol + '.pckl'):
        if os.path.isfile(symbol + '-generator.pckl'):
            gen = pickle.load(open(symbol + '-generator.pckl'))
        else:
            gen = Generator(symbol, 'PBE', scalarrel=True)
            gen.run(logderiv=True, **parameters[symbol])
            gen.txt = None
            pickle.dump(gen, open(symbol + '-generator.pckl', 'w'))
    
        tables = analyse(gen, show=show)

        t = TestAtom(symbol)
        t.run(True, False)
        hmin, B, n = t.summary(show=show)

        molecules = []
        for x in systems:
            if symbol in string2symbols(molecule_data[x]['symbols']):
                molecules.append(molecule_data[x]['name'])
        mols = ''
        if molecules:
            names = [rest(m) for m in molecules]
            if len(molecules) == 1:
                mols = names[0]
            elif len(molecules) > 5:
                mols = ', '.join(names[:5]) + ' ...'
            else:
                mols = ', '.join(names[:-1]) + ' and ' + names[-1]
        pickle.dump((tables, hmin, B, n, mols), open(symbol + '.pckl', 'w'))
    else:
        tables, hmin, B, n, mols = pickle.load(open(symbol + '.pckl'))

    return tables, hmin, B, n, mols

def write_rest(symbol, tables, hmin, B, n, mols):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]
    f = open(symbol + '.rst', 'w')

    f.write(""".. index:: %s

.. _%s:

================
%s
================

.. default-role:: math

""" % (name, name, name))

    if 1:
        f.write("""
Bulk tests
==========

XXX

""" % ())

    if mols:
        f.write("""
Molecule tests
==============

See tests for %s in :ref:`molecule_tests`.


""" % mols)

    if name[0] in 'AEIOUY':
        aname = 'an ' + name.lower()
    else:
        aname = 'a ' + name.lower()
    f.write(u"""
Convergence tests
=================

The energy af %s dimer (`E_d`) and %s atom (`E_a`) is
calculated at diferent grid-spacings (`h`).

.. image:: ../_static/%s-dimer-eggbox.png

A fit to `E_a` for `h` between %.2f Å and 0.20 Å gives:

.. math::  E_a \simeq A + B (h / h_0)^n,

with *B* = %.2f eV and *n* = %.2f (`h_0` = 0.20 Å).
This gives `dE_a/dh=nB/h_0` = %.3f eV/Å for `h=h_0`.

""" % (aname, aname, symbol, hmin, B, n, n * B / 0.2))


    tables = ['\n'.join(t) for t in tables]
    f.write(u"""
Setup details
=============

The setup is based on a scalar-relativistic spin-paired neutral
all-electron PBE calculation.


Electrons
---------

%s

Cutoffs
-------

%s

Energy Contributions
--------------------

%s

Core and valence States
-----------------------

%s

Wave functions, projectors, ...
-------------------------------

.. image:: ../_static/%s-setup.png



Back to :ref:`setups`.

.. default-role::

""" % (tuple(tables) + (symbol,)))



if 0:
    if len(argv) == 1:
        symbols = parameters.keys()
        show = False
    else:
        symbols = argv[1:]
        show = True
    data = {}
    for symbol in symbols:
        print symbol
        if os.path.isdir(symbol):
            data[symbol] = make(symbol, show)
    pickle.dump(data, open('setup-data.pckl', 'w'))
else:
    data = pickle.load(open('../_static/setup-data.pckl'))
    for symbol, things in data.items():
        write_rest(symbol, *things)

