# -*- coding: utf-8 -*-
import pickle

from ase.data import atomic_numbers, atomic_names


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

    bulk = []
    if symbol in 'Ni Pd Pt La Na Nb Mg Li Pb Rb Rh Ta Ba Fe Mo C K Si W V Zn Co Ag Ca Ir Al Cd Ge Au Cs Cr Cu'.split():
        bulk.append(symbol)
    for alloy in 'La,N Li,Cl Mg,O Na,Cl Ga,N Al,As B,P Fe,Al B,N Li,F Na,F Si,C Zr,C Zr,N Al,N V,N Nb,C Ga,P Al,P B,As Ga,As Mg,S Zn,O Ni,Al Ca,O'.split():
        alloy = alloy.split(',')
        if symbol in alloy:
            bulk.append(''.join(alloy))
    if len(bulk) > 0:
        if len(bulk) == 1:
            bulk = 'test for ' + bulk[0]
        else:
            bulk = 'tests for ' + ', '.join(bulk[:-1]) + ' and ' + bulk[-1]
        f.write("""
Bulk tests
==========

See %s here: :ref:`bulk_tests`.


""" % bulk)

    if mols:
        f.write("""
Molecule tests
==============

See tests for %s here: :ref:`molecule_tests`.


""" % mols)

    if name[0] in 'AEIOUY':
        aname = 'an ' + name.lower()
    else:
        aname = 'a ' + name.lower()
    f.write("""
Convergence tests
=================

The energy of %s dimer (`E_d`) and %s atom (`E_a`) is
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


data = pickle.load(open('../_static/setup-data.pckl'))
for symbol, things in data.items():
    write_rest(symbol, *things)

