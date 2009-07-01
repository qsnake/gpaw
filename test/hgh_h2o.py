#!/usr/bin/env python
"""Test of HGH pseudopotential implementation.

This is the canonical makes-sure-nothing-breaks test, which checks that the
numbers do not change from whatever they were before.

The test runs an HGH calculation on a misconfigured H2O molecule, such that
the forces are nonzero.

Energy is compared to a previous calculation; if it differs significantly,
that is considered an error.

Forces are compared to a previous finite-difference result.
"""

import numpy as np
from ase import *
from gpaw import GPAW
from gpaw.utilities import unpack
from gpaw.poisson import PoissonSolver
from gpaw.atom.basis import BasisMaker

obasis = BasisMaker('O').generate(1)
hbasis = BasisMaker('H').generate(1)

mol = Atoms('OHH', positions=[(0, 0, 0.3), (0, 0.55, -0.2), (0, -0.45, -0.5)],
            pbc=True)
mol.center(vacuum=1.5)
calc = GPAW(nbands=6, h=0.10, # Force is quite bad for h > 0.12, must be eggbox
            setups='hgh',
            poissonsolver=PoissonSolver(relax='GS'),
            mode='fd',
            basis={'H' : hbasis, 'O' : obasis},
            width=0.01,
            txt='-')
mol.set_calculator(calc)
e = mol.get_potential_energy()
F_ac = mol.get_forces()

F_ac_ref = np.array([[ -4.39699761e-05,  -1.70744195e+01,   1.60403209e+01],
                     [ -1.27836192e-02,   2.19624378e+01,  -1.43210490e+01],
                     [ -3.07146973e-03,  -4.90620722e+00,  -1.70774282e+00]])
eref = 728.521507176

eerr = abs(e - eref)

print 'energy', e
print 'ref energy', eref
print 'energy error', eerr

print 'forces'
print F_ac

print 'ref forces'
print F_ac_ref

ferr = np.abs(F_ac - F_ac_ref).max()
print 'max force error', ferr

fdcheck = False

if fdcheck:
    from ase.calculators import numeric_forces
    F_ac_fd = numeric_forces(mol)

    print 'Self-consistent forces'
    print F_ac
    print 'FD forces'
    print F_ac_fd
    print
    print repr(F_ac_fd)
    print
    err = np.abs(F_ac - F_ac_fd).max()
    print 'max err', err

wfs = calc.wfs
gd = wfs.gd
psit_nG = wfs.kpt_u[0].psit_nG
dH_asp = calc.hamiltonian.dH_asp

assert eerr < 1e-3, 'energy changed from reference'
assert ferr < 0.025, 'forces do not match FD check'
# actual force error is 0.013, 0.021 for usenewlfc as of this revision

# Sanity check.  In HGH, the atomic Hamiltonian is constant.
# Also the projectors should be normalized
for a, dH_sp in dH_asp.items():
    dH_p = dH_sp[0]
    K_p = wfs.setups[a].K_p
    #B_ii = wfs.setups[a].B_ii
    #assert np.abs(B_ii.diagonal() - 1).max() < 1e-3
    #print 'B_ii'
    #print wfs.setups[a].B_ii
    # Actually, H2O might not be such a good test, since there'll only
    # be one element in the atomic Hamiltonian for O and zero for H.
    #print 'dH_p', dH_p
    #print 'K_p', K_p

    assert np.abs(dH_p - K_p).max() < 1e-10, 'atomic Hamiltonian changed'

    #h_ii = setup.data.h_ii
    #print 'h_ii', h_ii
    #print 'dH_ii', dH_ii

# Sanity check: HGH is normconserving
for psit_G in psit_nG:
    norm = gd.integrate(psit_G**2) # Around 1e-15 !  Surprisingly good.
    assert abs(1 - norm) < 1e-10, 'Not normconserving'
