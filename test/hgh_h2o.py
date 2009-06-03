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
from gpaw import GPAW, extra_parameters
from gpaw.utilities import unpack
from gpaw.poisson import PoissonSolver
from gpaw.atom.basis import BasisMaker

obasis = BasisMaker('O').generate(1)
hbasis = BasisMaker('H').generate(1)

mol = Atoms('OHH', positions=[(0, 0, 0.3), (0, 0.55, -0.2), (0, -0.45, -0.5)],
            pbc=True)
mol.center(vacuum=1.5)
calc = GPAW(nbands=6, h=0.12, # Force is quite bad for h > 0.12, must be eggbox
            setups='hgh',
            poissonsolver=PoissonSolver(relax='GS'),
            mode='fd',
            basis={'H' : hbasis, 'O' : obasis},
            width=0.01,
            txt='hgh_h2o.txt')
mol.set_calculator(calc)
e = mol.get_potential_energy()
F_ac = mol.get_forces()

F_ac_ref = np.array([[  1.60558264e-02,  -1.87217919e+01,   1.61108415e+01],
                     [  1.90798227e-03,   2.15975929e+01,  -1.40629431e+01],
                     [ -9.00374977e-03,  -4.75381951e+00,  -1.47746523e+00]])
if extra_parameters.get('usenewlfc'):
    F_ac_ref = np.array([[ -3.35875330e-03, -1.86201492e+01,  1.68200423e+01],
                         [  2.81319723e-03,  2.16079466e+01, -1.41077550e+01],
                         [  1.18679540e-02, -4.76265566e+00, -1.53033927e+00]])

eref = -943.845730575
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

#assert eerr < 1e-3, 'energy changed from reference'
assert ferr < 0.027, 'forces do not match FD check'

# Sanity check.  In HGH, the atomic Hamiltonian is constant.
for a, dHsp in dH_asp.items():
    dH_p = dH_sp[0]
    K_p = wfs.setups[a].K_p
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
