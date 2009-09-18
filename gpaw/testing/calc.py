import os
import sys
import pickle

import numpy as npy
from numpy.linalg import inv

#from gpaw.testing import g2
from gpaw.utilities.bulk import Bulk
#from gpaw.paw import ConvergenceError
from gpaw import Calculator


def calculate_energy(formula, a=12., h=.16, sep=None, dislocation=None,
                     setups='paw', quick=False, txt='-'):
    try:
        if quick:
            (a, h) = (6., .3)
        system = molecule(formula, cell = (a,a,a))
        hund = (len(system) == 1)
        calc = Calculator(xc='PBE', h=h, setups=setups, txt=txt, hund=hund)
        system.set_calculator(calc)
        energy = system.get_potential_energy()
    finally:
        system.set_calculator(None)
    return energy, calc.niter


def molecular_energies(formula, a=12., h=.16, displacements=None,
                       setups='paw', quick=False, txt='-'):
    """Calculates the energy of the specified molecule using a range
    of different bond lengths."""
    try:
        if quick:
            (a, h) = (6., .3)
        if displacements is None:
            displacements = [(i - 2) * 0.015 for i in range(5)]

        system = molecule(formula, cell = (a,a,a))
        calc = Calculator(h=h, setups=setups, txt=txt, xc='PBE')
        system.set_calculator(calc)

        if len(system) != 2:
            displacement = 0.
            energy = system.get_potential_energy()
            niter = calc.niter
            return [displacement], [energy], [niter]

        energies = []
        niters = []
        originalpositions = system.positions.copy()
        for displ in displacements:
            system.set_positions(originalpositions +
                                 [[-displ/2., 0., 0.],
                                  [+displ/2., 0., 0.]])
            energy = system.get_potential_energy()
            niter = calc.niter
            energies.append(energy)
            niters.append(niter)
    finally:
        system.set_calculator(None)
    return displacements, energies, niters


def eggbox_energies(formula, a=10., gpts=48, direction=(1.,1.,1.), periods=.5,
                    count=12, setups='paw', quick=False, txt='-'):
    try:
        if quick:
            a, gpts, count = (6., 16, 4)

        direction = npy.asarray(direction)
        direction = direction / npy.dot(direction, direction) ** .5
        system = molecule(formula, cell = (a,a,a))
        system.set_pbc(1)
        calc = Calculator(xc='PBE', gpts=(gpts, gpts, gpts), setups=setups,
                          txt=txt)
        system.set_calculator(calc)

        originalpositions = system.positions.copy()
        h = float(a)/gpts
        #displacements = [i*h*periods/(count-1.) for i in range(count)]
        displacements = npy.linspace(0., h * periods, count)
        energies = []
        niters = []

        for amount in displacements:
            # Displace all components by 'amount' along 'direction'
            displacement_vector = direction * amount
            system.set_positions(originalpositions + displacement_vector)
            energy = system.get_potential_energy()
            energies.append(energy)
            niters.append(calc.niter)
    finally:
        system.set_calculator(None)
    return displacements, energies, niters


def grid_convergence_energies(formula, a=10., gpts=None, setups='paw',
                              quick=False, txt='-'):
    """Calcuates a range of energies for different gpoint counts."""
    try:
        if quick:
            a, gpts = (6., [20, 24])
        if gpts is None:
            hvalues = [.15, .2]
            gptmax, gptmin = [int(a/h)/4*4 for h in hvalues]
            gpts = range(gptmin, gptmax+1, 4)
        system = molecule(formula, cell = (a,a,a))
        system.set_pbc(1)
        calc = Calculator(gpts=gpts[0], txt=txt, xc='PBE', setups=setups)
        system.set_calculator(calc)
        energies = []
        niters = []

        for gptcount in gpts:
            calc.set(gpts=[gptcount]*3)
            energy = system.get_potential_energy()
            energies.append(energy)
            niters.append(calc.niter)
    finally:
        system.set_calculator(None)
    return gpts, energies, niters


def lattice_energies(symbol='C', gpts=(24,16,16), kpts=(6,8,8),
                     displacements=None, setups='paw',
                     quick=False, txt='-'):
    # Defaults are good for carbon. Actually, maybe this doesn't work at all
    # for non-carbon elements
    try:
        if quick:
            gpts = (12, 8, 8)
            kpts = (2, 2, 2)
        crystal = Bulk(symbol)
        system = crystal.atoms
        base_cell = system.get_cell()
        base_cell_size = base_cell[0,0]
        aref = base_cell[0,0]

        energies = []
        niters = []
        c = Calculator(xc='PBE', gpts=gpts, kpts=kpts, setups=setups, txt=txt)
        system.set_calculator(c)
        # .035 angstroms is probably okay.  This is about equivalent to the
        # number used in the molecule distance test
        if displacements is None:
            lattice_constants = npy.array([aref + i*.035 for i in (-1,0,1)])
        else:
            lattice_constants = npy.array([aref + d for d in displacements])

        for a in lattice_constants:
            system.set_cell(a/base_cell_size * base_cell, scale_atoms=True)
            energy = system.get_potential_energy()
            energies.append(energy)
            niters.append(c.niter)
    finally:
        system.set_calculator(None)
    return displacements, energies, niters


def interpolate(xvalues, yvalues):
    """Utility function for returning a 2nd order polynomial interpolating
    three points. TODO: generalize... """
    x = npy.asarray(xvalues)
    y = npy.asarray(yvalues)
    xmatrix = npy.transpose(npy.array([x**0, x**1, x**2]))
    coeffs = npy.dot(inv(xmatrix), y)
    xmin = - coeffs[1] / (2. * coeffs[2]) # "-b/(2a)"
    ymin = coeffs[0] + xmin * (coeffs[1] + xmin * coeffs[2])
    return xmin, ymin, coeffs


def test_atomization_energy(symbol='H', a=10., h=.18, setups='paw'):
    e_atom, niter = atomic_energy(symbol, a, h, setups=setups)
    e_molecule, niter2 = molecular_energy(symbol+'2', a, h, setups=setups)
    energy = e_molecule - 2*e_atom
    print 'E[atom]',e_atom
    print 'E[molecule]',e_molecule
    print 'Ea', energy
    print 'Iterations',niter,niter2
    return energy


def test_dimer_energy_curve(symbol='H', a=6., h=.25, displs=None):
    if displs is None:
        count = 17
        displs = [(i - count//2) * 0.04 for i in range(count)]
    e_atom, niter = atomic_energy(symbol, a, h)
    dists, energies, niters = molecular_energies(symbol+'2', a, h, displs)
    print dists
    print energies
    return dists, energies

if __name__ == '__main__':
    symbol = 'H'
    formula = symbol + '2'
    calculate_energy(symbol, quick=True)
    molecular_energies(formula, quick=True)
    eggbox_energies(formula, quick=True)
    grid_convergence_energies(formula, quick=True)
    lattice_energies('C', quick=True)
