import os
import sys
import pickle
from optparse import OptionParser

import Numeric as num
from ASE.Units import Convert
from ASE import ListOfAtoms, Atom

from gpaw.utilities.molecule import molecules
from gpaw.utilities import locked, center
from gpaw.paw import ConvergenceError
from gpaw import Calculator
import data

def setup_molecule(formula, a=12., h=None, gpts=None, sep=None, periodic=False,
                   setups='paw', txt='-'):
    """Constructs a ListOfAtoms corresponding to the specified molecule,
    and attaches a Calculator with suitable parameters."""
    system = molecules[formula].Copy()
    system.SetBoundaryConditions(periodic)
    system.SetUnitCell([a,a,a], fix=True)
    if sep is not None:
        if len(system.atoms) != 2:
            raise ValueError('Separation ambiguous for non-diatomic molecule')
        system.atoms[1].SetCartesianPosition([sep,0,0])
    center(system)
    if gpts is not None:
        gpts = [gpts]*3
    calc = Calculator(h=h, gpts=gpts, xc='PBE', setups=setups, txt=txt)
    system.SetCalculator(calc)
    return system, calc

def atomic_energy(symbol, a=12., h=.16, setups='paw', quick=False, txt='-'):
    """Places an atom of the specified type in the center of a unit cell
    and returns a tuple with the energy and iteration count."""
    if quick:
        a, h = (6., .3)
    c = a/2.
    magmom = data.get_magnetic_moment(symbol)
    system = ListOfAtoms([Atom(symbol, (c,c,c),
                               magmom=data.get_magnetic_moment(symbol))],
                         cell=(a,a,a), periodic = False)
    calc = Calculator(h=h, xc='PBE', setups=setups, hund=True, txt=txt)
    system.SetCalculator(calc)
    energy = system.GetPotentialEnergy()
    return energy, calc.niter

def molecular_energy(formula, a=12., h=.16, sep=None, dislocation=None,
                         setups='paw', quick=False, txt='-'):
    """Calculates the energy of the specified molecule. Returns a tuple
    with the energy and iteration count."""
    if quick:
        (a, h) = (6., .3)
    system, calc = setup_molecule(formula, a, h, sep=sep, setups=setups,
                                  txt=txt)
    energy = system.GetPotentialEnergy()
    return energy, calc.niter

def molecular_energies(formula, a=12., h=.16, displacements=None,
                       setups='paw', quick=False, txt='-'):
    """Calculates the energy of the specified molecule using a range
    of different bond lengths."""
    if quick:
        (a, h) = (6., .3)
    if displacements is None:
        displacements = [(i - 2) * 0.015 for i in range(5)]

    system, calc = setup_molecule(formula, a, h, setups=setups, txt=txt)

    if len(system) != 2:
        displacement = 0.
        energy = system.GetPotentialEnergy()
        niter = calc.niter
        return [displacement], [energy], [niter]

    energies = []
    niters = []
    originalpositions = system.GetCartesianPositions()    
    for displ in displacements:
        system.SetCartesianPositions(originalpositions +
                                     [[-displ/2., 0., 0.],
                                      [+displ/2., 0., 0.]])
        energy = system.GetPotentialEnergy()
        niter = calc.niter
        energies.append(energy)
        niters.append(niter)
        
    return displacements, energies, niters

def eggbox_energies(formula, a=10., gpts=48, count=12, displacements=None,
                    setups='paw', quick=False, txt='-'):
    """Calculates the energy of the specified molecule at 'count'
    different and equally spaced displacements from (0,0,0) to
    (1,1,1)(h/2)"""
    if quick:
        a, gpts, count = (6., 16, 6)
    system, calc = setup_molecule(formula, a, gpts=gpts, periodic=True,
                                  setups=setups, txt=txt)
    originalpositions = system.GetCartesianPositions()
    if displacements is None:
        h = float(a)/gpts
        displacements = [i*h/2./(count-1.) for i in range(count)]
    energies = []
    niters = []

    for displ in displacements:
        # Displace all components by displ along *all* axes
        system.SetCartesianPositions(originalpositions + displ)
        energy = system.GetPotentialEnergy()
        energies.append(energy)
        niters.append(calc.niter)
    return displacements, energies, niters

def grid_convergence_energies(formula, a=10., gpts=None, setups='paw',
                              quick=False, txt='-'):
    """Calcuates a range of energies for different gpoint counts."""
    if quick:
        a, gpts = (6., [20, 24])
    if gpts is None:
        hvalues = [.16, .25]
        gptmax, gptmin = [int(a/h)/4*4 for h in hvalues]
        gpts = range(gptmin, gptmax+1, 4)
    system, calc = setup_molecule(formula, a, gpts=gpts[0], txt=txt,
                                  periodic=True)
    energies = []
    niters = []
    for gptcount in gpts:
        # Bad! Bug! Looks like we'll have to create a new calculator for now
        #calc.set_gpts([gptcount]*3) <-- that doesn't seem to work
        # Also remember that paw doesn't init gd to None, which is bad
        """
        Traceback (most recent call last):
          File "<stdin>", line 1, in ?
          File "main.py", line 116, in calc_grid_convergence_energies
            calc.set_gpts([gptcount]*3)
          File "/home/camp/s021864/gpaw/gpaw/paw.py", line 917, in set_gpts
            self.reset()
          File "/home/camp/s021864/gpaw/gpaw/paw.py", line 902, in reset
            self.stop_paw()
        """
        calc = Calculator(xc='PBE', gpts=[gptcount]*3, setups=setups, txt=txt)
        system.SetCalculator(calc)
        
        energy = system.GetPotentialEnergy()
        energies.append(energy)
        niters.append(calc.niter)
    return gpts, energies, niters

def test_atomization_energy(symbol='H', a=10., h=.18):
    e_atom, niter = calc_atomic_energy(symbol, a, h)
    e_molecule, niter2 = calc_molecular_energy(symbol+'2', a, h)
    energy = e_molecule - 2*e_atom
    print energy, niter, niter2
    return energy

def test_nice_atomization_curve(symbol='H', a=6., h=.25, displs=None):
    if displs is None:
        count = 17
        displs = [(i - count//2) * 0.04 for i in range(count)]
    e_atom, niter = calc_atomic_energy(symbol, a, h)
    dists, energies, niters = calc_molecular_energies(symbol+'2', a, h, displs)
    print dists
    print energies
    return dists, energies

