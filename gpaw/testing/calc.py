import os
import sys
import pickle
from optparse import OptionParser

import Numeric as num
from ASE.Units import Convert
from ASE import ListOfAtoms, Atom
from LinearAlgebra import inverse

from gpaw.testing import g2, data
from gpaw.utilities.bulk import Bulk
from gpaw.utilities import center
from gpaw.paw import ConvergenceError
from gpaw import Calculator

def setup_molecule(formula, a=12., h=None, gpts=None, sep=None, periodic=False,
                   setups='paw', txt='-'):
    """Constructs a ListOfAtoms corresponding to the specified molecule,
    and attaches a Calculator with suitable parameters."""
    try:
        system = data.molecules[formula]
    except: # Try single atom
        system = ListOfAtoms([Atom(formula, magmom=g2.atoms[formula])])
    system.SetBoundaryConditions(periodic)
    system.SetUnitCell([a,a,a], fix=True)
    if sep is not None:
        if len(system.atoms) != 2:
            raise ValueError('Separation ambiguous for non-diatomic molecule')
        system.atoms[1].SetCartesianPosition([sep,0,0])
    center(system)
    if gpts is not None:
        gpts = [gpts]*3
    # Warning: bad hacks...
    if len(system) == 1:
        calc = Calculator(h=h, gpts=gpts, xc='PBE', setups=setups, txt=txt,
                          width=.1)
    else:
        calc = Calculator(h=h, gpts=gpts, xc='PBE', setups=setups, txt=txt)
    system.SetCalculator(calc)
    return system, calc

def atomic_energy(symbol, a=12., h=.16, displacement = None,
                  setups='paw', quick=False, txt='-'):
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
    if displacement is not None:
        system.SetCartesianPositions(system.GetCartesianPositions() +
                                     displacement)
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

def eggbox_energies(formula, a=10., gpts=48, direction=(1.,1.,1.), periods=.5,
                    count=12, setups='paw', quick=False, txt='-'):
    if quick:
        a, gpts, count = (6., 16, 4)

    direction = num.asarray(direction)
    direction = direction / num.dot(direction, direction) # normalize
    system, calc = setup_molecule(formula, a, gpts=gpts, periodic=True,
                                  setups=setups, txt=txt)
    originalpositions = system.GetCartesianPositions()
    h = float(a)/gpts
    displacements = [i*h*periods/(count-1.) for i in range(count)]
    energies = []
    niters = []

    for amount in displacements:
        # Displace all components by 'amount' along 'direction'
        displacement_vector = direction * amount
        system.SetCartesianPositions(originalpositions + displacement_vector)
        energy = system.GetPotentialEnergy()
        energies.append(energy)
        niters.append(calc.niter)
        
    return displacements, energies, niters


def eggbox_energies_old(formula, a=10., gpts=48, count=12, displacements=None,
                        setups='paw', quick=False, txt='-'):
    """Calculates the energy of the specified molecule at 'count'
    different and equally spaced displacements from (0,0,0) to
    (1,1,1)(h/2)"""
    if quick:
        a, gpts, count = (6., 16, 4)
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
        hvalues = [.15, .2]
        gptmax, gptmin = [int(a/h)/4*4 for h in hvalues]
        gpts = range(gptmin, gptmax+1, 4)
    system, calc = setup_molecule(formula, a, gpts=gpts[0], txt=txt,
                                  periodic=True)
    energies = []
    niters = []

    for gptcount in gpts:
        if len(system) == 1:
            calc = Calculator(xc='PBE', gpts=[gptcount]*3, setups=setups,
                              txt=txt, width=.1)
        else:
            calc = Calculator(xc='PBE', gpts=[gptcount]*3, setups=setups,
                              txt=txt)
        system.SetCalculator(calc)
        
        energy = system.GetPotentialEnergy()
        energies.append(energy)
        niters.append(calc.niter)
    return gpts, energies, niters

def lattice_energies(symbol='C', gpts=(24,16,16), kpts=(6,8,8),
                     displacements=None, setups='paw',
                     quick=False, txt='-'):
    # Defaults are good for carbon. Actually, maybe this doesn't work at all
    # for non-carbon elements
    if quick:
        gpts = (8, 8, 8)
        kpts = (2, 2, 2)
    crystal = Bulk(symbol)
    atoms = crystal.atoms
    base_cell = atoms.GetUnitCell()
    base_cell_size = base_cell[0,0]
    aref = base_cell[0,0]
    
    energies = []
    niters = []
    c = Calculator(xc='PBE', gpts=gpts, kpts=kpts, setups=setups, txt=txt)
    atoms.SetCalculator(c)
    # .035 angstroms is probably okay.  This is about equivalent to the
    # number used in the molecule distance test
    if displacements is None:
        lattice_constants = num.array([aref + i*.035 for i in (-1,0,1)])
    else:
        lattice_constants = num.array([aref + d for d in displacements])

    for a in lattice_constants:
        atoms.SetUnitCell(a/base_cell_size * base_cell)
        energy = atoms.GetPotentialEnergy()
        energies.append(energy)
        niters.append(c.niter)

    return displacements, energies, niters

def interpolate(xvalues, yvalues):
    """Utility function for returning a 2nd order polynomial interpolating
    three points. TODO: generalize... """
    x = num.asarray(xvalues)
    y = num.asarray(yvalues)
    xmatrix = num.transpose(num.array([x**0, x**1, x**2]))
    coeffs = num.dot(inverse(xmatrix), y)
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
