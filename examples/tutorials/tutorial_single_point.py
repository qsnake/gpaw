#!/usr/bin/env python

# This file shows how to do a simple single point calculation for a sodium
# dimer.

# Import ASE classes Atom and ListOfAtoms
from ASE import Atom, ListOfAtoms
# Import GPAW class Calculator
from gpaw import Calculator

# Define your system by creating a list of atoms and defining the unit cell
atoms = ListOfAtoms(
    # List of atoms begins
    [
        # First atom, sodium at (4Å,4Å,4Å)
        Atom('Na', (4, 4, 4)),
        # Second atom, sodium at (4Å,4Å,7Å)
        Atom('Na', (4, 4, 7))
        # List of atoms ends
        ],
    # Cell
    # Size is 8Å x 8Å x 11Å
    cell=(8, 8, 11),
    # Non-periodic in all directions
    periodic=(False,False,False)
    # End of system definition
    )

# Define your calculator
calc = Calculator(
    # Number of electronic bands to be calculated
    nbands=1,
    # Grid spacing in Ångstroms
    # 0.3 Å is enough for alkali and alkaline earth metals
    # 0.2 Å is should be used for non-metals
    h=0.3,
    # Set exchange-correlation functional to Perdew-Wang LSDA
    xc='X-C_PW')

# Assosiate your calculator with your system
atoms.SetCalculator(calc)

# Calculate your system and get the ground state energy
# (The potential energy is the potential energy of your molecular system
# and as the atoms/nuclei are not moving, it's the same as the ground
# state energy.)
e = atoms.GetPotentialEnergy()

# Write everything to a file
calc.write('Na2.gpw', 'all')
