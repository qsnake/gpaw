#!/usr/bin/env python

# This file shows how to do a geometry optimization for a sodium dimer.

# Import ASE classes Atom and ListOfAtoms
from ASE import Atom, ListOfAtoms
# Import GPAW class Calculator
from gpaw import Calculator

# Import ASE classes MDMin and ConjugateGradient
from ASE.Dynamics.MDMin import MDMin
from ASE.Dynamics.ConjugateGradient import ConjugateGradient

# Read your calculator from a file
calc = Calculator('Na2.gpw')
# Get your system from the calculator
atoms = calc.get_atoms()
# HACK!!! BAD!!! SHOULD NOT NEED THIS LINE
atoms[0].SetCartesianPosition(atoms[0].GetCartesianPosition()+ (0,0,0.00001))

# Calculate unrelaxed energy and bond length
eu = atoms.GetPotentialEnergy()
ru = atoms[1].GetCartesianPosition()[2] - atoms[0].GetCartesianPosition()[2]
print 'Unrelaxed energy = ', eu
print 'Unrelaxed bond length = ', ru

# Do not be too strict with the convergence to speed-up the calculation
calc.set(convergence={'energy' : 0.01, 'density' : 0.01, 'eigenstates' : 1e-5})
# Do a molecular dynamics minimization to get an approximately relaxed structure
md_min = MDMin(atoms, fmax = 1.0)
md_min.Converge()

# Now, for the final structure, be more strict
calc.set(convergence={'energy' : 0.001, 'density' : 0.001, 'eigenstates' : 1e-9})
# Do a conjugate gradient minimization to get the final structure
cg_min = ConjugateGradient(atoms, fmax = 0.05)
cg_min.Converge()

# Calculate relaxed energy and bond length
er = atoms.GetPotentialEnergy()
rr = atoms[1].GetCartesianPosition()[2] - atoms[0].GetCartesianPosition()[2]
print 'Unrelaxed energy = ', eu
print 'Relaxed energy   = ', er
print 'Unrelaxed bond length = ', ru
print 'Relaxed bond length   = ', rr


# Write everything to a file
calc.write('Na2_opt.gpw', 'all')
