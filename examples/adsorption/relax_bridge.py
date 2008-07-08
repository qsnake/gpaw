from ase import *
from gpaw import Calculator
from build_fcc import fcc100


a = 4.05
fcc = fcc100('Al', a, 2, 12.0)

# Position of the top layer Al atom:
p = fcc.positions[-1]

# Al - H distance
d = 1.8

# Add the adsorbate:
fcc.append(Atom('H', p + (a / (2 * sqrt(2)), 0, sqrt(d**2 - a**2 / 8))))

calc = Calculator(nbands=2 * 5,
                  kpts=(4, 4, 1),
                  h = 0.25,
                  txt='bridge.txt')
fcc.set_calculator(calc)

# Fix the two Al atoms, so only the H atom is relaxed:
fcc.set_constraint(FixAtoms([0, 1]))

dyn = QuasiNewton(fcc, trajectory='bridge.traj')

# Find optimal height.  The stopping criteria is: the force on the
# H atom should be less than 0.05 eV/Ang
dyn.run(fmax=0.05)

calc.write('bridge.gpw') # Write gpw output after the minimization

print 'Energy bridge:', fcc.get_potential_energy()
print 'Al - H bond:', linalg.norm(fcc.positions[-2] - fcc.positions[-1])

## Energy bridge: -10.7147632731
## Al - H bond: 1.800
