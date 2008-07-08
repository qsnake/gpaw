from ase import *
from gpaw import Calculator
from build_fcc import fcc100


a = 4.05
fcc = fcc100('Al', a, 2, 12.0)

# Position of the top layer Al atom:
p = fcc.positions[-1]

# Al - H distance = Sum of covalent radii
d = 1.18 + 0.37

# Add the adsorbate:
fcc.append(Atom('H', p + (0, 0, d)))

calc = Calculator(nbands=2 * 5,
                  kpts=(4, 4, 1),
                  h = 0.25,
                  txt='ontop.txt')
fcc.set_calculator(calc)

# Fix the two Al atoms, so only the H atom is relaxed:
fcc.set_constraint(FixAtoms([0, 1]))

dyn = QuasiNewton(fcc, trajectory='ontop.traj')

# Find optimal height.  The stopping criteria is: the force on the
# H atom should be less than 0.05 eV/Ang
dyn.run(fmax=0.05)

calc.write('ontop.gpw') # Write gpw output after the minimization

print 'Energy ontop:', fcc.get_potential_energy()
print 'Al - H bond:', linalg.norm(fcc.positions[-2] - fcc.positions[-1])

## Energy ontop: -10.6734520914
## Al - H bond: 1.580
