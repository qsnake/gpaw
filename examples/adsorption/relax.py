from ase import *
from gpaw import Calculator
from build_fcc import fcc100


a = 4.05
fcc = fcc100('Al', a, 2, 12.0)

# Position of the first layer Al atom:
p = fcc.positions[1]

# Add the adsorbate:
fcc.append(Atom('H', p + (0, 0, 1.55)))

calc = Calculator(nbands=2 * 5,
                  kpts=(4, 4, 1),
                  h = 0.25,
                  txt='ontop.txt')
fcc.set_calculator(calc)

# Only the height (z-coordinate) of the H atom is relaxed:
fcc.set_constraint(FixAtoms([0, 1]))

dyn = QuasiNewton(fcc, trajectory='ontop.traj')

# Find optimal height.  The stopping criteria is: the force on the
# H atom should be less than 0.05 eV/Ang
dyn.run(fmax=0.05)

calc.write('ontop.gpw') # Write gpw output after the minimization

print 'ontop:', fcc.get_potential_energy()
print 'height:', fcc.positions[-1, 2]

pseudo_density = calc.get_pseudo_valence_density()
ae_density = calc.get_all_electron_density()

for format in ['cube', 'plt']:
    write('pseudo.' + format, fcc, data=pseudo_density)
    write('ae.' + format, fcc, data=ae_density)
