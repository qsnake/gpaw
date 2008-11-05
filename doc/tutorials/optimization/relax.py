from ase import *
from gpaw import *

molecule, calc = restart('H2.gpw', txt='H2-relaxed.txt')

e2 = molecule.get_potential_energy()
d0 = molecule.get_distance(0, 1)

print 'experimental bond length:'
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'bondlength              : %5.2f Ang' % d0

# Find the theoretical bond length:
relax = QuasiNewton(molecule, logfile='qn.log')
relax.run(fmax=0.05)

e2 = molecule.get_potential_energy()
d0 = molecule.get_distance(0, 1)

print
print 'PBE energy minimum:'
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'bondlength              : %5.2f Ang' % d0
