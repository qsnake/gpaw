import os
from gpaw import Calculator
from ase import *
from gpaw.wannier import Wannier
from gpaw.utilities import equal

natoms = 1
hhbondlength = 0.9
atoms = Atoms([Atom('H', (0, 4.0, 4.0))],
                    cell=(hhbondlength, 8., 8.),
                    pbc=True).repeat((natoms, 1, 1))

# Displace kpoints sligthly, so that the symmetry program does
# not use inversion symmetry to reduce kpoints.
assert natoms < 5
kpts = [21, 11, 7, 1][natoms - 1]
occupationenergy = [30., 0., 0., 0.][natoms - 1]
kpts = monkhorst_pack((kpts, 1, 1)) + 2e-5

if 1:
    # GPAW calculator:
    calc = Calculator(nbands=natoms // 2 + 4,
                      kpts=kpts,
                      width=.1,
                      spinpol=False,
                      convergence={'eigenstates': 1e-7})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('hwire%s.gpw' % natoms, 'all')
else:
    calc = Calculator('hwire%s.gpw' % natoms, txt=None)

wannier = Wannier(numberofwannier=natoms,
                  calculator=calc,
                  occupationenergy=occupationenergy,)
#                  initialwannier=[[[1.* i / natoms, .5, .5], [0,], .5]
#                                  for i in range(natoms)])

wannier.localize()
wannier.translate_all_wannier_functions_to_cell([1, 0, 0])

centers = wannier.get_centers()
for i in wannier.get_sorted_indices():
    center = centers[i]['pos']
    print center
    quotient = round(center[0] / hhbondlength)
    equal(hhbondlength*quotient - center[0], 0., 2e-3)
    equal(center[1], 4., 2e-3)
    equal(center[2], 4., 2e-3)

for i in range(natoms):
    wannier.write_cube(i, 'hwire%s.cube' % i, real=True)

os.system('rm hwire1.gpw hwire*.cube')
