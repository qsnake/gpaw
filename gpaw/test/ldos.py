import os
import numpy as np
from ase import *
from gpaw import GPAW
from gpaw.utilities.dos import raw_orbital_LDOS, raw_wignerseitz_LDOS, RawLDOS
from gpaw.utilities import equal
import gpaw.mpi as mpi
import numpy as np

comms = [mpi.world.new_communicator(np.array([r])) for r in range(mpi.size)]
comm = comms[mpi.rank]

Hnospin = Atoms([Atom('H')], cell=[5, 5, 5], pbc=False)
Hspin = Atoms([Atom('H', magmom=1)], cell=[5, 5, 5], pbc=False)
LiH = Atoms([Atom('Li', [.0, .0, .41]),
             Atom('H', [.0, .0, -1.23])])
Hnospin.center()
Hspin.center()
LiH.center(vacuum=3.0)

# This is needed for the Wigner-Zeitz test to give
# architecture-independent results:
LiH.translate(0.003234)

calc = GPAW(communicator=comm)
Hnospin.set_calculator(calc)
Hnospin.get_potential_energy()
energies, sweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')
energies, pdfweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='pdf')

calc = GPAW(fixmom=True, hund=True, communicator=comm)
Hspin.set_calculator(calc)
Hspin.get_potential_energy()
energies,sweight_spin = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')

calc = GPAW(nbands=2, #eigensolver='dav',
            communicator=comm)
LiH.set_calculator(calc)
LiH.get_potential_energy()
energies, Li_orbitalweight = raw_orbital_LDOS(calc, a=0, spin=0, angular=None)
energies, H_orbitalweight = raw_orbital_LDOS(calc, a=1, spin=0, angular=None)
energies, Li_wzweight = raw_wignerseitz_LDOS(calc, a=0, spin=0)
energies, H_wzweight = raw_wignerseitz_LDOS(calc, a=1, spin=0)
n_a = calc.get_wigner_seitz_densities(spin=0)

print sweight, pdfweight
print sweight_spin
print Li_wzweight
print H_wzweight
print n_a

equal(sweight[0], 1., .06) 
equal(pdfweight[0], 0., .0001) 
equal(sweight_spin[0], 1.14, .06) 
assert ((Li_wzweight - [.13, .93]).round(2) == 0).all()
assert ((H_wzweight - [.87, .07]).round(2) == 0).all()
assert ((Li_wzweight + H_wzweight).round(5) == 1).all()
equal(n_a.sum(), 0., 1e-5)
equal(n_a[1], .737, .001)

print Li_orbitalweight
print H_orbitalweight
#               HOMO    s   py  pz  px  *s
Li_orbitalweight[0] -= [.5, .0, .6, .0, .0]
H_orbitalweight[0]  -= [.7, .0, .0, .0, .0]

#              LUMO       s  py   pz  px  *s
Li_orbitalweight[1] -= [1.0, .0, 0.9, .0, .0]
H_orbitalweight[1]  -= [0.1, .0, 0.0, .0, .0]

assert not Li_orbitalweight.round(1).any()
assert not H_orbitalweight.round(1).any()

ldos = RawLDOS(calc)
fname = 'ldbe.dat'
ldos.by_element_to_file(fname, shift=False)
ldos.by_element_to_file(fname, 2.0, shift=False)
