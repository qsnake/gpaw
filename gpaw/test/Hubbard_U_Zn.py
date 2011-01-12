from ase import Atom
from ase.units import Hartree

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster
from gpaw.test import equal

h =.3
box = 4.
energy_tolerance = 0.0004

l=2                         # d-orbitals
U_ev=3                      # U in eV
U_au=U_ev / Hartree   # U in atomic units
scale=1                     # Do not scale (does not seem to matter much)
store=0                     # Do not store (not in use yet)

s = Cluster([Atom('Zn')])
s.minimal_box(box, h=h)

E = {}
E_U = {}
for spin in [0, 1]:
    c = GPAW(h=h, spinpol=bool(spin), 
             charge=1, occupations=FermiDirac(width=0.1)
             )
    s.set_calculator(c)
    E[spin] = s.get_potential_energy()
    for setup in c.hamiltonian.setups:
        setup.set_hubbard_u(U_au, l, scale, store) # Apply U
    c.scf.reset()
    E_U[spin] = s.get_potential_energy()

print "E=", E
equal(E[0], E[1], energy_tolerance)
print "E_U=", E_U
equal(E_U[0], E_U[1], energy_tolerance)
