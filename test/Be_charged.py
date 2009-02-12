from gpaw import GPAW
from ase import *
from gpaw.utilities import equal

a = 7.0

Be_solid = Atoms([Atom('Be', (0.0, 0.0, 0.0), magmom=0)], 
                 cell=(a, a, a), pbc=True)
Be_alone = Atoms([Atom('Be', (a/2., a/2., a/2.), magmom=0)], 
                 cell=(a, a, a), pbc=False)

Be_solidC = Atoms([Atom('Be', (0.0, 0.0, 0.0), magmom=1)], 
                 cell=(a, a, a), pbc=True)
Be_aloneC = Atoms([Atom('Be', (a/2., a/2., a/2.), magmom=1)], 
                 cell=(a, a, a), pbc=False)


Be_solid.set_calculator(GPAW(h=0.3, nbands=1))
E_solid_neutral = Be_solid.get_potential_energy()
Be_solidC.set_calculator(GPAW(h=0.3, charge=+1, nbands=1))
E_solid_charged = Be_solidC.get_potential_energy()

Be_alone.set_calculator(GPAW(h=0.3, nbands=1))
E_alone_neutral = Be_alone.get_potential_energy()
Be_aloneC.set_calculator(GPAW(h=0.3, charge=+1, nbands=1))
E_alone_charged = Be_aloneC.get_potential_energy()

print "A test for periodic charged calculations"
print "Be neutal solid:  ", E_solid_neutral, " eV"
print "Be neutal alone:  ", E_alone_neutral, " eV"
print "Be charged solid: ", E_solid_charged, " eV"
print "Be charged alone: ", E_alone_charged, " eV"
IPs = E_solid_neutral - E_solid_charged
IPa = E_alone_neutral - E_alone_charged
print "Ionization potential solid", IPs, " eV"
print "Ionization potential alone", IPa, " eV"

# Make sure that the ionization potential won't differ by more than 0.05eV
equal(Ips, IPa, 0.05) 

from gpaw.utilities import equal


"""Some results:

WITH CORRECTION TURNED ON:
a = 12.0
Be neutal solid:   0.000241919665987  eV
Be neutal alone:   0.000520077168021  eV
Be charged solid:  8.95928916603  eV
Be charged alone:  9.03395959412  eV
Ionization potential solid -8.95904724637  eV
Ionization potential alone -9.03343951695  eV
Difference: 0.07eV

WITHOUT CORRECTION:
Be neutal solid:     0.000241919665987  eV
Be neutal alone:     0.000520077168021  eV
Be charged solid:    7.30611652236 eV !!!
Be charged alone:    9.03395959412  eV
Difference: 1.73eV
"""
