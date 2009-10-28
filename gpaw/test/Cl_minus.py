from ase import *
from gpaw import GPAW
from gpaw.test import equal

s = Atoms('Cl')
s.center(vacuum=3)
c = GPAW(xc='PBE', nbands=-4, charge=-1, h=0.3)
s.set_calculator(c)

e = s.get_potential_energy()
niter = c.get_number_of_iterations()

print e, niter
energy_tolerance = 0.0003
niter_tolerance = 0
equal(e, -2.9554689657346014, energy_tolerance) # svnversion 5252
#equal(niter, 17, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 16 <= niter <= 18, niter
