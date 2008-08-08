from ase import *
from gpaw import restart

slab, calc = restart('ontop.gpw', txt=None)
AuAl = slab.copy()
AuAl_density = calc.get_pseudo_density()

# Remove gold atom and do a clean slab calculation:
del slab[4]
slab.get_potential_energy()
Al_density = calc.get_pseudo_density()

# Remove Al atoms and do a calculation for Au only:
slab, calc = restart('ontop.gpw', txt=None)
del slab[:4]
calc.set(kpts=None)
slab.get_potential_energy()
Au_density = calc.get_pseudo_density()

diff = AuAl_density - Au_density - Al_density
write('diff.cube', AuAl, data=diff)
write('diff.plt', AuAl, data=diff)
