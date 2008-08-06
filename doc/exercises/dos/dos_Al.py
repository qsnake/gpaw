from ase import *
from gpaw import *

a = 4.05
atoms = Atoms(symbols='Al4', cell=(a, a, a), pbc=True,
              scaled_positions=[(.0, .0, .0),
                                (.5, .5, .0),
                                (.0, .5, .5),
                                (.5, .0, .5)])
calc = Calculator(nbands=16, h=0.23, kpts=(6, 6, 6), txt='Al.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()

energy, dos = calc.get_dos(width=.5)

import pylab as p
p.plot(energy, dos)
p.axis('tight')
p.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
p.ylabel('Density of States (1/eV)')
p.show()
