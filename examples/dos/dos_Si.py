from ase import *
from gpaw import Calculator

a = 5.475
atoms = Atoms(symbols='Si4', pbc=True, cell=[a / sqrt(2), a / sqrt(2), a],
              scaled_positions=[(.0, .0, .0),
                                (.5, .5, .5),
                                (.0, .5, .75),
                                (.5, .0, .25),])
calc = Calculator(h=.23, kpts=(6, 6, 4), txt='si.txt', nbands=10)
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('si_664.gpw')

energy, dos = calc.get_dos(width=.2)

import pylab as p
p.plot(energy, dos)
p.axis('tight')
p.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
p.ylabel('Density of States (1/eV)')
p.show()
