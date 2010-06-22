from math import sqrt

import pylab as plt

from ase import Atoms
from gpaw import GPAW

# Lattice constant
a = 5.475

atoms = Atoms(symbols='Si4',
              scaled_positions=[(.0, .0, .0),
                                (.5, .5, .5),
                                (.0, .5, .75),
                                (.5, .0, .25)],
              cell=(a / sqrt(2), a / sqrt(2), a),
              pbc=True)

calc = GPAW(h=.23,
            kpts=(6, 6, 4),
            nbands=10,
            txt='si.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('si_664.gpw')

energy, dos = calc.get_dos(width=.2)

plt.plot(energy, dos)
plt.axis('tight')
plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel('Density of States (1/eV)')
plt.show()
