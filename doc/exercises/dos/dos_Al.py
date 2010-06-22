import pylab as plt

from ase import Atoms
from gpaw import GPAW

# Lattice constant
a = 4.05

atoms = Atoms(symbols='Al4',
              scaled_positions=[(.0, .0, .0),
                                (.5, .5, .0),
                                (.0, .5, .5),
                                (.5, .0, .5)],
              cell=(a, a, a),
              pbc=True)

calc = GPAW(nbands=16,
            kpts=(6, 6, 6),
            h=0.23,
            txt='Al.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()

energy, dos = calc.get_dos(width=.5)

plt.plot(energy - calc.get_fermi_level(), dos)
plt.axis('tight')
plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel('Density of States (1/eV)')
plt.show()
