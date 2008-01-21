from ase import *
from gpaw import Calculator

a = 5.475
b = a / 2
si = Atoms(symbols='Si4',
           pbc=True,
           cell=(a, a, a),
           positions=[(0, 0, 0),
                      (0, b, b),
                      (b, 0, b),
                      (b, b, 0)])
si += si
si.positions[4:] += a / 4
calc = Calculator(nbands=16,
                  h=0.25,
                  txt='si.txt')
si.set_calculator(calc)
si.get_potential_energy()
calc.write('si.gpw', mode='all')
