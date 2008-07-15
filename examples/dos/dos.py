import sys
import pylab
from gpaw import Calculator

filename = sys.argv[1]
if len(sys.argv) > 2:
    width = float(sys.argv[2])
else:
    width = None

calc = Calculator(filename, txt=None)
energy, dos = calc.get_dos(spin=0, width=width)
pylab.plot(energy, dos)
if calc.get_number_of_spins() == 2:
    energy, dos = calc.get_dos(spin=1, width=width)
    pylap.plot(energy, dos)
    pylap.legend(('up', 'down'), loc='upper left')
pylap.show()
