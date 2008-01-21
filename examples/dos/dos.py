import sys
import pylab as p
from ase import *
from gpaw import Calculator

filename = sys.argv[1]
if len(sys.argv) > 2:
    width = float(sys.argv[2])
else:
    width = None

calc = Calculator(filename, txt=None)
dos = DOS(calc, width=width)
if calc.get_number_of_spins() == 1:
    p.plot(dos.get_energies(), dos.get_dos())
else:
    p.plot(dos.get_energies(), dos.get_dos(0), label='up')
    p.plot(dos.get_energies(), dos.get_dos(1), label='down')
    p.legend()
p.show()
