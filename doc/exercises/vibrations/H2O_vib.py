"""Calculate the vibrational modes of a H2O molecule."""
from ase import *
from ase.vibrations import Vibrations
from gpaw import Calculator

h2o = Calculator('h2o.gpw', txt=None).get_atoms()
vib = Vibrations(h2o)
vib.run()
vib.summary()

# Make trajectory files to visualize normal modes:
for mode in range(9):
    vib.write_mode(mode)
