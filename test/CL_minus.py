from ase import *
from gpaw import GPAW

s = Atoms('Cl')
s.center(vacuum=3)
c = GPAW(xc='PBE', nbands=-4, charge=-1, h=0.3)
c.calculate(s)
