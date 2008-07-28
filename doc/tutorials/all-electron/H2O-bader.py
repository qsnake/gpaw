from math import sqrt, pi, cos, sin
from ASE import  ListOfAtoms, Atom
from gpaw import Calculator
from gpaw.utilities import center
from ASE.IO.Cube import WriteCube
from ASE.Units import units, Convert

xy = 7.0  # unit cell length in the plane of molecule
z  = 5.0  # unit cell length perpendicular to the plane of molecule

# Water molecule:
d = 0.9575
t = pi / 180 * 104.51
H2O = ListOfAtoms([Atom('O', (0, 0, 0)),
                   Atom('H', (d, 0, 0)),
                   Atom('H', (d * cos(t), d * sin(t), 0))],
                  cell=[xy, xy, z], periodic=True)
center(H2O)

calc = Calculator(xc='PBE', h=.18)
H2O.SetCalculator(calc)
H2O.GetPotentialEnergy()
calc.write('H2O.gpw')

gridrefinement = 2
n_g = calc.GetAllElectronDensity(gridrefinement)
length = Convert(1, 'Ang', 'Bohr')
WriteCube(H2O, n_g / length**3, 'H2O.cube')
