from gpaw.atom.configurations import configurations

moleculedata = {}
"""This is a dictionary of formulas containing various reference data.
All the stuff initialized below is stored in this variable."""

def get_magnetic_moment(symbol):
    """Returns the magnetic moment of one atom of the specified element,
    using Hunds rules"""
    magmom = 0.
    for n, l, f, e in configurations[symbol][1]:
        magmom += min(f, 2 * (2 * l + 1) - f)
    return magmom

# The first column containing numbers seems to be identical to
# some pbe values from gpaw.testing.atomization_data, and the
# molecules are obviously taken from that hashtable.
# The final column, however, does not seem to appear anywhere else.
# These values are from "reference 1" and "reference 2", respectively
Ea12 = [('H2',   104.6, 104.5),
        ('LiH',   53.5,  53.5),
        ('CH4',  419.8, 419.2),
        ('NH3',  301.7, 301.0),
        ('OH',   109.8, 109.5),
        ('H2O',  234.2, 233.8),
        ('HF',   142.0, 141.7),
        ('Li2',   19.9,  19.7),
        ('LiF',  138.6, 139.5),
        ('Be2',    9.8,   9.5),
        ('C2H2', 414.9, 412.9),
        ('C2H4', 571.5, 570.2),
        ('HCN',  326.1, 324.5),
        ('CO',   268.8, 267.6),
        ('N2',   243.2, 241.2),
        ('NO',   171.9, 169.7),
        ('O2',   143.7, 141.7),
        ('F2',    53.4,  51.9),
        ('P2',   121.1, 117.2),
        ('Cl2',   65.1,  63.1)]

for formula, Ea1, Ea2 in Ea12:
    moleculedata[formula] = {'dref': []}
    moleculedata[formula]['Earef'] = [(Ea1 / 23.0605, 2), (Ea2 / 23.0605, 3)]

# These values are from "reference 1"
for formula, d in [('Cl2', 1.999),
                   ('CO',  1.136),
                   ('F2',  1.414),
                   ('Li2', 2.728),
                   ('LiF', 1.583),
                   ('LiH', 1.604),
                   ('N2',  1.103),
                   ('O2',  1.218)]:
    moleculedata[formula]['dref'].append((d, 1))

# These values are from "reference 4"
for formula, d in [('H2', 1.418),
                   ('N2', 2.084),
                   ('NO', 2.189),
                   ('O2', 2.306),
                   ('F2', 2.672)]:
    moleculedata[formula]['dref'].append((d * 0.529177, 4))


# The following lists of atoms are from the now deprecated
# gpaw.utilities.molecule.py module.
# These are likely to be removed in the future!

from ASE import Atom, ListOfAtoms
from math import sqrt, pi, sin, cos

# Diatomic molecules: (Angstrom units)
H2 = ListOfAtoms([Atom('H', (0, 0, 0)),
                  Atom('H', (0.7414, 0, 0))])
HF = ListOfAtoms([Atom('H', (0, 0, 0)),
                  Atom('F', (0.9169, 0, 0))])
OH = ListOfAtoms([Atom('O', (0, 0, 0), magmom=0.5),
                  Atom('H', (1.0, 0, 0), magmom=0.5)]) # ??????????
CO = ListOfAtoms([Atom('C', (0, 0, 0)),
                  Atom('O', (1.1283, 0, 0))])
C2 = ListOfAtoms([Atom('C', (0, 0, 0)),
                  Atom('C', (1.0977, 0, 0))])
N2 = ListOfAtoms([Atom('N', (0, 0, 0)),
                  Atom('N', (1.0977, 0, 0))])
NO = ListOfAtoms([Atom('N', (0, 0, 0), magmom=0.6),
                  Atom('O', (1.1506, 0, 0), magmom=0.4)])
O2 = ListOfAtoms([Atom('O', (0, 0, 0), magmom=1),
                  Atom('O', (1.2074, 0, 0), magmom=1)])
F2 = ListOfAtoms([Atom('F', (0, 0, 0)),
                  Atom('F', (1.4119, 0, 0))])
P2 = ListOfAtoms([Atom('P', (0, 0, 0)),
                  Atom('P', (1.8931, 0, 0))])
LiH = ListOfAtoms([Atom('Li', (0, 0, 0)),
                   Atom('H', (1.5949, 0, 0))])
Li2 = ListOfAtoms([Atom('Li', (0, 0, 0)),
                   Atom('Li', (2.6729, 0, 0))])
Be2 = ListOfAtoms([Atom('Be', (0, 0, 0)),
                   Atom('Be', (2.45, 0, 0))])
Cl2 = ListOfAtoms([Atom('Cl', (0, 0, 0)),
                   Atom('Cl', (1.9878, 0, 0))])
LiF = ListOfAtoms([Atom('Li', (0, 0, 0)),
                   Atom('F', (1.5639, 0, 0))])
Cl2 = ListOfAtoms([Atom('Cl', (0, 0, 0)),
                   Atom('Cl', (1.9878, 0, 0))])

# Methane molecule
x = 1.087 / sqrt(3)
CH4 = ListOfAtoms([Atom('C', (0, 0, 0)),
                   Atom('H', (x, x, x)),
                   Atom('H', (-x, -x, x)),
                   Atom('H', (x, -x, -x)),
                   Atom('H', (-x, x, -x))])

# Ammonia molecule
d = 1.012
t = pi / 180 * 106.7
x = 2 * d / sqrt(3) * sin(t / 2)
z = sqrt(d**2 - x**2)
NH3 = ListOfAtoms([Atom('N', (0, 0, 0)),
                   Atom('H', (x, 0, z)),
                   Atom('H', (-x / 2, sqrt(3) * x / 2, z)),
                   Atom('H', (-x / 2, -sqrt(3) * x / 2, z))])

# Water molecule
d = 0.9575
t = pi / 180 * 104.51
H2O = ListOfAtoms([Atom('O', (0, 0, 0)),
                   Atom('H', (d, 0, 0)),
                   Atom('H', (d * cos(t), d * sin(t), 0))])

# Acetylene molecule
d1 = 1.203
d2 = 1.060
C2H2 = ListOfAtoms([Atom('C', (0, 0, 0)),
                    Atom('C', (d1, 0, 0)),
                    Atom('H', (-d2, 0, 0)),
                    Atom('H', (d1 + d2, 0, 0))])

# Ethylene molecule
d1 = 1.339
d2 = 1.087
t = pi / 180 * 121.3
C2H4 = ListOfAtoms([Atom('C', (0, 0, 0)),
                    Atom('C', (d1, 0, 0)),
                    Atom('H', (d2 * cos(t), d2 * sin(t), 0)),
                    Atom('H', (d2 * cos(t), -d2 * sin(t), 0)),
                    Atom('H', (d1 - d2 * cos(t), d2 * sin(t), 0)),
                    Atom('H', (d1 - d2 * cos(t), -d2 * sin(t), 0))])

# HCN molecule
d1 = 1.0655
d2 = 1.1532
HCN = ListOfAtoms([Atom('H', (-d1, 0, 0)),
                   Atom('C', (0, 0, 0)),
                   Atom('N', (d2, 0, 0))])

molecules = {}
for m in ['H2', 'LiH', 'CH4', 'NH3', 'OH',
          'H2O', 'HF', 'Li2', 'LiF', 'Be2',
          'C2H2', 'C2H4', 'HCN', 'CO', 'N2',
          'NO', 'O2', 'F2', 'P2', 'Cl2']:
    molecules[m] = eval(m)

