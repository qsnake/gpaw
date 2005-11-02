from math import pi
import Numeric as num
from ASE.Utilities.BEE import GetEnsembleEnergies
from ASE import Atom, ListOfAtoms
from gridpaw import Calculator

a = 4.0  # Size of unit cell (Angstrom)

# Hydrogen molecule:
d0 = 0.74  # Experimental bond length
molecule = ListOfAtoms([Atom('H', (0, 0, 0)),
                        Atom('H', (d0, 0, 0))],
                       cell=(a, a, a), periodic=True)
calc = Calculator(h=0.2, nbands=1, xc='PBE', out='H2.out')
molecule.SetCalculator(calc)

e2 = molecule.GetPotentialEnergy()
e2_i = GetEnsembleEnergies(molecule)

dd = 0.02
molecule[1].SetCartesianPosition([d0 + dd, 0, 0])
e3 = molecule.GetPotentialEnergy()
e3_i = GetEnsembleEnergies(molecule)

molecule[1].SetCartesianPosition([d0 - dd, 0, 0])
e1 = molecule.GetPotentialEnergy()
e1_i = GetEnsembleEnergies(molecule)

print 'PBE:'
print d0 - dd, e1
print d0, e2
print d0 + dd, e3
print

# Fit to parabola: a * (d - d0)^2 + b * (d - d0) + c

a = 0.5 * (e1 - 2 * e2 + e3) / dd**2
b = 0.5 * (e3 - e1) / dd
d = d0 - 0.5 * b / a

s = 6.62606876e-34 / pi / 1.66053873e-27**0.5 / 1.6021765e-19**0.5 * 1e13
print 'd =', d, 'Ang'
print 'hv =', a**0.5 * s, 'meV'
print

a_i = 0.5 * (e1_i - 2 * e2_i + e3_i) / dd**2
b_i = 0.5 * (e3_i - e1_i) / dd
d_i = d0 - 0.5 * b_i / a_i

n = len(d_i)
d = num.sum(d_i) / n
sigma = (num.sum((d_i - d)**2) / n)**0.5
print 'Best fit:',
print 'd =', d, '+-', sigma, 'Ang'
hv_i = a_i**0.5 * s
hv = num.sum(hv_i) / n
sigma = (num.sum((hv_i - hv)**2) / n)**0.5
print 'hv =', hv, '+-', sigma, 'meV'

"""
PBE:
0.72 -7.2283408032
0.74 -7.26129060662
0.76 -7.27945669829

d = 0.774575819653 Ang
hv = 555.869866352 meV

Best fit: d = 0.772293333053 +- 0.0216817263585 Ang
hv = 556.574927757 +- 10.1218253384 meV
"""
