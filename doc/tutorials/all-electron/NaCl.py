#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import Numeric as num

unitcell = num.array([6., 6., 9.])
gridrefinement = 2
parameters = {'h': .19, 'xc': 'PBE', 'convergence': {'eigenstates': 1e-7}}

# Pseudo- and all-electron densities of all three systems
nt = {}
n = {}

# Setup lists of atoms
b = unitcell / 2. # middle of unit cell
d = num.array([0, 0, 2.3608]) # experimental bond length
Na = ListOfAtoms([Atom('Na', b, magmom=1)],
                 periodic=False, cell=unitcell)
Cl = ListOfAtoms([Atom('Cl', b, magmom=1)],
                 periodic=False, cell=unitcell)
NaCl = ListOfAtoms([Atom('Na', b - d / 2, magmom=1),
                    Atom('Cl', b + d / 2, magmom=1)],
                   periodic=False, cell=unitcell)
calc = Calculator(**parameters)

# Determine densities
for loa in ('Na', 'Cl', 'NaCl'):
    # Do calculation
    eval(loa).SetCalculator(calc)
    eval(loa).GetPotentialEnergy()

    # Get densities (sum over spin)
    nt[loa] = num.sum(calc.GetDensityArray())
    n[loa] = num.sum(calc.GetAllElectronDensity(gridrefinement))

    dv = num.product(calc.GetGridSpacings())
    print 'Charge of', loa
    print 'nt =', num.sum(nt[loa].flat) * dv
    print 'n =', num.sum(n[loa].flat) * dv / gridrefinement**3

import cPickle as pickle
pickle.dump((nt, n), open('temp.pickle', 'w'))

# Do the plotting
from pylab import axes, plot, title, xlabel, ylabel, axis, savefig, figure, legend

# Extract z-axis
hz = calc.GetGridSpacings()[2]
hzt = hz / gridrefinement
zt = num.arange(hz, unitcell[2] - 1e-4, hz)
z = num.arange(hzt, unitcell[2] - 1e-4, hzt)
N = calc.GetNumberOfGridPoints()

f = figure(num=1, figsize=(10,6))
axes_pos = {'Na': [.15, .6, .25, .25],
            'Cl': [.60, .6, .25, .25],
            'NaCl': [.05, .06, .9, .88]}
axes_kw = {'Na': {'axisbg': 'y', 'xticks': [], 'yticks': []},
           'Cl': {'axisbg': 'y', 'xticks': [], 'yticks': []},
           'NaCl': {'axisbg': 'w'}}

for loa in ('NaCl', 'Na', 'Cl'):
    axes(axes_pos[loa], **axes_kw[loa])
    nt_z = nt[loa][N[0] / 2, N[1] / 2]
    n_z = n[loa][N[0] / 2 * gridrefinement, N[1] / 2 * gridrefinement]
    plot(zt, nt_z, 'b--', z, n_z, 'r-')
    title(loa)
    if loa == 'NaCl':
        xlabel('z'); ylabel('Density')
        legend((r'$\tilde{n}$', r'$n$'), loc='lower right')
    axis([unitcell[2] / 4, 3 * unitcell[2] / 4, 0, 2 * max(nt_z)])

savefig('NaCl.png', dpi=60)
## CHARGE:
## Sys :  nt    n
##   Na:  1.60 11.00
##   Cl:  7.50 17.00
## NaCl:  9.07 28.00
