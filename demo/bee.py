import Numeric as na
from ASE.Utilities.BEE import GetEnsembleEnergies
from ASE import Atom, ListOfAtoms
from gridpaw import Calculator

a = 4.0  # Size of unit cell (Angstrom)


# Hydrogen atom:
atom = ListOfAtoms([Atom('H', (0, 0, 0), magmom=1)],
                   cell=(a, a, a), periodic=True)

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = ListOfAtoms([Atom('H', (0, 0, 0)),
                        Atom('H', (d, 0, 0))],
                       cell=(a, a, a), periodic=True)

for xc in ['PBE', 'RPBE']:
    print xc + ':'
    calc = Calculator(h=0.2, nbands=1, xc=xc, out='H1.%s.out' % xc)
    atom.SetCalculator(calc)

    e1 = atom.GetPotentialEnergy()
    c1 = calc.GetEnsembleCoefficients()

    calc = Calculator(h=0.2, nbands=1, xc=xc, out='H2.%s.out' % xc)
    molecule.SetCalculator(calc)
    
    e2 = molecule.GetPotentialEnergy()
    c2 = calc.GetEnsembleCoefficients()

    print 'hydrogen atom energy:     %5.2f eV' % e1
    print 'hydrogen molecule energy: %5.2f eV' % e2
    print 'atomization energy:       %5.2f eV' % (2 * e1 - e2)
    print c1
    print c2
    print c2 - 2 * c1
    
    e1i = GetEnsembleEnergies(c1)
    e2i = GetEnsembleEnergies(c2)
    eai = 2 * e1i - e2i

    n = len(eai)
    ea0 = na.sum(eai) / n
    sigma = (na.sum((eai - ea0)**2) / n)**0.5
    print 'Best fit:', ea0, '+-', sigma, 'eV'

"""
PBE:
hydrogen atom energy:     -1.54 eV
hydrogen molecule energy: -7.26 eV
atomization energy:        4.18 eV
[ 6.07613143 -6.66827244 -1.40262756 -0.34670767]
[  9.79272264 -14.90407254  -3.06734853  -0.82174081]
[-2.35954022 -1.56752766 -0.26209342 -0.12832547]
Best fit: 4.22325811868 +- 0.212531968795 eV
RPBE:
hydrogen atom energy:     -1.49 eV
hydrogen molecule energy: -7.23 eV
atomization energy:        4.25 eV
[ 6.36508117 -6.7626159  -1.42095805 -0.35338538]
[ 10.23571019 -14.97441942  -3.07945711  -0.82672741]
[-2.49445214 -1.44918761 -0.23754101 -0.11995665]
Best fit: 4.21904060409 +- 0.200264479551 eV
"""
