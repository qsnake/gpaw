#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import center, equal
from gpaw.atom.all_electron import AllElectron as AE
from gpaw.exx import atomic_exact_exchange as aExx
from ASE.Units import units
units.SetEnergyUnit('Hartree')

a = 5.0 
h = 0.178571
b = a / 2
d = 0.74 # H - H bonding distance

H = ListOfAtoms([Atom('H', [0, 0, 0], magmom=1)],
                periodic=False,
                cell=(a, a, a))

Hs= ListOfAtoms([Atom('H', [0, 0, 0])],
                periodic=False,
                cell=(a, a, a))

H2= ListOfAtoms([Atom('H', [0, 0, 0], magmom=0),
                 Atom('H', [0, 0, d], magmom=0)],
                periodic=False,
                cell=(a, a, a))
center(H)
center(Hs)
center(H2)

calc = Calculator(nbands=2, h=h, xc='PBE', tolerance=1e-7,
                  out='exx_H2.txt', softgauss=False)

# spin compensated calculation for H
Hs.SetCalculator(calc)
esH    = Hs.GetPotentialEnergy()
esH   += calc.GetReferenceEnergy()
esxxH  = calc.GetExactExchange()

# calculation for H2
H2.SetCalculator(calc)
eH2   = H2.GetPotentialEnergy()
eH2  += calc.GetReferenceEnergy()
excH2 = calc.GetXCEnergy()
exxH2 = calc.GetExactExchange()

# spin polarized calculation for H
H.SetCalculator(calc)
eH    = H.GetPotentialEnergy()
eH   += calc.GetReferenceEnergy()
excH  = calc.GetXCEnergy()
exxH  = calc.GetExactExchange()

# spin compensated calculation for H with all-electron calculator
atom     = AE('H'); atom.run()
eTotAtom = (atom.Ekin + atom.Epot + atom.Exc)
exxAtom  = aExx(atom)

print '\n|-------------------------OUTPUT---------------------------|\n'
print '          H atom  |  H2 molecule'
print 'ENERGIES :        |'
print 'potential: %6.2f | %6.2f (PAW spin polarized)' %(eH,eH2)
print 'exchange : %6.2f | %6.2f (PAW spin polarized)' %(exxH, exxH2)
print 'potential: %6.2f |   ---  (analytic result)' %(-.5)
print 'exchange : %6.2f |   ---  (analytic result)' %(-5 / 16.)
print 'potential: %6.2f | %6.2f (PAW spin compensated)' %(esH, eH2)
print 'exchange : %6.2f | %6.2f (PAW spin compensated)' %(esxxH, exxH2)
print 'potential: %6.2f |   ---  (all-electron spin compensated)' %eTotAtom
print 'exchange : %6.2f |   ---  (all-electron spin compensated)' %exxAtom
print ' '
print 'ATOMIZATION ENERGIES (spin polarized):'
print 'potential: %5.2f' %(eH2 - 2 * eH) 
print 'exchange : %5.2f' %((eH2 - excH2 + exxH2) - 2 * (eH - excH + exxH)) 
print '\n|-------------------------OUTPUT---------------------------|\n'

equal(eH,    -0.5,     1e-2)
equal(exxH,  -5 / 16., 1e-2)
equal(esH,   eTotAtom, 1e-2)
equal(esxxH, exxAtom,  1e-2)
equal(eH2,   -1.16417, 1e-3)
equal(exxH2, -0.66378, 1e-3)
