from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from gridpaw.atom.all_electron import AllElectron as AE
from gridpaw.exx import atomic_exact_exchange as aExx

a = 4.8 # => N = 5.6 / 0.2 = 28
b = a / 2
d = 0.74

H = ListOfAtoms([Atom('H', [b, b, b], magmom=1)],
                periodic=False,
                cell=(a, a, a))

Hs= ListOfAtoms([Atom('H', [b, b, b])],
                periodic=False,
                cell=(a, a, a))

H2= ListOfAtoms([Atom('H', [b, b, b+d/2], magmom=0),
                 Atom('H', [b, b, b-d/2], magmom=0)],
                periodic=False,
                cell=(a, a, a))

calc  = Calculator(nbands=2, h=0.2, xc='PBE', tolerance=1e-7, hosts=1)

# Hydrogen reference energy and energy conversion factor:
ref = -12.48992
eV  = 27.211395655517311

# spin compensated calculation for H
Hs.SetCalculator(calc)
esH    = Hs.GetPotentialEnergy()
esxxH  = calc.GetExactExchange(method='real')

# calculation for H2
H2.SetCalculator(calc)
eH2   = H2.GetPotentialEnergy()
excH2 = calc.GetXCEnergy()
exxH2 = calc.GetExactExchange(method='real')

# spin polarized calculation for H
H.SetCalculator(calc)
eH    = H.GetPotentialEnergy()
excH  = calc.GetXCEnergy()
exxH  = calc.GetExactExchange(method='real')

# spin compensated calculation for H with all-electron calculator
atom     = AE('H'); atom.run()
eTotAtom = (atom.Ekin + atom.Epot + atom.Exc) * eV - ref
exxAtom  = aExx(atom) * eV

#Test numbers
Test=[
      calc.GetExactExchange(wannier=False, method='recip', ewald=True),
      calc.GetExactExchange(wannier=False, method='recip', ewald=False),
      calc.GetExactExchange(wannier=False, method='real'),
      calc.GetExactExchange(wannier=True, method='recip',ewald=True,calc=calc),
      calc.GetExactExchange(wannier=True,method='recip',ewald=False,calc=calc),
      calc.GetExactExchange(wannier=True, method='real',calc=calc)
      ]

print '\n|-------------------------OUTPUT---------------------------|\n'
print '          H atom  |  H2 molecule'
print 'ENERGIES :        |'
print 'potential: %6.2f | %6.2f (PAW spin polarized)' %(eH,eH2)
print 'exchange : %6.2f | %6.2f (PAW spin polarized)' %(exxH,exxH2)
print 'potential: %6.2f |   ---  (analytic result)' %(-eV/2 - ref)
print 'exchange : %6.2f |   ---  (analytic result)' %(-5/16. * eV)
print 'potential: %6.2f | %6.2f (PAW spin compensated)' %(esH,eH2)
print 'exchange : %6.2f | %6.2f (PAW spin compensated)' %(esxxH,exxH2)
print 'potential: %6.2f |   ---  (all-electron spin compensated)' %eTotAtom
print 'exchange : %6.2f |   ---  (all-electron spin compensated)' %exxAtom
print ' '
print 'ATOMIZATION ENERGIES (spin polarized):'
print 'potential: %5.2f' %(eH2-2*eH) 
print 'exchange : %5.2f' %((eH2-excH2 + exxH2)-2*(eH-excH+exxH)) 
print ' '
print 'METHOD TESTS:'
print 'No wannier'
print 'recip, ewald    :', Test[0]
print 'recip, not ewald:', Test[1]
print 'real            :', Test[2]
print 'With wannier'
print 'recip, ewald    :', Test[3]
print 'recip, not ewald:', Test[4]
print 'real            :', Test[5]
print '\n|-------------------------OUTPUT---------------------------|\n'

## |-------------------------OUTPUT---------------------------|
##
##           H atom  |  H2 molecule
## ENERGIES :        |
## potential:  -1.02 |  -6.69 (PAW spin polarized)
## exchange :  -8.65 | -18.12 (PAW spin polarized)
## potential:  -1.12 |   ---  (analytic result)
## exchange :  -8.50 |   ---  (analytic result)
## potential:   0.14 |  -6.69 (PAW spin compensated)
## exchange :  -4.14 | -18.12 (PAW spin compensated)
## potential:   0.36 |   ---  (all-electron spin compensated)
## exchange :  -3.85 |   ---  (all-electron spin compensated)
##
## ATOMIZATION ENERGIES (spin polarized):
## potential: -4.66
## exchange : -3.70
##
## METHOD TESTS:
## No wannier
## recip, ewald    : -8.5891064438
## recip, not ewald: -8.6500483466
## real            : -8.65023409136
## With wannier
## recip, ewald    : -8.60474457157
## recip, not ewald: -8.66568810365
## real            : -8.66587227881
##
## |-------------------------OUTPUT---------------------------|

