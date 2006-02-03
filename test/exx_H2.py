from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from gridpaw.atom.all_electron import AllElectron as AE
from gridpaw.exx import atomic_exact_exchange as aExx

a = 5.2
b = a / 2
d = 0.74

H = ListOfAtoms([Atom('H', [b, b, b], magmom=1)],
                periodic=False,
                cell=(a, a, a))

H2= ListOfAtoms([Atom('H', [b, b, b+d/2], magmom=0),
                 Atom('H', [b, b, b-d/2], magmom=0)],
                periodic=False,
                cell=(a, a, a))

calc  = Calculator(nbands=2, h=0.2, xc = 'PBE', softgauss=False)
calc2 = Calculator(nbands=2, h=0.2, xc = 'PBE', softgauss=False)

H.SetCalculator(calc)
H2.SetCalculator(calc2)

eH    = H.GetPotentialEnergy()
eH2   = H2.GetPotentialEnergy()
excH  = calc.GetXCEnergy()
excH2 = calc2.GetXCEnergy()
exxH  = calc.GetExactExchange()
exxH2 = calc2.GetExactExchange()

atom    = AE('H'); atom.run()
ExxAtom = aExx(atom)

print '\n|-------------------------OUTPUT---------------------------|\n'
print '\t\t\tH atom\t|\tH2 molecule'
print 'ENERGIES:'
print 'potential:\t%5.5f\t|\t%5.5f' %(eH,eH2)
print 'exchange:\t%5.5f\t|\t%5.5f' %(exxH,exxH2)
print ' '
print 'ATOMIZATION ENERGIES'
print 'potential:\t\t%5.5f' %(eH2-2*eH) 
print 'exchange:\t\t%5.5f' %((eH2-excH2 + exxH2)-2*(eH-excH+exxH)) 
print ' '
print 'Exchange-only atomization energy in kcal/mol: ', ((eH2-excH2 + exxH2)-2*(eH-excH+exxH))*23.2407
print ' '
print 'All electron single atom exchange energy: (spin saturated)'
print '  = ', ExxAtom*27.2
print ' '
print 'Exact exchange for hydrogen in eV: ', -5/16.*27.2
print '\n|-------------------------OUTPUT---------------------------|\n'
