from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from gridpaw.atom.all_electron import AllElectron as AE

a = 6.8
b = a / 2
d = 1.207

O = ListOfAtoms([Atom('O', [b, b, b], magmom=2)],
                periodic=False,
                cell=(a, a, a))

O2= ListOfAtoms([Atom('O', [b, b, b+d/2], magmom=1),
                 Atom('O', [b, b, b-d/2], magmom=1)],
                periodic=False,
                cell=(a, a, a))

calc  = Calculator(nbands=4, h=0.2, xc='PBE', softgauss=False, hund=True,
                   tolerance=1e-6)#, lmax=2)
calc2 = Calculator(nbands=7, h=0.2, xc='PBE', softgauss=False)#,lmax=2)

O.SetCalculator(calc)
O2.SetCalculator(calc2)

eO    = O.GetPotentialEnergy()
eO2   = O2.GetPotentialEnergy()
excO  = calc.GetXCEnergy()
excO2 = calc2.GetXCEnergy()
exxO  = calc.GetExactExchange()
exxO2 = calc2.GetExactExchange()

atom    = AE('O'); atom.run()
ExxAtom = atom.exactExchange()

print '\t\t\tO atom\t|\tO2 molecule'
print 'ENERGIES:'
print 'potential:\t%2.5f\t|\t%2.5f' %(eO,eO2) 
print 'exchange:\t%2.5f\t|\t%2.5f' %(exxO,exxO2)
print ' '
print 'ATOMIZATION ENERGIES'
print 'potential:\t\t%3.5f' %(eO2-2*eO) 
print 'exchange:\t\t%3.5f' %((eO2-excO2 + exxO2)-2*(eO-excO+exxO)) 
print ' '
print 'Exchange-only atomization energy in kcal/mol: ', ((eO2-excO2 + exxO2)-2*(eO-excO+exxO))*23.2407
print ' '
print 'All electron single atom radial calculator exchange energy:\n
[Eval-val, Eval-core, Ecore-core]'
print ExxAtom
