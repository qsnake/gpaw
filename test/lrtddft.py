from gpaw import Calculator
from gpaw.utilities import equal
from ASE import ListOfAtoms, Atom, Crystal
from gpaw.lrtddft import LrTDDFT

R=0.7 # approx. experimental bond length
a = 6
c = 8
H2 = ListOfAtoms([Atom('H', (a/2,a/2,(c-R)/2)),
                  Atom('H', (a/2,a/2,(c+R)/2))],
                 cell=(a,a,c))
calc = Calculator(xc='PBE',nbands=2,spinpol=False)
H2.SetCalculator(calc)
H2.GetPotentialEnergy()

# without spin
lr = LrTDDFT(calc,xc=None)
lr.diagonalize()
t1 = lr[0]

# with spin
lr = LrTDDFT(calc,xc=None,nspins=2)
lr.diagonalize()
t2 = lr[1] # the triplet is lower

equal(t1.GetEnergy(), t2.GetEnergy(), 5.e-7)
