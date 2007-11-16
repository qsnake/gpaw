import os

from gpaw import Calculator
from gpaw.utilities import equal
from ASE import ListOfAtoms, Atom, Crystal
from gpaw.lrtddft import LrTDDFT

R=0.7 # approx. experimental bond length
a = 2
c = 3
H2 = ListOfAtoms([Atom('H', (a/2,a/2,(c-R)/2)),
                  Atom('H', (a/2,a/2,(c+R)/2))],
                 cell=(a,a,c))
calc = Calculator(xc='PBE',nbands=2,spinpol=False)
H2.SetCalculator(calc)
H2.GetPotentialEnergy()

xc='LDA'

# without spin
lr = LrTDDFT(calc,xc=xc)
lr.diagonalize()
t1 = lr[0]

# with spin
lr = LrTDDFT(calc,xc=xc,nspins=2)
lr.diagonalize()
# the triplet is lower, so that the second is the first singlet
# excited state
t2 = lr[1]

equal(t1.get_energy(), t2.get_energy(), 5.e-7)

# course grids
for finegrid in [1,0]:
    lr = LrTDDFT(calc, xc=xc, finegrid=finegrid)
    lr.diagonalize()
    t3 = lr[0]
    print "finegrid, t1, t3=", finegrid, t1.get_energy(),t3.get_energy()
    equal(t1.get_energy(), t3.get_energy(), 5.e-4)

# io
fname = 'lr.dat.gz'
lr.write(fname)
lr = LrTDDFT(filename=fname)
os.remove(fname)
t4 = lr[0]

equal(t3.get_energy(), t4.get_energy(), 1.e-7)

