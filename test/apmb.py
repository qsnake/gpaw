from ASE import ListOfAtoms, Atom, Crystal

from gpaw.utilities import equal
from gpaw import Calculator
from gpaw.lrtddft import LrTDDFT

txt='-'
txt='/dev/null'

load = False
##    load = True
    
if not load:
    R=0.7 # approx. experimental bond length
    a = 3
    c = 4
    H2 = ListOfAtoms([Atom('H', (a/2,a/2,(c-R)/2)),
                      Atom('H', (a/2,a/2,(c+R)/2))],
                     cell=(a,a,c))
    calc = Calculator(xc='PBE', nbands=2, spinpol=False, txt=txt)
    H2.SetCalculator(calc)
    H2.GetPotentialEnergy()
##    calc.write('H2.gpw', 'all')
else:
    calc = Calculator('H2.gpw', txt=txt)
calc.initialize_wave_functions()

xc='LDA'

# no spin

lr = LrTDDFT(calc, xc=xc)
lr.diagonalize()

lr_ApmB = LrTDDFT(calc, xc=xc, force_ApmB=True)
lr_ApmB.diagonalize()
print 'lr=', lr
print 'ApmB=', lr_ApmB
equal(lr[0].get_energy(), lr_ApmB[0].get_energy(), 5.e-24)

# with spin
print '------ with spin'

if not load:
    c_spin = Calculator(xc='PBE', nbands=2, spinpol=True, txt=txt)
    H2.SetCalculator(c_spin)
    c_spin.calculate()
    c_spin.write('H2spin.gpw', 'all')
else:
    c_spin = Calculator('H2spin.gpw', txt=txt)
lr = LrTDDFT(c_spin, xc=xc)
lr.diagonalize()

lr_ApmB = LrTDDFT(c_spin, xc=xc, force_ApmB=True)
lr_ApmB.diagonalize()
print 'lr=', lr
print 'ApmB=', lr_ApmB
equal(lr[0].get_energy(), lr_ApmB[0].get_energy(), 5.e-10)
equal(lr[1].get_energy(), lr_ApmB[1].get_energy(), 5.e-10)

# with spin virtual
print '------ with virtual spin'

lr = LrTDDFT(calc, xc=xc, nspins=2)
lr.diagonalize()

# ApmB
lr_ApmB = LrTDDFT(calc, xc=xc, nspins=2)
lr_ApmB.diagonalize()
print 'lr=', lr
print 'ApmB=', lr_ApmB
equal(lr[0].get_energy(), lr_ApmB[0].get_energy(), 5.e-24)
equal(lr[1].get_energy(), lr_ApmB[1].get_energy(), 5.e-24)
    
