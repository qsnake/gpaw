import os
from ase import *
from gpaw import GPAW
from gpaw.utilities import equal
from gpaw.lrtddft import LrTDDFT

txt = '-'
#txt = None
load = False

R = 0.7  # approx. experimental bond length
a = 3.0
c = 4.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))
calc = GPAW(xc='PBE', nbands=2, spinpol=False, txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()
calc.write('H2saved.gpw', 'all')

xc = 'LDA'

# this works ---------------------

# without spin
lr = LrTDDFT(calc, xc=xc)

# this fails ---------------------
lr = LrTDDFT(GPAW('H2saved.gpw', txt=txt), xc=xc)
