from ase import *
from gpaw import *
from gpaw.lrtddft import *

from gpaw.pes.dos import DOSPES
from gpaw.pes.tddft import TDDFTPES

txt='/dev/null'
R=0.7 # approx. experimental bond length
a = 3.0
c = 3.0
h = .3
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

H2_plus = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

calc = GPAW(xc='PBE', nbands=1, h=h, spinpol=True, txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()


calc_plus = GPAW(xc='PBE', nbands=2, h=h, spinpol=True, txt=txt)
calc_plus.set(charge=+1)
H2_plus.set_calculator(calc_plus)
H2_plus.get_potential_energy()

xc='LDA'

lr = LrTDDFT(calc_plus, xc=xc)

pes=DOSPES(calc, calc_plus)
pes.save_folded_pes(filename=txt, folding=None)

pes=TDDFTPES(calc, lr)
pes.save_folded_pes(filename=txt, folding='Gauss')
