import os
from gridpaw import Calculator
from ASE import Atom, ListOfAtoms
from gridpaw.utilities import equal


if 1:
    h = ListOfAtoms([Atom('H')], cell=(4, 4, 4), periodic=1)
    calc = Calculator(nbands=1, gpts=(16, 16, 16),)# out=None)
    h.SetCalculator(calc)
    e = h.GetPotentialEnergy()
    calc.Write('tmp.nc')
h = Calculator.ReadAtoms('tmp.nc', out=None)
equal(e, h.GetPotentialEnergy(), 3e-5)

calc = h.GetCalculator()
elec_states = calc.GetElectronicStates()
equal(len(elec_states),1,0)

os.remove('tmp.nc')
