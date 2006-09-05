import os
from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.utilities import equal


netcdf = True
try:
    import Scitentiic.IO.NetCDF
except ImportError:
    netcdf = False

if 1:
    h = ListOfAtoms([Atom('H')], cell=(4, 4, 4), periodic=1)
    calc = Calculator(nbands=1, gpts=(16, 16, 16),)# out=None)
    h.SetCalculator(calc)
    e = h.GetPotentialEnergy()
    calc.Write('tmp.gpw')
    if netcdf:
        calc.Write('tmp.nc')

h = Calculator.ReadAtoms('tmp.gpw', out=None)
equal(e, h.GetPotentialEnergy(), 3e-5)

if netcdf:
    h = Calculator.ReadAtoms('tmp.nc', out=None)
    equal(e, h.GetPotentialEnergy(), 3e-5)

if netcdf:
    calc = h.GetCalculator()
    elec_states = calc.GetElectronicStates()
    equal(len(elec_states),1,0)

os.remove('tmp.gpw')
if netcdf:
    os.remove('tmp.nc')
