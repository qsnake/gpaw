import os
from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.utilities import equal


netcdf = True
try:
    import Scientific.IO.NetCDF
except ImportError:
    netcdf = False

if 1:
    h = ListOfAtoms([Atom('H')], cell=(4, 4, 4), periodic=1)
    calc = Calculator(nbands=1, gpts=(16, 16, 16),)# out=None)
    h.SetCalculator(calc)
    e = h.GetPotentialEnergy()
    calc.write('tmp.gpw')
    if netcdf:
        calc.write('tmp.nc', 'all')

h = Calculator('tmp.gpw', txt=None)
equal(e, h.GetPotentialEnergy(), 3e-5)

if netcdf:
    h = Calculator('tmp.nc', txt=None)
    equal(e, h.GetPotentialEnergy(), 3e-5)

if netcdf:
    calc = h
    elec_states = calc.GetElectronicStates()
    equal(len(elec_states),1,0)

os.remove('tmp.gpw')
if netcdf:
    os.remove('tmp.nc')
    os.remove('tmp27.nc')
