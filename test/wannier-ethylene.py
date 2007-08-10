""" gpaw wannier example for ethylene, 
    corresponding to the ASE Wannier tutorial. 
"""
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import equal

if 1:
    a = 6.0  # Size of unit cell (Angstrom)

    ethylene = ListOfAtoms([
                       Atom('H', (-1.235, 0.936 , 0 ),tag=0),
                       Atom('H', ( 1.235,-0.936 , 0 ),tag=1),
                       Atom('H', ( 1.235, 0.936 , 0 ),tag=1),
                       Atom('H', (-1.235,-0.936 , 0 ),tag=1),
                       Atom('C', ( 0.660, 0.000 , 0 ),tag=1),
                       Atom('C', (-0.660, 0.000 , 0 ),tag=1)],
                       cell=(a,a,a), periodic=True)


    # display to the center of the cell
    pos = ethylene.GetCartesianPositions() 
    pos += a/2. 
    ethylene.SetCartesianPositions(pos)

    calc = Calculator(nbands=8, h=0.20, tolerance=0.001)
    ethylene.SetCalculator(calc)
    print ethylene.GetPotentialEnergy()
    calc.write('ethylene.gpw', 'all')

try:
    import Scientific.IO.NetCDF
except ImportError:
    print 'This test needs Scientific.IO.NetCDF'
else:
    from ASE.Utilities.Wannier import Wannier

    ethylene = Calculator('ethylene.gpw').get_atoms()
    print ethylene.GetPotentialEnergy()
    wannier = Wannier(numberofwannier=6, calculator=ethylene.GetCalculator())
    wannier.Localize()

    value = wannier.GetFunctionalValue() 
    equal(13.2806, value, 0.015)

    for w in wannier.GetCenters():
        print w['radius'], w['pos']

    ethylene.extend(wannier.GetCentersAsAtoms())

    for n in range(1): 
        wannier.WriteCube(n,"ethylene%d.cube"%n)


