from ase import *
from gpaw import *

unitcell = array([6.5, 6.6, 9.])
gridrefinement = 2
calc = GPAW(xc='PBE')

for formula in ('Na', 'Cl', 'NaCl',):
    calc = GPAW(xc='PBE', nbands=-5, txt=formula + '.txt')
    if formula == 'Cl': calc.set(hund=True)
    sys = molecule(formula, cell=unitcell, calculator=calc)
    sys.center()
    sys.get_potential_energy()

    # Get densities
    nt = calc.get_pseudo_density()
    n = calc.get_all_electron_density(gridrefinement=gridrefinement)

    # Get integrated values
    dv = product(calc.get_grid_spacings())
    It = nt.sum() * dv
    I = n.sum() * dv / gridrefinement**3

    print '%-4s %4.2f %5.2f' % (formula, It, I)
