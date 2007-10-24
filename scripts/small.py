from gpaw import Calculator
from gpaw.utilities import center
from gpaw.testing.g2 import get_g2

formula = 'H'
cell = [5.5, 5.6, 5.7]
calc_parameters = {'h'          : 0.2,
                   'xc'         : 'PBE',
                   'txt'        : '-',
                   'mix'        : (0.25, 3, 1.0),
                   'lmax'       : 2,
                   'nbands'     : 2,
                   'setups'     : {'Li': 'nocore'},
                   'stencils'   : (2, 'M', 3),
                   'convergence': {'eigenstates': 1e-9},
                   'eigensolver': 'rmm-diis'}

loa = get_g2(formula)
loa.SetUnitCell(cell, fix=True)
center(loa)
loa.SetBoundaryConditions(periodic=False)
loa.SetCalculator(Calculator(**calc_parameters))
loa.GetPotentialEnergy()
