from gpaw import GPAW
from ase.data.molecules import molecule

formula = 'H'
cell = [5.5, 5.6, 5.7]
calc_parameters = {'h'          : 0.2,
                   'xc'         : 'PBE',
                   'txt'        : '-',
                   'lmax'       : 2,
                   'nbands'     : 2,
                   'setups'     : {'Li': 'nocore'},
                   'convergence': {'eigenstates': 1e-9},
                   'eigensolver': 'rmm-diis'}

loa = molecule(formula, cell=cell, calculator=GPAW(**calc_parameters))
loa.center()
loa.get_potential_energy()
