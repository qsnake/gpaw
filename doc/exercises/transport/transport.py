from ase import *
from gpaw import GPAW
from ase.transport import Calculator


atoms = Atoms(...)

calc_dft = GPAW(h=0.2,eigensolver='lcao',basis='sz')
atoms.set_calculator(calc)

atoms_left = range(4)
atoms_right = range(-4,0)
cell_left =(4*2.41,7,7)
cell_right = cell_left

calc_tran = Calculator(calc_dft,
                       pl_atoms=(atoms_left,atoms_right),
                       lead_cells=(left_cell,right_cell)
                       energies=npy.arange(-7,3,0.025))

calc.update_hamiltonian()
T = calc.get_transmission()

