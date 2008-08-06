from ase import *
from gpaw import Calculator, Mixer
from ase.transport import Calculator as Calculator2

calc_dft = Calculator(h=0.2,eigensolver='lcao',basis='sz')

pl_atoms_left = range(4)
pl_atoms_right = range(-4,0)
calc = Calculator2(calc_dft,
                   pl_atoms=[pl_atoms_left,pl_atoms_right],
                   energies=npy.arange(-7,3,0.025))

calc.update_hamiltonian()
T = calc.get_transmission()

