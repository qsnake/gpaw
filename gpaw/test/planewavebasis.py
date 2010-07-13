from ase import Atoms
from ase.data.molecules import molecule
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.mpi import world

k = 2
if world.size == 1:
    a = molecule('H', pbc=1, magmoms=[0])
    a.center(vacuum=2)
    a.set_calculator(
        GPAW(mode=PW(250),
             kpts=(k, k, k)))
    a.get_potential_energy()
