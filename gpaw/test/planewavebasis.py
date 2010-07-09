from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.wavefunctions.pw import PW
from gpaw.mpi import world

if world.size == 1:
    a = Atoms('H', pbc=1)
    a.center(vacuum=2)
    a.set_calculator(
        GPAW(mode=PW(200),
             eigensolver=RMM_DIIS(),
             occupations=FermiDirac(0.1)))
    a.get_potential_energy()
