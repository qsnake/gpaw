# see changeset 4891
from ase import *
from gpaw import *
from gpaw.mpi import world
from gpaw.test import equal

a = 2.5
H = Atoms('H', cell=[a, a, a], pbc=True)

if world.size >= 3:
    H.set_calculator(GPAW(kpts=[6, 6, 1],
                          spinpol=True,
                          parsize=world.size,
                          txt='H-a.txt'))
    e1 = H.get_potential_energy()
    assert H.get_calculator().wfs.kpt_comm.size == 1

    comm = world.new_communicator(np.array([0, 1, 2]))
    if world.rank < 3:
        H.set_calculator(GPAW(kpts=[6, 6, 1],
                              spinpol=True,
                              communicator=comm,
                              txt='H-b.txt'))
        e2 = H.get_potential_energy()
        assert H.get_calculator().wfs.kpt_comm.size == 3
        equal(e1, e2, 1e-11)
        

