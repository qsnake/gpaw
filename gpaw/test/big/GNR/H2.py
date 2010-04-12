from ase import *
from gpaw import *
from ase.structure import graphene_nanoribbon


from ase import *
from gpaw import *
from ase.structure import graphene_nanoribbon

H2 = molecule('H2',cell=(10,10,10))
H2.center()
calc = GPAW()
H2.set_calculator(calc)
dyn = QuasiNewton(H2, trajectory='H2.traj')
dyn.run(fmax=0.05)



