#Script fails within 10 sec on 4 nodes on niflheim when compiled with icc/ifort

from ase import *
from gpaw  import *
from ase.lattice.tetragonal import SimpleTetragonalFactory

class RutileFactory(SimpleTetragonalFactory):
    u = 0.3061
    bravais_basis = [[1.0, 0.0, 0.0],
                     [0.5, 0.5, 0.5],
                     [  u,   u, 0.0],
                     [1-u, 1-u, 0.0],
                     [0.5 - u, 0.5 + u, 0.5],
                     [0.5 + u, 0.5 - u, 0.5]]
    element_basis = (0, 0, 1, 1, 1, 1)

TiO2 = RutileFactory()

bulk = TiO2(directions=[[0, 0, 1], [1, -1, 0], [1, 1, 0]],
            size =(1, 1, 1),
            symbol=['Ti', 'O'],
            pbc=(1, 1, 1),
            latticeconstant={'a':4.691, 'c':2.975})

calc = GPAW(nbands=60,
            txt='TiO2.txt',
            kpts=(2, 2, 2),
            convergence={'energy': 0.01,
                         'density': 1.0e-3,
                         'eigenstates': 1.0e-3}) 
bulk.set_calculator(calc)

Eref = -105.98934 # The actual reference value
Eref = -121.15661 # The reduced quality GPAW calc of rev 0.6.4293
Etot = bulk.get_potential_energy()
assert abs(Etot - Eref) < 1e-2
