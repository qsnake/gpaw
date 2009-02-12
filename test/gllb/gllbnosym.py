from gpaw import GPAW
from ase import *
from gpaw.utilities import equal
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

atom = 'Si'
g = Generator(atom, xcname ='GLLB', scalarrel=False,nofiles=True)
g.run(**parameters[atom])

setup_paths.insert(0, '.')

a = 5.404
bulk = Atoms(symbols='Si8',
             positions=[(0, 0, 0),
                        (0, 0.5, 0.5),
                        (0.5, 0, 0.5),
                        (0.5, 0.5, 0),
                        (0.25, 0.25, 0.25),
                        (0.25, 0.75, 0.75),
                        (0.75, 0.25, 0.75),
                        (0.75, 0.75, 0.25)],
             pbc=True)
bulk.set_cell((a, a, a), scale_atoms=True)
n = 20

parameters = { 'gpts':(n,n,n),
               'nbands':8*3,
               'width':0.1,
               'kpts':(2,2,2),
               'usesymm':True,
               'xc':'GLLB'}
        

bulk.set_calculator(GPAW(**parameters))
E = bulk.get_potential_energy()
parameters['usesymm'] = False
bulk.set_calculator(GPAW(**parameters))
E2 = bulk.get_potential_energy()

equal(E, E2, 1e-3)

