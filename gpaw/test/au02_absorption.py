from ase.io import read
import numpy as np
from gpaw import GPAW, FermiDirac, Mixer
from ase import Atom, Atoms
from ase.visualize import view
from gpaw.response.df import DF

GS = 1
ABS = 1
if GS:
    cluster = Atoms([Atom('Au', (0, 0, 0)),
                     Atom('Au', (0, 0, 2.564))
                     ], pbc=True)
    cluster.set_cell((12.,12.,12.),
                   scale_atoms=False)
    cluster.center()
    #view(cluster)    
    calc=GPAW(xc='RPBE',
              h=0.15,
              mode='lcao',
              basis='dzp',
              occupations=FermiDirac(0.01),
              stencils=(3,3))
    
    cluster.set_calculator(calc)
    cluster.get_potential_energy()
    calc.write('Au02.gpw','all')


if ABS:
    df = DF(calc='Au02.gpw', 
            q=np.array([0.0, 0.0, 0.00001]), 
            w=np.linspace(0,14,141),
            eta=0.1,
            ecut=10,
            optical_limit=True,
            kcommsize=4)              

    df.get_absorption_spectrum()             

    d = np.loadtxt('Absorption.dat')
    wpeak = 2.5 # eV
    Nw = 25
    if d[Nw, 4] > d[Nw-1, 4] and d[Nw, 4] > d[Nw+1, 4]:
        pass
    else:
        raise ValueError('Plasmon peak not correct ! ')
    
    if np.abs(d[Nw, 4] - 0.25788817927) > 5e-5:
        print d[Nw, 0], d[Nw, 4]
        raise ValueError('Please check spectrum strength ! ')
    

