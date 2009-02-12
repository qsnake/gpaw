import os
import numpy as np
#numpy.seterr(all='raise')

from ase import *
from ase.io.plt import write_plt
from gpaw.utilities import equal
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.io.plt import read_plt

txt='-'
txt='/dev/null'

load = False
#load = True

R=0.7 # approx. experimental bond length
a = 4
c = 4
H2 = Cluster([Atom('H', (a/2,a/2,(c-R)/2)),
              Atom('H', (a/2,a/2,(c+R)/2))],
             cell=(a,a,c))
H2.rotate([1.,1.,1.])
##H2.write('H2.xyz')

fname = 'H2.gpw'
if (not load) or (not os.path.exists(fname)):
    calc = GPAW(xc='PBE', nbands=2, spinpol=False, txt=txt)
    H2.set_calculator(calc)
    H2.get_potential_energy()
    if load:
        calc.write(fname, 'all')
else:
    calc = GPAW(fname, txt=txt)
    calc.initialize_wave_functions()

import gpaw.mpi as mpi
fname = 'aed.plt'
cell = calc.get_atoms().get_cell()
#aed = calc.get_all_electron_density(1, pad=False)
aed = calc.get_pseudo_density(1, pad=False)
#aed = calc.gd.collect(aed)

if mpi.size == 1:
    data_org = [cell, aed, np.array([0., 0., 0.])]
    write_plt(fname, calc.get_atoms(), aed)
    
    # check if read arrays match the written ones
    data = read_plt(fname)
    ##print data[0], data[2]
    for d, do in zip(data, data_org):
        dd2 = (d - do)**2
        norm = dd2.sum() 
        print norm
        assert(norm < 1e-10)
