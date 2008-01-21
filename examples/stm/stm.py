# -*- coding: utf-8 -*-
from sys import argv
from ase import *
from gpaw import *

filename = argv[1]

if len(argv) > 2:
    z = float(argv[2])
else:
    z = 2.5
    
atoms = Calculator(filename, txt=None).get_atoms()
stm = STM(atoms, symmetries=[0, 1, 2])
c = stm.get_averaged_current(z)

# Get 2d array of constant current heights:
h = stm.scan(c)

print u'Min: %.2f Å, Max: %.2f Å' % (h.min(), h.max())

import pylab as p
p.contourf(h, 40)
p.hot()
p.colorbar()
p.show()
