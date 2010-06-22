from sys import argv

import pylab as plt

from ase.dft import STM
from gpaw import restart

filename = argv[1]

if len(argv) > 2:
    z = float(argv[2])
else:
    z = 2.5
    
atoms, calc = restart(filename, txt=None)

stm = STM(atoms, symmetries=[0, 1, 2])
c = stm.get_averaged_current(z)

# Get 2d array of constant current heights:
h = stm.scan(c)

print u'Min: %.2f Ang, Max: %.2f Ang' % (h.min(), h.max())

plt.contourf(h, 40)
plt.hot()
plt.colorbar()
plt.show()
