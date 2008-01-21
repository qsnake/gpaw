import numpy as npy
from gpaw import Calculator
from pylab import *

calc = Calculator('Na_harris.gpw', txt=None)
nbands = calc.get_number_of_bands()
kpts = calc.get_ibz_k_points()
nkpts = len(kpts)

eigs = npy.empty((nbands, nkpts))

for k in range(nkpts):
    eigs[:, k] = calc.get_eigenvalues(kpt=k)

eigs -= calc.get_fermi_level()
for n in range(nbands):
    plot(kpts[:, 0], eigs[n], '.m')
#xlabel('Kpoint', fontsize=22)
#ylabel('Eigenvalue', fontsize=22)
show()
