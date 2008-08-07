from gpaw import GPAW
from pylab import *

calc = GPAW('Na_harris.gpw', txt=None)
nbands = calc.get_number_of_bands()
kpts = calc.get_ibz_k_points()
nkpts = len(kpts)

eigs = empty((nbands, nkpts), float)

for k in range(nkpts):
    eigs[:, k] = calc.get_eigenvalues(kpt=k)

# Subtract Fermi level from the self-consistent calculation
eigs -= GPAW('Na_sc.gpw', txt=None).get_fermi_level()
for n in range(nbands):
    plot(kpts[:, 0], eigs[n], '.m')
show()
