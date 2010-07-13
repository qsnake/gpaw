from gpaw import GPAW
from ase.structure import bulk
from ase.dft.kpoints import ibz_points, get_bandpath
import numpy as np

si = bulk('Si', 'diamond', a=5.459)

if 1:
    k = 6
    calc = GPAW(kpts=(k, k, k),
                xc='PBE')
    si.set_calculator(calc)
    e = si.get_potential_energy()
    efermi = calc.get_fermi_level()
    calc.write('Si')
else:
    efermi = GPAW('Si', txt=None).get_fermi_level()

points = ibz_points['fcc']
G = points['Gamma']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
kpts, x, X = get_bandpath([W, L, G, X, W, K], si.cell)
print len(kpts), len(x), len(X)
point_names = ['W', 'L', '\Gamma', 'X', 'W', 'K']

if 1:
    calc = GPAW('Si',
                kpts=kpts,
                fixdensity=True,
                usesymm=None,#False,
                basis='dzp',
                convergence=dict(nbands=8))
    e = calc.get_atoms().get_potential_energy()
    calc.write('Sibs')

calc = GPAW('Sibs', txt=None)
import matplotlib.pyplot as plt
e = np.array([calc.get_eigenvalues(k) for k in range(len(kpts))])
e -= efermi
emin = e.min() - 1
emax = e.max() + 1

for n in range(8):
    plt.plot(x, e[:, n])
for p in X:
    plt.plot([p, p], [emin, emax], 'k-')
plt.xticks(X, ['$%s$' % n for n in point_names])
plt.axis(xmin=0, xmax=X[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')
plt.ylabel('Energy (eV)')
plt.title('PBE bandstructure of Silicon')
plt.savefig('Si.png')
plt.show()
