from gpaw import Calculator
from ASE import ListOfAtoms, Atom

a = 4.00
d = a / 2**0.5
z = 1.1
b = 1.5

slab = ListOfAtoms([Atom('Al', (0, 0, 0)),
                    Atom('Al', (a, 0, 0)),
                    Atom('Al', (a/2, d/2, -d/2)),
                    Atom('Al', (3*a/2, d/2, -d/2)),
                    Atom('Al', (0, 0, -d)),
                    Atom('Al', (a, 0, -d)),
                    Atom('Al', (a/2, d/2, -3*d/2)),
                    Atom('Al', (3*a/2, d/2, -3*d/2)),
                    Atom('Al', (0, 0, -2*d)),
                    Atom('Al', (a, 0, -2*d)),
                    Atom('H', (a/2-b/2, 0, z)),
                    Atom('H', (a/2+b/2, 0, z))],
                   cell=(2*a, d, 5*d), periodic=(1, 1, 1))
slab.SetCalculator(Calculator(h=0.25, nbands=28, kpts=(2, 6, 1),
                              tolerance=1e-5))
e = slab.GetPotentialEnergy()

for i in range(1):
    b += 0.02
    slab[-2].SetCartesianPosition((a/2-b/2, 0, z))
    slab[-1].SetCartesianPosition((a/2+b/2, 0, z))
    e = slab.GetPotentialEnergy()
