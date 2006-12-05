#!/usr/bin/env python
import sys
from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 4.0

de1 = 0.0
def f(kpts, n, magmom, periodic, dd):
    global de1
    H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=magmom)],
                    periodic=periodic,
                    cell=(a, a, a))
    
    H.SetCalculator(Calculator(nbands=1, gpts=(n, n, n), kpts=kpts,
                               out=None, maxiter=1,
                               parsize=dd, hosts=8))
    e = H.GetPotentialEnergy()
    H.GetCalculator().Write('H-par.gpw')
    H = Calculator.ReadAtoms('H-par.gpw', out=None)
    de = abs(H.GetPotentialEnergy() - e)
    if de > de1:
        de1 = de
    assert de < 1e-15
    return e

de2 = 0.0
for k1 in [1, 2]:
 for k2 in [1, 2]:
  for k3 in [1, 2]:
    kpts = (k1, k2, k3)
    for n in [16, 20, 24, 32]:
        for magmom in [0, 1]:
            e = [None, None, None, None]
            for p in range(8):
                periodic = [bool(p & 2**c) for c in range(3)]
                np = sum([pp > 0 for pp in periodic])
                ok = 1
                for c in range(3):
                    if not periodic[c] and kpts[c] > 1:
                        ok = 0
                        break
                if not ok:
                    continue

                if magmom == 0:
                    if n == 32:
                        d = [(1,2,4),(1,4,2),(2,4,1),(2,1,4),(4,1,2),(4,2,1),
                             (2,2,2)]
                    else:
                        d = [(2,2,2)]
                else:
                    if n == 32:
                        d = [(1,2,2),(2,1,2),(2,2,1),(1,1,4),(1,4,1),(4,1,1)]
                    else:
                        d = [(1,2,2),(2,1,2),(2,2,1)]
                for dd in d:
                    print kpts, n, magmom, periodic, dd, np,
                    sys.stdout.flush()
                    e0 = f(kpts, n, magmom, periodic, dd)
                    print e0,
                    if e[np] is not None:
                        de = abs(e0 - e[np])
                        print de
                        if de > de2:
                            de2 = de
                        assert abs(de) < 0.0007
                    else:
                        print
                    e[np] = e0

print de1, de2
