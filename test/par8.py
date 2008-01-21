#!/usr/bin/env python
import sys
from ase import *
from gpaw import Calculator
from gpaw.mpi import rank

a = 4.0

de1 = 0.0
def f(kpts, n, magmom, pbc, dd):
    global de1
    H = Atoms([Atom('H',(a/2, a/2, a/2), magmom=magmom)],
                    pbc=pbc,
                    cell=(a, a, a))
    
    H.set_calculator(Calculator(nbands=1, gpts=(n, n, n), kpts=kpts,
                               txt=None, tolerance=0.0001,
                               parsize=dd))
    e = H.get_potential_energy()
    H.get_calculator().write('H-par.gpw')
    c = Calculator('H-par.gpw', txt=None)
    H = c.get_atoms()
    de = abs(H.get_potential_energy() - e)
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
                pbc = [bool(p & 2**c) for c in range(3)]
                np = sum([pp > 0 for pp in pbc])
                ok = 1
                for c in range(3):
                    if not pbc[c] and kpts[c] > 1:
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
                    if rank == 0:
                        print kpts, n, magmom, pbc, dd, np,
                        sys.stdout.flush()
                    e0 = f(kpts, n, magmom, pbc, dd)
                    if rank == 0:
                        print e0,
                    if e[np] is not None:
                        de = abs(e0 - e[np])
                        if rank == 0:
                            print de
                        if de > de2:
                            de2 = de
                        assert abs(de) < 0.0007
                    else:
                        if rank == 0:
                            print
                    e[np] = e0

print de1, de2
