#!/usr/bin/env python
import sys
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.mpi import rank

a = 4.0

def f(n, magmom, periodic, dd):
    H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=magmom)],
                    periodic=periodic,
                    cell=(a, a, a))
    
    H.SetCalculator(Calculator(nbands=1, gpts=(n, n, n),
                               txt=None, tolerance=0.001,
                               parsize=dd))
    e = H.GetPotentialEnergy()
    H.GetCalculator().write('H-par.gpw')
    H = Calculator('H-par.gpw', txt=None).get_atoms()
    assert e == H.GetPotentialEnergy()
    return e
    
for n in [24]:
    for magmom in [0, 1]:
        e = [None, None, None, None]
        for p in range(8):
            periodic = [p & 2**c for c in range(3)]
            np = sum([pp > 0 for pp in periodic])
            if magmom == 0:
                d = [(1,2,3),(1,3,2),(2,3,1),(2,1,3),(3,1,2),(3,2,1)]
            else:
                d = [(1,1,3),(1,3,1),(3,1,1)]
            for dd in d:
                if rank == 0:
                    print n, magmom, periodic, dd, np,
                    sys.stdout.flush()
                e0 = f(n, magmom, periodic, dd)
                if rank == 0:
                    print e0,
                if e[np] is not None:
                    de = e0 - e[np]
                    if rank == 0:
                        print de
                    assert abs(de) < 0.0008
                else:
                    if rank == 0:
                        print
                e[np] = e0
