#!/usr/bin/env python
import sys
from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 4.0

def f(n, magmom, periodic, dd):
    H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=magmom)],
                    periodic=periodic,
                    cell=(a, a, a))
    
    H.SetCalculator(Calculator(nbands=1, gpts=(n, n, n),
                               out=None, maxiter=1,
                               parsize=dd, hosts=8))
    e = H.GetPotentialEnergy()
    H.GetCalculator().Write('H-par.gpw')
    H = Calculator.ReadAtoms('H-par.gpw', out=None)
    assert e == H.GetPotentialEnergy()
    return e
    
for n in [16, 20, 24, 32]:
    for magmom in [0, 1]:
        e = [None, None, None, None]
        for p in range(8):
            periodic = [p & 2**c for c in range(3)]
            np = sum([pp > 0 for pp in periodic])
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
                print n, magmom, periodic, dd, np,
                sys.stdout.flush()
                e0 = f(n, magmom, periodic, dd)
                print e0,
                if e[np] is not None:
                    de = e0 - e[np]
                    print de
                    assert abs(de) < 0.0007
                else:
                    print
                e[np] = e0
