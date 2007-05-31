# -*- coding: utf-8 -*-
from math import sqrt

import numpy as npy


def NudgedElasticBand(atoms):
    N = atoms.repeat.prod()
    natoms = atoms.natoms // N

    R = atoms.RR[:, :natoms]
    E = atoms.EE
    F = atoms.FF[:, :natoms]
    
    n = atoms.nframes
    Efit = npy.empty((n - 1) * 20 + 1)
    Sfit = npy.empty((n - 1) * 20 + 1)

    s = [0]
    for i in range(n - 1):
        s.append(s[-1] + sqrt(((R[i + 1] - R[i])**2).sum()))

    import pylab
    import matplotlib
    matplotlib.use('GTK')

    pylab.ion()
    x = 2.95
    pylab.figure(figsize=(x * 2.5**0.5, x))

    E -= E[0]
    pylab.plot(s, E, 'o')

    for i in range(n):
        if i == 0:
            d = R[1] - R[0]
            ds = 0.5 * s[1]
        elif i == n - 1:
            d = R[-1] - R[-2]
            ds = 0.5 * (s[-1] - s[-2])
        else:
            d = R[i + 1] - R[i - 1]
            ds = 0.25 * (s[i + 1] - s[i - 1])

        d = d / sqrt((d**2).sum())
        dEds = -(F[i] * d).sum()
        x = npy.linspace(s[i] - ds, s[i] + ds, 21)
        y = E[i] + dEds * (x - s[i])
        pylab.plot(x, y, '-g')

        if i > 0:
            s0 = s[i - 1]
            s1 = s[i]
            x = npy.linspace(s0, s1, 20, endpoint=False)
            c = npy.linalg.solve(npy.array([(1, s0,   s0**2,     s0**3),
                                            (1, s1,   s1**2,     s1**3),
                                            (0,  1,  2 * s0, 3 * s0**2),
                                            (0,  1,  2 * s1, 3 * s1**2)]),
                                 npy.array([E[i - 1], E[i], dEds0, dEds]))
            y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            Sfit[(i - 1) * 20:i * 20] = x
            Efit[(i - 1) * 20:i * 20] = y
        
        dEds0 = dEds

    Sfit[-1] = s[-1]
    Efit[-1] = E[-1]
    pylab.plot(Sfit, Efit, 'k-')
    pylab.xlabel(u'path [Å]')
    pylab.ylabel(u'energy [eV]')
    pylab.title('Maximum: %.3f eV' % max(Efit))
    pylab.show()
