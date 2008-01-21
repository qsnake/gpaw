import numpy as npy
from math import pi
from gpaw.coulomb import Coulomb
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world, parallel
from gpaw.utilities.gauss import coordinates
from gpaw.utilities import equal
import time

def test_coulomb(N=2**6, a=20):
    d  = Domain((a, a, a),
            pbc=(0,0,0)) # domain object
    Nc = (N, N, N)            # Number of grid point
    d.set_decomposition(world, N_c=Nc) # decompose domain on processors
    gd = GridDescriptor(d, Nc)# grid-descriptor object
    xyz, r2 = coordinates(gd) # matrix with the square of the radial coordinate
    r  = npy.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = npy.exp(-2 * r) / pi # density of the hydrogen atom
    C = Coulomb(gd)           # coulomb calculator
    
    if parallel:
        C.load('real')
        t0 = time.time()
        print 'Processor %s of %s: %s Ha in %s sec'%(
            d.comm.rank + 1,
            d.comm.size,
            -.5 * C.coulomb(nH, method='real'),
            time.time() - t0)
        return
    else:
        C.load('recip_ewald')
        C.load('recip_gauss')
        C.load('real')
        test = {}
        t0 = time.time()
        test['dual density'] = (-.5 * C.coulomb(nH, nH.copy()),
                                time.time() - t0)
        for method in ('real', 'recip_gauss', 'recip_ewald'):
            t0 = time.time()
            test[method] = (-.5 * C.coulomb(nH, method=method),
                            time.time() - t0)
        return test

analytic = -5 / 16.
res = test_coulomb(N=48, a=15)
if not parallel:
    print 'Units: Bohr and Hartree'
    print '%12s %8s %8s' % ('Method', 'Energy', 'Time')
    print '%12s %2.6f %6s' % ('analytic', analytic, '--')
    for method, et in res.items():
        print '%12s %2.6f %1.7f' % ((method,) + et)

    equal(res['real'][0],         analytic, 6e-3)
    equal(res['recip_gauss'][0],  analytic, 6e-3)
    equal(res['recip_ewald'][0],  analytic, 2e-2)
    equal(res['dual density'][0], res['recip_gauss'][0], 1e-9)


# mpirun -np 2 python coulomb.py --gpaw-parallel --gpaw-debug
