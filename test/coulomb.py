import Numeric as num
from math import pi
from gpaw.coulomb import Coulomb
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world, parallel
from gpaw.utilities.gauss import coordinates
from gpaw.utilities import equal
import time

def test_coulomb(N=2**6, a=20):
    d  = Domain((a, a, a))    # domain object
    Nc = (N, N, N)            # tuple with number of grid point along each axis
    d.set_decomposition(world, N_c=Nc) # decompose domain on processors
    gd = GridDescriptor(d, Nc)# grid-descriptor object
    xyz, r2 = coordinates(gd) # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2 * r) / pi # density of the hydrogen atom
    C = Coulomb(gd)           # coulomb calculator
    
    if parallel:
        C.load('real')
        t0 = time.time()
        print 'Processor %s of %s: %s in %s'%(
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

if __name__ == '__main__':
    analytic = -5 / 16.
    res = test_coulomb(N=48, a=18)
    print 'Units: Bohr and Hartree'
    print '%12s %8s %8s' % ('Method', 'Energy', 'Time')
    print '%12s %2.6f %6s' % ('analytic', analytic, '--')
    for method, et in res.items():
        print '%12s %2.6f %1.7f' % ((method,) + et)

    equal(res['real'][0],         analytic, 1e-1)
    equal(res['recip_gauss'][0],  analytic, 4e-3)
    equal(res['recip_ewald'][0],  analytic, 4e-3)
    equal(res['dual density'][0], res['recip_gauss'][0], 1e-9)
