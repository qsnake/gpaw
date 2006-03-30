import Numeric as num
from Numeric import pi
from gridpaw.utilities.complex import real
from FFT import fftnd
from gridpaw.poisson_solver import PoissonSolver
from gridpaw.utilities import DownTheDrain
from gridpaw.utilities.gauss import construct_gauss

def construct_reciprocal(gd):
    """Construct the reciprocal lattice vectors correspoding to the
       grid defined in input grid-descriptor 'gd'
    """
    if gd.domain.comm.size > 1:
        raise RuntimeError('Cannot do parallel FFT')

    # calculate reciprocal lattice vectors
    dim = num.reshape(gd.N_c, (3, 1, 1, 1))
    dk = 2 * pi / gd.domain.cell_c
    dk.shape = (3, 1, 1, 1)
    k = ((num.indices(gd.N_c) + dim / 2)%dim - dim / 2) * dk
    k2 = sum(k**2)
    k2[0,0,0] = 1.0

    # determine N^3
    N3 = gd.N_c[0] * gd.N_c[1] * gd.N_c[2]

    return k2, N3

class Coulomb:
    """Class used to evaluate coulomb integrals"""
    def __init__(self, gd):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gridpaw module
        """        
        self.gd = gd

    def get_single_exchange(self, n, Z=None, ewald=True, method='recip'):
        """Returns exchange energy of input density 'n' defined as
                                              *
                              /    /      n(r)  n(r')
          -1/2 (n | n) = -1/2 | dr | dr'  ------------
	                      /    /        |r - r'|
	   where n could be complex.
        """
        # determine exchange energy of neutral density using specified method
        if method=='real':
            return -0.5 * self.coulomb(n1=n, Z1=Z)
        elif method=='recip':
            if ewald: return -0.5 * self.coulomb_ewald(n1=n)
            else:
                return -0.5 * self.coulomb(n1=n, Z1=Z, space='recip')
        else:
            raise RunTimeError('method name ', method, 'not recognized')
    
    def coulomb_ewald(self, n1, n2=None):
        """Evaluates the coulomb integral:
                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------
	              /    /         |r - r'|
	   where n1 and n2 could be complex.
           Divergence at division by k^2 is avoided by utilizing the Ewald /
           Tuckermann trick, which formaly requires the densities to be
           localized within half of the unit cell.
	"""
        # make sure that k-space related stuff has been initialized
        if not hasattr(self, 'k2'):
            self.k2, self.N3 = construct_reciprocal(self.gd)

        # Make ewald corection
        if not hasattr(self, 'ewald'):
            # cutoff radius
            rc = 0.5 * num.average(self.gd.domain.cell_c)
            # ewald potential: 1 - cos(k rc)
            self.ewald = num.ones(self.gd.N_c) - num.cos(num.sqrt(self.k2) *rc)
            # lim k ->0 ewald / k2 
            self.ewald[0,0,0] = 0.5 * rc**2

        n1k = fftnd(n1)
        if n2 == None: n2k = n1k
        else: n2k = fftnd(n2)

        I = num.conjugate(n1k) * n2k * self.ewald * 4 * pi /(self.k2 * self.N3)

        if n2 == None: return real(self.gd.integrate(I))
        return self.gd.integrate(I)
    
    def coulombNEW(self, n1, n2=None, Z2=None):
        # load gaussian related stuff if needed
        if not hasattr(self, 'ng'):
            self.ng, self.vg = construct_gauss(self.gd)

        # Allocate array for the final integrand
        I = self.gd.new_array()

        # Determine total charges
        if n2 == None: n2 = n1.copy()
        if Z2 == None: Z2 = self.gd.integrate(n2)

        solve = PoissonSolver(self.gd, out=DownTheDrain()).solve
        n2_neutral = n2 - Z2 * self.ng
        solve(I, n2_neutral)
        I += Z2 * self.vg
        I *= num.conjugate(n1)

        return self.gd.integrate(I)

    def coulombNEW2(self, n1, n2=None, Z2=None):
        I = self.gd.new_array()
        if n2 == None: n2 = n1.copy()
        solve = PoissonSolver(self.gd, out=DownTheDrain()).solve
        solve(I, n2, charge=Z2)
        I *= num.conjugate(n1)

        return self.gd.integrate(I)

    def coulomb(self, n1, n2=None, Z1=None, Z2=None, space='real'):
        """Evaluates the coulomb integral:
                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------
	              /    /         |r - r'|
	   where n1 and n2 could be complex.
           Done by removing total charge of n1 and n2 with gaussian density ng
                                                  *          *    *
           (n1|n2) = (n1 - Z1 ng|n2 - Z2 ng) + (Z2 n1 + Z1 n2 - Z1 Z2 ng | ng)

           The evaluation of the integral (n1 - Z1 ng|n2 - Z2 ng) is done in
           either k-space using FFT or in real-space using a poisson solver
           to get the potential of (n2 - Z2 ng).
	"""
        # load gaussian related stuff if needed
        if not hasattr(self, 'ng'):
            self.ng, self.vg = construct_gauss(self.gd)

        # Allocate array for the final integrand
        I = self.gd.new_array()

        # Determine total charges
        if Z1 == None: Z1 = self.gd.integrate(n1)
        if Z2 == None and n2 != None: Z2 = self.gd.integrate(n2)

        # Determine the integrand of the neutral system
        # (n1 - Z1 ng)* int dr'  (n2 - Z2 ng) / |r - r'|
        # in either real or reciprocal space
        if space == 'real':
            solve = PoissonSolver(self.gd, out=DownTheDrain()).solve

            n1_neutral = n1 - Z1 * self.ng
            if n2 == None: n2_neutral = n1_neutral
            else: n2_neutral = n2 - Z2 * self.ng
            solve(I, n2_neutral)
            I *= num.conjugate(n1_neutral)
        elif space == 'recip':
            if not hasattr(self, 'k2'):
                self.k2, self.N3 = construct_reciprocal(self.gd)

            nk1 = fftnd(n1 - Z1 * self.ng)
            if n2 == None:
                I += num.absolute(nk1)**2 * 4 * pi / (self.k2 * self.N3)
            else:
                nk2 = fftnd(n2 - Z2 * self.ng)
                I += num.conjugate(nk1) * nk2 * 4 * pi / (self.k2 * self.N3)
        else:
            raise RunTimeError('Space must be either "real" or "recip" ')

        # add the corrections to the integrand due to neutralization
        if n2 == None:
            I += (2 * real(num.conjugate(Z1) * n1) - abs(Z1)**2 * self.ng) \
                 * self.vg
        else:
            I += (num.conjugate(Z1) * n2 + Z2 * num.conjugate(n1) -
                  num.conjugate(Z1) * Z2 * self.ng) * self.vg
        return self.gd.integrate(I)

def test(parallel=False):
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor
    from gridpaw.utilities.mpi import world
    from gridpaw.utilities.gauss import rSquared

    d  = Domain((20, 20, 20)) # domain object
    N  = 2**6                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    d.set_decomposition(world, N_c=Nc) # decompose domain on processors
    gd = GridDescriptor(d,Nc) # grid-descriptor object
    r2 = rSquared(gd)         # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2*r)/pi     # density of the hydrogen atom

    C = Coulomb(gd)
    if parallel:
        print 'Processor %s of %s: %s'%(d.comm.rank + 1, d.comm.size,
                                        -.5 * C.coulombNEW2(nH, space='real'))
    else:
        print 'Dual density:', -.5 * C.coulomb(nH, nH.copy())
        print 'Realspace   :', -.5 * C.coulomb(nH, space='real')
        print 'Reciprocal  :', -.5 * C.coulomb(nH, space='recip')
        print 'Ewald trick :', -.5 * C.coulomb_ewald(nH)
        print 'NEW         :', -.5 * C.coulombNEW(nH)
        print 'NEW2        :', -.5 * C.coulombNEW2(nH)

if __name__ == '__main__':
    import os, sys
    parallel = False

    if len(sys.argv) > 1:
        parallel = eval(sys.argv[1])
        
    print 'Analytic result:  ', -5/16.
    print 'Numerical results: '

    if parallel:
        if not os.path.exists('hostfile'):
            f = open('hostfile','w')
            print >>f, 'bose'
            print >>f, 'bose'
            f.close()
        job = 'python -c "from coulomb import test; test(True)"'
        cmd = ''
        cmd += 'lamboot -H hostfile;'
        cmd += 'mpirun -nw -x GRIDPAW_PARALLEL=1 C %s' %job
        err = os.system(cmd)
        if err != 0:
            raise RuntimeError
    else:
        test(False)
