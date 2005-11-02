import Numeric as num
from math import pi

class ExxSingle:
    '''Class used to calculate the exchange energy of given
    single orbital electron density'''
    
    def __init__(self, gd):
        '''Class should be initialized with a grid_descriptor 'gd' from
        the gridpaw module'''       
        self.gd = gd

        # determine r^2 and r matrices
        r2 = rSquared(gd)
        r  = num.sqrt(r2)

        # construct gauss density, potential, and self energy for Z=1
        # 'width' of gaussian distribution
        a                = 22./min(gd.domain.cell_i)**2
        # gaussian density for Z=1
        self.ng1         = num.exp(-a*r2)*(a/pi)**(1.5)
        # gaussian potential for Z=1
        self.vgauss1     = erf3D(num.sqrt(a)*r)/r
        # gaussian self energy for Z=1
        self.EGaussSelf1 = -num.sqrt(a/2/pi)

        # calculate reciprocal lattice vectors
        dim = num.array(gd.N_i,typecode=num.Int);
        dim = num.reshape(dim,(3,1,1,1))
        dk  = 2*pi / num.array(gd.domain.cell_i,typecode=num.Float);
        dk  = num.reshape(dk,(3, 1, 1, 1)) 
        k   = ((num.indices(self.gd.N_i)+dim/2)%dim - dim/2)*dk
        self.k2 = 1.0*sum(k**2); self.k2[0,0,0]=1.0

        # determine N^3
        self.N3 = self.gd.N_i[0]*self.gd.N_i[1]*self.gd.N_i[2]

    def getExchangeEnergy(self,n, method='recip'):
        '''Returns exchange energy of input density 'n' '''

        # make density charge neutral, and get energy correction
        Ecorr = self.neutralize(n)

        # determine exchange energy of neutral density using specified method
        if method=='real':
            from gridpaw.poisson_solver import PoissonSolver
            solver = PoissonSolver(self.gd)
            v = self.gd.array()
            solver.solve(v,n)
            exx = -0.5*(v*n).sum()*self.gd.dv
        elif method=='recip':
            from numarray.fft import fftnd
            nk = fftnd(n)
            exx = -0.5*self.gd.integrate(num.absolute(nk)**2*4*pi/self.k2)/(self.N3)
        else:
            print 'method name ', method, 'not recognized'

        # return resulting exchange energy
        return exx+Ecorr
    
    def neutralize(self, n):
        '''Method for neutralizing input density 'n' with nonzero total
        charge. Returns energy correction caused by making 'n' neutral'''
        
        Z = self.gd.integrate(n)
        #print 'Total charge before neutralize: ', Z
        
        if Z<1e-8: return 0
        else:
            # construct gauss density array
            ng = Z*self.ng1 # gaussian density
            
            # calculate energy corrections
            EGaussN    = -0.5*self.gd.integrate(n*Z*self.vgauss1)
            EGaussSelf = Z**2*self.EGaussSelf1
            
            # neutralize density
            n -= ng

            # determine correctional energy contribution due to neutralization
            Ecorr = - EGaussSelf + 2 * EGaussN
            return Ecorr

''' AUXHILLIARY FUNCTIONS... should be moved to Utillities module'''

def rSquared(gd):
    I=num.indices(gd.N_i)
    dr = num.reshape(gd.h_i,(3,1,1,1))
    r0 = -0.5*num.reshape(gd.domain.cell_i,(3,1,1,1))
    r0 = num.ones(I.shape)*r0
    r2 = num.sum((r0+I*dr)**2)

    # remove zero at origin
    middle = gd.N_i/2.
    if num.alltrue(middle==num.floor(middle)):
        z=middle.astype(int)
        r2[z[0],z[1],z[2]]=1e-12
    # return r squared matrix
    return r2

def erf3D(M):
    from gridpaw.utilities import erf
    dim = M.shape
    res = num.zeros(dim,num.Float)
    for k in range(dim[0]):
        for l in range(dim[1]):
            for m in range(dim[2]):
                res[k,l,m] = erf(M[k,l,m])
    return res
    
def packNEW(M2):
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] =  M2[r, c] + num.conjugate(M2[c,r])
            p += 1
    assert p == len(M)
    return M
