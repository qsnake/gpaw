import Numeric as num
from math import pi

class ExxSingle:
    """Class used to calculate the exchange energy of given
    single orbital electron density"""
    
    def __init__(self, gd):
        """Class should be initialized with a grid_descriptor 'gd' from
        the gridpaw module"""
        
        self.gd = gd

        # determine r^2 and r matrices
        r2 = rSquared(gd)
        r  = num.sqrt(r2)

        # 'width' of gaussian distribution
        # a=a0/... => ng~exp(-a0/4) on the boundary of the domain
        a = 25./min(gd.domain.cell_c)**2

        # gaussian density for Z=1
        self.ng1 = num.exp(-a*r2)*(a/pi)**(1.5)

        # gaussian potential for Z=1
        self.vgauss1 = erf3D(num.sqrt(a)*r)/r

        # gaussian self energy for Z=1
        self.EGaussSelf1 = -num.sqrt(a/2/pi)

        # calculate reciprocal lattice vectors
        dim = gd.N_c.copy()
        dim.shape = (3,1,1,1)
        dk = 2*pi / gd.domain.cell_c
        dk.shape = (3, 1, 1, 1)
        k = ((num.indices(self.gd.N_c)+dim/2)%dim - dim/2)*dk
        self.k2 = sum(k**2)
        self.k2[0,0,0] = 1.0

        # determine N^3
        self.N3 = self.gd.N_c[0]*self.gd.N_c[1]*self.gd.N_c[2]

    def get_single_exchange(self, n, method = 'recip', Z = None):
        """Returns exchange energy of input density 'n' """

        # make density charge neutral, and get energy correction
        Ecorr = self.neutralize(n, Z)

        # determine exchange energy of neutral density using specified method
        if method=='real':
            from gridpaw.poisson_solver import PoissonSolver
            solver = PoissonSolver(self.gd)
            v = self.gd.new_array()
            solver.solve(v,n)
            exx = -0.5*self.gd.integrate(v*n)
        elif method=='recip':
            from FFT import fftnd
            nk = fftnd(n)
            exx = -0.5*self.gd.integrate(num.absolute(nk)**2*4*pi/self.k2)\
                  /(self.N3)
        else: raise RunTimeError('method name ', method, 'not recognized')

        # return resulting exchange energy
        return exx + Ecorr
    
    def neutralize(self, n, Z):
        """Method for neutralizing input density 'n' with nonzero total
        charge. Returns energy correction caused by making 'n' neutral"""

        if Z==None: Z = self.gd.integrate(n)
        
        if Z<1e-8: return 0
        else:
            # construct gauss density array
            ng = Z*self.ng1 # gaussian density
            
            # calculate energy corrections
            EGaussN    = -0.5 * Z * self.gd.integrate(n*self.vgauss1)
            EGaussSelf = Z**2 * self.EGaussSelf1
            
            # neutralize density
            n -= ng

            # determine correctional energy contribution due to neutralization
            Ecorr = - EGaussSelf + 2 * EGaussN
            return Ecorr

def get_exact_exchange(wf, nuclei, gd, decompose = False):
    """Calculate exact exchange energy"""
    
    from gridpaw.localized_functions import create_localized_functions

    # ensure gamma point calculation
    assert wf.typecode == num.Float

    # construct gauss functions
    gt_aL=[]
    for nucleus in nuclei:
        gSpline = nucleus.setup.get_shape_functions()
        gt_aL.append(create_localized_functions(gSpline, gd,
                                                nucleus.spos_c))

    # load single exchange calculator
    exx_single = ExxSingle(gd).get_single_exchange

    # calculate exact exchange
    exxs = exxa = 0.0
    for spin in range(wf.nspins):
        for n in range(wf.nbands):
            for m in range(wf.nbands):
                # calculate joint occupation number
                fnm = (wf.kpt_u[spin].f_n[n] *
                       wf.kpt_u[spin].f_n[m]) * wf.nspins / 2.
                
                # determine current exchange density
                n_G = wf.kpt_u[spin].psit_nG[m]*\
                      wf.kpt_u[spin].psit_nG[n]
                
                for a, nucleus in enumerate(nuclei):
                    # generate density matrix
                    Pm_i = nucleus.P_uni[spin,m]
                    Pn_i = nucleus.P_uni[spin,n]
                    D_ii = num.outerproduct(Pm_i,Pn_i)
                    D_p  = packNEW(D_ii)

                    # add compensation charges to exchange density
                    Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    gt_aL[a].add(n_G, Q_L)

                    # add atomic contribution to exchange energy
                    C_pp  = nucleus.setup.M_pp
                    exxa -= fnm*num.dot(D_p, num.dot(C_pp, D_p))
                    
                # determine total charge of exchange density
                if n == m: Z = 1
                else: Z = 0

                # add the nm contribution to exchange energy
                exxs += fnm * exx_single(n_G, Z = Z)

        # Determine the valence-core and core-core contributions
        ExxValCore = ExxCore = 0.0
        import pickle
        from gridpaw import home

        # add val-core and core-core contribution for each nucleus
        for nucleus in nuclei:
            try:
                # load data from file
                filename = home + '/trunk/gridpaw/atom/VC/' + \
                           nucleus.setup.symbol + '.' + \
                           nucleus.setup.xcname + '.VC'
                f = open(filename,'r')
                Exxc, X_p = pickle.load(f)

                # add core-core contribution from current nucleus
                ExxCore += Exxc
                
                # add val-core contribution from current nucleus
                D_p = nucleus.D_sp[0]

##                Hack for atomic Neon                
##                 D_p =  num.zeros((13,13))
##                 D_p[0,0] = D_p[2,2] = D_p[3,3] = D_p[4,4] = 2.
##                 D_p = packNEW(D_p)    

##                Hack for atomic Magnesium
##                 D_p =  num.zeros((5,5))
##                 D_p[0,0] = 2.
##                 D_p = packNEW(D_p)    

##                Hack for atomic Oxygen
##                 D_ii =  num.zeros((13,13),num.Float)
##                 D_ii[0,0] = 2.
##                 D_ii[2,2] = D_ii[3,3] = D_ii[4,4] = 4/3.
##                 D_p = packNEW(D_ii)
##                 D_p[0],D_p[13],D_p[25],D_p[36],D_p[46]
                
##                 print D_p
##                 from gridpaw.utilities import unpack
##                 print unpack(D_p)
                
                ExxValCore += -num.dot(D_p,X_p)
               
            except IOError:
                print 'WARNING: no VC file for', nucleus.setup.symbol
                print 'file', filename, 'missing'
                print 'Exact exchange energy may be incorrect'
                
    # add all contributions, to get total exchange energy
    exx = exxs + exxa + ExxValCore + ExxCore

    # return result
    if decompose:
        return num.array([exx, exxs + exxa, ExxValCore, ExxCore])
    else:
        return exx

def atomic_exact_exchange(atom, type = 'all'):
    """Returns the exact exchange energy of the atom defined by the
    instantiated AllElectron object 'atom' """

    # get Gaunt coefficients
    from gridpaw.gaunt import gaunt

    # get Hartree potential calculator
    from gridpaw.setup import Hartree

    # maximum angular momentum
    Lmax = 2 * max(atom.l_j) + 1

    # number of valence, Nj, and core, Njcore, orbitals
    Nj     = len(atom.n_j)
    Njcore = coreStates(atom.symbol, atom.n_j, atom.l_j, atom.f_j)

    # determine relevant states for chosen type of exchange contribution
    if type == 'all': nstates = mstates = range(Nj)
    elif type == 'val-val': nstates = mstates = range(Njcore,Nj)
    elif type == 'core-core': nstates = mstates = range(Njcore)
    elif type == 'val-core':
        nstates = range(Njcore,Nj)
        mstates = range(Njcore)
    else: raise RunTimeError('Unknown type of exchange: ', type)

    # diagonal +-1 elements in Hartree matrix
    a1_g  = 1.0 - 0.5 * (atom.d2gdr2 * atom.dr**2)[1:]
    a2_lg = -2.0 * num.ones((Lmax, atom.N - 1), num.Float)
    x_g   = (atom.dr[1:] / atom.r[1:])**2
    for l in range(1, Lmax):
        a2_lg[l] -= l * (l + 1) * x_g
    a3_g = 1.0 + 0.5 * (atom.d2gdr2 * atom.dr**2)[1:]

    # initialize potential calculator (returns v*r^2*dr/dg)
    H = Hartree(a1_g, a2_lg, a3_g, atom.r, atom.dr).solve

    # do actual calculation of exchange contribution
    Exx = 0.0
    for j1 in nstates:
        # angular momentum of first state
        l1 = atom.l_j[j1]

        for j2 in mstates:
            # angular momentum of second state
            l2 = atom.l_j[j2]

            # joint occupation number
            f12 = .5 * atom.f_j[j1]/(2. * l1 + 1) * \
                       atom.f_j[j2]/(2. * l2 + 1)

            # electron density
            n = atom.u_j[j1]*atom.u_j[j2]
            n[1:] /= atom.r[1:]**2

            # determine potential times r^2 times length element dr/dg
            vr2dr = num.zeros(atom.N, num.Float)

            # L summation
            for l in range(l1 + l2 + 1):
                # get potential for current l-value
                vr2drl = H(n, l)

                # take all m1 m2 and m values of Gaunt matrix of the form
                # G(L1,L2,L) where L = {l,m}
                G2 = gaunt[l1**2:(l1+1)**2, l2**2:(l2+1)**2,\
                           l**2:(l+1)**2]**2

                # add to total potential
                vr2dr += vr2drl * num.sum(G2.copy().flat)

            # add to total exchange the contribution from current two states
##             print -.5 * num.dot(n,vr2dr) # * 27.211395655517311
##             print f12, j1, j2
            Exx += -.5 * f12 * num.dot(n,vr2dr)

    # double energy if mixed contribution
    if type == 'val-core': Exx *= 2.

    # return exchange energy
    return Exx

def constructX(gen):
    """Construct the X_p^a matrix for the given atom"""

    # get Gaunt coefficients
    from gridpaw.gaunt import gaunt

    # get Hartree potential calculator
    from gridpaw.setup import Hartree

    # maximum angular momentum
    Lmax = 2 * max(gen.l_j,gen.lmax) + 1

    # unpack valence states * r:
    uv_j = []
    lv_j = []
    Nvi  = 0 
    for l, u_n in enumerate(gen.u_ln):
        for u in u_n:
            uv_j.append(u) # unpacked valence state array
            lv_j.append(l) # corresponding angular momenta
            Nvi += 2*l+1   # number of valence states (including m)

    # number of core and valence orbitals (j only, i.e. not m-number)
    Njcore = gen.njcore
    Njval  = len(lv_j)

    # core states * r:
    uc_j = gen.u_j[:Njcore]

    # diagonal +-1 elements in Hartree matrix
    a1_g  = 1.0 - 0.5 * (gen.d2gdr2 * gen.dr**2)[1:]
    a2_lg = -2.0 * num.ones((Lmax, gen.N - 1), num.Float)
    x_g   = ((gen.dr / gen.r)**2)[1:]
    for l in range(1, Lmax):
        a2_lg[l] -= l * (l + 1) * x_g
    a3_g = 1.0 + 0.5 * (gen.d2gdr2 * gen.dr**2)[1:]

    # initialize potential calculator (returns v*r^2*dr/dg)
    H = Hartree(a1_g, a2_lg, a3_g, gen.r, gen.dr).solve

    # initialize X_ii matrix
    X_ii = num.zeros((Nvi,Nvi), num.Float)

    # sum over core states
    for jc in range(Njcore):
        lc = gen.l_j[jc]

        # sum over first valence state index
        i1 = 0
        for jv1 in range(Njval):
            lv1 = lv_j[jv1] 

            # electron density 1
            n1c = uv_j[jv1]*uc_j[jc]
            n1c[1:] /= gen.r[1:]**2  

            # sum over second valence state index
            i2 = 0
            for jv2 in range(Njval):
                lv2 = lv_j[jv2]
                
                # electron density 2
                n2c = uv_j[jv2]*uc_j[jc]
                n2c[1:] /= gen.r[1:]**2  
            
                # sum expansion in angular momenta
                for l in range(min(lv1,lv2) + lc + 1):
                    # density * potential
                    nv = num.dot(n1c,H(n2c, l))

                    # expansion coefficients
                    A_mm = X_ii[i1:i1 + 2 * lv1 + 1, i2:i2 + 2 * lv2 + 1]
                    for mc in range(2*lc+1):
                        for m in range(2*l+1):
                            G1c = gaunt[lv1**2:(lv1 + 1)**2,
                                        lc**2+mc,l**2 + m]
                            G2c = gaunt[lv2**2:(lv2 + 1)**2,
                                        lc**2+mc,l**2 + m]
                            A_mm += nv * num.outerproduct(G1c,G2c)
                            
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = packNEW(X_ii, symmetric = True)
    return X_p

def coreStates(symbol, n,l,f):
    """method returning the number of core states for given element"""
    
    from gridpaw.atom.configurations import configurations
    from gridpaw.atom.generator import parameters

    try:
        core, rcut = parameters[symbol]
        extra = None
    except ValueError:
        core, rcut, extra = parameters[symbol]
    
    # Parse core string:
    j = 0
    if core.startswith('['):
        a, core = core.split(']')
        core_symbol = a[1:]
        j = len(configurations[core_symbol][1])
        
    while core != '':
        assert n[j] == int(core[0])
        assert l[j] == 'spdf'.find(core[1])
        assert f[j] == 2 * (2 * l[j] + 1)
        j += 1
        core = core[2:]
    Njcore = j

    return Njcore

# AUXHILLIARY FUNCTIONS... should be moved to Utillities module... XXX

def rSquared(gd):
    """constructs and returns a matrix containing the square of the distance
    from the origin which is placed in the center of the box described by the
    given griddescriptor 'gd'. """
    
    I  = num.indices(gd.N_c)
    dr = num.reshape(gd.h_c,(3,1,1,1))
    r0 = -0.5*num.reshape(gd.domain.cell_c,(3,1,1,1))
    r0 = num.ones(I.shape)*r0
    r2 = num.sum((r0+I*dr)**2)

    # remove singularity at origin and replace with small number
    middle = gd.N_c/2.
    if num.alltrue(middle==num.floor(middle)):
        z = middle.astype(int)
        r2[z[0],z[1],z[2]] = 1e-12

    # return r^2 matrix
    return r2

def erf3D(M):
    """return matrix with the value of the error function evaluated for each
    element in input matrix 'M'. """
    
    from gridpaw.utilities import erf
    
    dim = M.shape
    res = num.zeros(dim,num.Float)
    for k in range(dim[0]):
        for l in range(dim[1]):
            for m in range(dim[2]):
                res[k,l,m] = erf(M[k,l,m])
    return res
    
def packNEW(M2, symmetric = False):
    """new pack method"""
    
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] =  M2[r, c] + num.conjugate(M2[c,r])
            p += 1
            if symmetric:
                error = abs(M2[r, c] - num.conjugate(M2[c, r]))
                if error > 1e-6: print 'Error not symmetric by: ', error
    assert p == len(M)
    return M

if __name__ == '__main__':
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor

    d  = Domain((20,20,20))   # domain object
    N  = 2**5                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    gd = GridDescriptor(d,Nc) # grid-descriptor object
    r2 = rSquared(gd)         # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2*r)/pi     # density of the hydrogen atom

    exx = ExxSingle(gd).get_single_exchange(nH, method = 'recip')
    print 'Numerical result: ', exx
    print 'Analytic result:  ', -5/16.
