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

        # 'width' of gaussian distribution
        a = 22./min(gd.domain.cell_c)**2

        # gaussian density for Z=1
        self.ng1 = num.exp(-a*r2)*(a/pi)**(1.5)

        # gaussian potential for Z=1
        self.vgauss1 = erf3D(num.sqrt(a)*r)/r

        # gaussian self energy for Z=1
        self.EGaussSelf1 = -num.sqrt(a/2/pi)

        # calculate reciprocal lattice vectors
        dim = num.array(gd.N_c,typecode=num.Int);
        dim = num.reshape(dim,(3,1,1,1))
        dk  = 2*pi / num.array(gd.domain.cell_c,typecode=num.Float);
        dk  = num.reshape(dk,(3, 1, 1, 1)) 
        k   = ((num.indices(self.gd.N_c)+dim/2)%dim - dim/2)*dk
        self.k2 = 1.0*sum(k**2)
        self.k2[0,0,0] = 1.0

        # determine N^3
        self.N3 = self.gd.N_c[0]*self.gd.N_c[1]*self.gd.N_c[2]

    def getExchangeEnergy(self,n, method='recip', Z='none'):
        '''Returns exchange energy of input density 'n' '''

        # make density charge neutral, and get energy correction
        Ecorr = self.neutralize(n, Z)

        # determine exchange energy of neutral density using specified method
        if method=='real':
            from gridpaw.poisson_solver import PoissonSolver
            solver = PoissonSolver(self.gd)
            v = self.gd.array()
            solver.solve(v,n)
            exx = -0.5*(v*n).sum()*self.gd.dv
        elif method=='recip':
            from FFT import fftnd
            nk = fftnd(n)
            exx = -0.5*self.gd.integrate(num.absolute(nk)**2*4*pi/self.k2)/(self.N3)
        else:
            print 'method name ', method, 'not recognized'

        # return resulting exchange energy
        return exx+Ecorr
    
    def neutralize(self, n, Z):
        '''Method for neutralizing input density 'n' with nonzero total
        charge. Returns energy correction caused by making 'n' neutral'''

        if (Z=='none'):
            Z = self.gd.integrate(n)
        elif (Z.__class__ == int):
            Z=float(Z)
        elif (Z.__class__ != float):
            print 'Error total charge not a number or string "none"!'
        
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

def exactExchange(wf, nuclei, gd):
    '''Calculate exact exchange energy'''
    from gridpaw.localized_functions import create_localized_functions

    # ensure gamma point calculation
    assert (wf.nkpts == 1)

    # construct gauss functions
    gt_aL=[]
    for nucleus in nuclei:
        gSpline = nucleus.setup.get_shape_function()
        gt_aL.append(create_localized_functions(gSpline, gd,
                                                nucleus.spos_c))

    # load single exchange calculator
    exx_single = ExxSingle(gd)

    # calculate exact exchange
    exx = exxa = 0.0
    for spin in range(wf.nspins):
        for n in range(wf.nbands):
            for m in range(wf.nbands):
                # calculate joint occupation number
                fnm = (wf.kpt_u[spin].f_n[n] *
                       wf.kpt_u[spin].f_n[m]) * wf.nspins / 2

                # determine current exchange density
                n_G = wf.kpt_u[spin].psit_nG[m]*\
                      wf.kpt_u[spin].psit_nG[n]
                for a, nucleus in enumerate(nuclei):
                    # generate density matrix
                    Pm_i = nucleus.P_uni[spin,m]
                    Pn_i = nucleus.P_uni[spin,n]
                    D_ii = num.outerproduct(Pm_i,Pn_i)
                    D_p = packNEW(D_ii)

                    # add compensation charges to exchange density
                    Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    gt_aL[a].add(n_G, Q_L)

                    # add atomic contribution to exchange energy
                    C_pp = nucleus.setup.M_pp
                    exxa-= fnm*num.dot(D_p, num.dot(C_pp, D_p))
                # determine total charge of exchange density
                if (n == m): Z = 1
                else: Z = 0

                # add the nm contribution to exchange energy
                exx += fnm*exx_single.getExchangeEnergy(n_G, Z=Z)

##         ExxValCore = ExxCore = 0.0
##         from gridpaw.atom.data import values
##         for nucleus in nuclei:
##             symbol = nucleus.setup.symbol
##             Exxc, X_p = values[symbol]
##             ExxCore += Exxc
##             print len(nucleus.D_sp[0])
##             ExxValCore += num.dot(nucleus.D_sp[0],X_p)

    return num.array([exx+exxa, exxa])#, ExxValCore, ExxCore])

def atomicExactExchange(atom, type = 'all'):
    '''Returns the exact exchange energy of the atom defined by the
    instantiated AllElectron obcejt "atom" '''

    # get Gaunt coefficients
    from gridpaw.gaunt import gaunt

    # get Hartree potential calculator
    from gridpaw.setup import Hartree

    # maximum angular momentum
    Lmax=(2*max(atom.l_j)+1)**2

    # number of orbitals
    Nj = len(atom.n_j)
    Njcore = coreStates(atom.symbol, atom.n_j, atom.l_j, atom.f_j)

    if type == 'all': nstates = mstates = range(Nj)
    elif type == 'val-val': nstates = mstates = range(Njcore,Nj)
    elif type == 'val-core':
        nstates = range(Njcore,Nj); mstates = range(Njcore)
    elif type == 'core-core': nstates = mstates = range(Njcore)
    else: 'ERROR unknown type: ', type

    # diagonal +-1 elements in Hartree matrix
    a1_g = 1.0 - 0.5 * (atom.d2gdr2 * atom.dr**2)[1:]
    a2_lg = -2.0 * num.ones((Lmax, atom.N - 1), num.Float)
    x_g = ((atom.dr / atom.r)**2)[1:]
    for l in range(1, Lmax):
        a2_lg[l] -= l * (l + 1) * x_g
    a3_g = 1.0 + 0.5 * (atom.d2gdr2 * atom.dr**2)[1:]

    # initialize potential calculator (returns v*r^2*dr/dg)
    H = Hartree(a1_g, a2_lg, a3_g, atom.r, atom.dr).solve

    # Hydrogen hack!
    # activate hack by uncommenting below 2 lines:
    #atom.u_j[0] = 2. * num.exp(-atom.r)*atom.r
    #print 'WARNING: Doing Hydrogen hack!!!'

    Exx = 0.0
    for j1 in nstates:
        l1 = atom.l_j[j1]
        for j2 in mstates:
            l2 = atom.l_j[j2]

            # joint occupation number
            f12 = .5*atom.f_j[j1]/(2.*l1+1)*\
                     atom.f_j[j2]/(2.*l2+1)

            # electron density
            n = atom.u_j[j1]*atom.u_j[j2]
            n[1:] /= atom.r[1:]**2

            # L summation
            vr2dr = num.zeros(atom.N, num.Float)
            for l in range(l1 + l2 + 1):
                vr2drl = H(n, l)
                G2 = gaunt[l1**2:(l1+1)**2, l2**2:(l2+1)**2,\
                           l**2:(l+1)**2]**2
                vr2dr += vr2drl * num.sum(G2.copy().flat)

            Exx +=-.5*f12*num.dot(n,vr2dr)

    # double energy if mixed contribution
    if type == 'val-core': Exx *= 2.
    return Exx

def constructX(atom):
    '''Construct the X_p^a matrix for the given atom'''

    # get Gaunt coefficients
    from gridpaw.gaunt import gaunt

    # get Hartree potential calculator
    from gridpaw.setup import Hartree

    # maximum angular momentum
    Lmax=(2*max(atom.l_j)+1)**2

    # number of orbitals
    Nj = len(atom.n_j)
    Njcore = atom.Njcore

    j_states = range(Njcore,Nj)
    alpha_states = range(Njcore)

    # diagonal +-1 elements in Hartree matrix
    a1_g = 1.0 - 0.5 * (atom.d2gdr2 * atom.dr**2)[1:]
    a2_lg = -2.0 * num.ones((Lmax, atom.N - 1), num.Float)
    x_g = ((atom.dr / atom.r)**2)[1:]
    for l in range(1, Lmax):
        a2_lg[l] -= l * (l + 1) * x_g
    a3_g = 1.0 + 0.5 * (atom.d2gdr2 * atom.dr**2)[1:]

    # initialize potential calculator (returns v*r^2*dr/dg)
    H = Hartree(a1_g, a2_lg, a3_g, atom.r, atom.dr).solve

    Np = len(j_states)
    X_p = num.zeros(Np * (Np + 1) / 2, num.Float)
    for alpha in alpha_states:
        p = 0
        for j1 in j_states:
            l1 = atom.l_j[j1] 

            # electron density
            nj1 = atom.u_j[j1]*atom.u_j[alpha]
            nj1[1:] /= atom.r[1:]**2  

            vr2dr = num.zeros(atom.N, num.Float)
            for l in range(2*l1 + 1):
                vr2drl = H(nj1, l)
                G2 = gaunt[l1**2:(l1+1)**2, l1**2:(l1+1)**2,\
                           l**2:(l+1)**2]**2
                vr2dr += vr2drl * num.sum(G2.copy().flat)

            X_p[p] += -num.dot(nj1,vr2dr)

            p += 1
            for j2 in range(j1+1, Nj):
                l2 = atom.l_j[j2] 

                # electron densities
                nj2 = atom.u_j[j2]*atom.u_j[alpha]
                nj2[1:] /= atom.r[1:]**2  

                X_p[p] += -2*num.dot(nj2,vr2dr)

                p += 1
    return X_p

''' AUXHILLIARY FUNCTIONS... should be moved to Utillities module'''

def rSquared(gd):
    I=num.indices(gd.N_c)
    dr = num.reshape(gd.h_c,(3,1,1,1))
    r0 = -0.5*num.reshape(gd.domain.cell_c,(3,1,1,1))
    r0 = num.ones(I.shape)*r0
    r2 = num.sum((r0+I*dr)**2)

    # remove zero at origin
    middle = gd.N_c/2.
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

def coreStates(symbol, n,l,f):
    from gridpaw.atom.configurations import configurations

    parameters = {
        #     (rcut, core)  
        'H' : (0.9, ''),
        'He': (1.5, ''),
        'Li': (1.5, '[He]'),
        'Be': (1.5, '[He]'),
        'C' : (1.0, '[He]'),
        'N' : (1.1, '[He]'),
        'O' : (1.2, '[He]'),
        'F' : (1.2, '[He]'),
        'Na': (2.3, '[Ne]'),
        'Mg': (2.2, '[Ne]'),
        'Al': (2.0, '[Ne]'),
        'Si': (2.0, '[Ne]'),
        'P' : (2.0, '[Ne]'),
        'S' : (1.87, '[Ne]'),
        'Cl': (1.5, '[Ne]'),
        'V' : (2.2, '[Ar]'),
        'Fe': (2.2, '[Ar]'),
        'Cu': (2.0, '[Ar]'),
        'Ga': (2.0, '[Ar]3d'),
        'As': (2.0, '[Ar]'),
        'Zr': (2.0, '[Ar]3d'),
        'Mo': (2.3, '[Kr]'),
        'Ru': (2.4, '[Kr]'),
        'Pt': (2.5, '[Xe]4f'),
        'Au': (2.5, '[Xe]4f')
        }
    
    rcut, core = parameters[symbol]
    
    # Parse core string:
    j = 0
    if core.startswith('['):
        a, core = core.split(']')
        core_symbol = a[1:]
        j = len(configurations[core_symbol][1])
        
    while core != '':
        assert n[j] == int(core[0])
        assert l[j] == 'spdf'.find(core[1])
        assert f[j] == 2 * (2 * l_j[j] + 1)
        j += 1
        core = core[2:]
    Njcore = j

    return Njcore
