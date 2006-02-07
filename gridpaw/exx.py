import Numeric as num
from math import pi
from gridpaw.utilities.complex import real

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
        dim = num.reshape(gd.N_c, (3,1,1,1))
        dk = 2*pi / gd.domain.cell_c
        dk.shape = (3, 1, 1, 1)
        k = ((num.indices(self.gd.N_c)+dim/2)%dim - dim/2)*dk
        self.k2 = sum(k**2)
        self.k2[0,0,0] = 1.0

        # Ewald corection
        rc = num.reshape(gd.domain.cell_c, (3,1,1,1)) / 2. 
        self.ewald = num.ones(gd.N_c) - num.cos(sum(k * rc))

        # determine N^3
        self.N3 = self.gd.N_c[0]*self.gd.N_c[1]*self.gd.N_c[2]

    def get_single_exchange(self, n, Z=None, ewald=True, method='recip'):
        """Returns exchange energy of input density 'n' """

        # make density charge neutral, and get energy correction
        Ecorr = self.neutralize(n, Z)

        # determine exchange energy of neutral density using specified method
        if method=='real':
            from gridpaw.poisson_solver import PoissonSolver
            solve = PoissonSolver(self.gd).solve
            v = self.gd.new_array()
            solve(v,n)
            exx = -0.5*self.gd.integrate(v * n)
        elif method=='recip':
            from FFT import fftnd
            I = num.absolute(fftnd(n))**2 * 4 * pi / self.k2
            if ewald: I *= self.ewald
            exx = -0.5*self.gd.integrate(I) / self.N3
        else: raise RunTimeError('method name ', method, 'not recognized')

        # return resulting exchange energy
        return exx + Ecorr
    
    def neutralize(self, n, Z):
        """Method for neutralizing input density 'n' with nonzero total
        charge. Returns energy correction caused by making 'n' neutral"""

        if Z == None: Z = self.gd.integrate(n)
        if type(Z) == complex: print '!!!!!!COMPLEX CHARGE!!!!!!' # XXX
        
        if Z < 1e-8: return 0.0
        else:
            # construct gauss density array
            ng = Z*self.ng1 # gaussian density
            
            # calculate energy corrections
            EGaussN    = -0.5 * num.conjugate(Z) * \
                                              self.gd.integrate(n*self.vgauss1)
            EGaussSelf = num.absolute(Z)**2 * self.EGaussSelf1
            
            # neutralize density
            n -= ng
            
            # determine correctional energy contribution due to neutralization
            Ecorr = - EGaussSelf + 2 * real(EGaussN)
            return Ecorr

def get_exact_exchange(calc, decompose = False, wannier = False,
                       ewald = True, method = 'recip'):
    """Calculate exact exchange energy using Kohn-Sham orbitals"""

    # Get valence-valence contribution using specified method
    if wannier:
        ExxVal = __valence_wannier__(calc, ewald, method)
    else:
        ExxVal = __valence_kohn_sham__(calc.paw, ewald, method)

    # Get valence-core and core-core exact exchange contributions
    ExxValCore, ExxCore = __valence_core_core__(calc.paw.nuclei,
                                                calc.paw.wf.nspins)
    
    # add all contributions, to get total exchange energy
    Exx = ExxVal + ExxValCore + ExxCore    

    # return result
    if decompose: return num.array([Exx, ExxVal, ExxValCore, ExxCore])
    else: return Exx

def __gauss_functions__(nuclei, gd):
    """construct gauss functions"""
    from gridpaw.localized_functions import create_localized_functions
    from gridpaw.polynomium import a_i, c_l
    from gridpaw.spline import Spline
    
    gt_aL = []
    for nucleus in nuclei:
        rcut = nucleus.setup.rcut
        lmax = nucleus.setup.lmax
        x = num.arange(101) / 100.0
        s = num.zeros(101, num.Float)
        for i in range(4):
            s += a_i[i] * x**i
        gSpline = [Spline(l, rcut, c_l[l] / rcut**(3 + 2 * l) * s)
                                                      for l in range(lmax + 1)]
        gt_aL.append(create_localized_functions(gSpline, gd, nucleus.spos_c))

    return gt_aL

def __valence_kohn_sham__(paw, ewald, method):
    """Calculate valence-valence contribution to exact exchange
    energy using Kohn-Sham orbitals"""
    wf = paw.wf
    nuclei = paw.nuclei
    gd = paw.finegd

    # allocate space for fine grid density
    n_g = gd.new_array()

    # ensure gamma point calculation
    assert wf.typecode == num.Float

    # get gauss functions
    gt_aL = __gauss_functions__(nuclei, gd)

    # load single exchange calculator
    exx_single = ExxSingle(gd).get_single_exchange

    # calculate exact exchange
    ExxVal = 0.0
    for spin in range(wf.nspins):
        for n in range(wf.nbands):
            for m in range(n, wf.nbands):
                # determine double count factor:
                DC = 2 - (n == m)
                
                # calculate joint occupation number
                fnm = (wf.kpt_u[spin].f_n[n] *
                       wf.kpt_u[spin].f_n[m]) * wf.nspins / 2.
                
                # determine current exchange density
                n_G = wf.kpt_u[spin].psit_nG[m]*\
                      wf.kpt_u[spin].psit_nG[n]

                # and interpolate to the fine grid
                paw.interpolate(n_G, n_g)
                
                for a, nucleus in enumerate(nuclei):
                    # generate density matrix
                    Pm_i = nucleus.P_uni[spin,m]
                    Pn_i = nucleus.P_uni[spin,n]
                    D_ii = num.outerproduct(Pm_i,Pn_i)
                    D_p  = packNEW(D_ii)

                    # add compensation charges to exchange density
                    Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    gt_aL[a].add(n_g, Q_L)

                    # add atomic contribution to exchange energy
                    C_pp  = nucleus.setup.M_pp
                    Exxa = -fnm*num.dot(D_p, num.dot(C_pp, D_p)) * DC
                    ExxVal += Exxa
                    
                # determine total charge of exchange density
                if n == m: Z = 1
                else: Z = 0

                # add the nm contribution to exchange energy
                Exxs = fnm * exx_single(n_g, Z=Z, ewald=ewald,
                                        method=method) * DC
                ExxVal += Exxs
    return ExxVal
    
def __valence_wannier__(calculator, ewald, method):
    """Calculate valence-valence contribution to exact exchange
    energy using Wannier function"""
    from ASE.Utilities.Wannier import Wannier
    from gridpaw.utilities.blas import axpy, rk, r2k, gemm
    from gridpaw.transformers import Interpolator

    paw = calculator.paw
    wf = paw.wf
    nuclei = paw.nuclei
    gd = paw.finegd
    interpolate = Interpolator(paw.gd, 5, num.Complex).apply
    states = wf.nvalence * wf.nspins / 2
    print states

    # allocate space for fine grid density
    n_g = gd.new_array()

    # get gauss functions
    gt_aL = __gauss_functions__(nuclei, gd)

    # load single exchange calculator
    exx_single = ExxSingle(gd).get_single_exchange

    if wf.kpt_u[0].Htpsit_nG == None:
        wannierwave_nG = num.zeros((states,)+tuple(paw.gd.N_c),num.Float)
    else:
        wannierwave_nG = wf.kpt_u[0].Htpsit_nG[:states]
    
    # calculate exact exchange
    ExxVal = 0.0
    for spin in range(wf.nspins):
        # do wannier stuff
        wannier = Wannier(numberofwannier=states,
                          calculator=calculator,
                          spin=spin)
        wannier.Localize()
        rotation = wannier.rotationmatrix[0]
##         print 'BEFORE:', rotation
##         rotation = num.absolute(rotation)
##         print 'AFTER:', rotation
        
        psit_nG = wf.kpt_u[spin].psit_nG[:states]
        print wannierwave_nG.shape, wannierwave_nG.typecode()
        print psit_nG.shape, psit_nG.typecode()
        print rotation.shape, rotation.typecode()
        print rotation

        gemm(1.0, psit_nG, rotation, 0.0, wannierwave_nG)
        
        for n in range(states):
            for m in range(n, states):
                # determine double count factor:
                DC = 2 - (n == m)

                # determine current exchange density
                n_G = num.conjugate(wannierwave_nG[n]) * \
                      wannierwave_nG[n]

                # and interpolate to the fine grid
                interpolate(n_G, n_g)
                
                for a, nucleus in enumerate(nuclei):
                    # generate density matrix
                    Ni = len(nucleus.P_uni[0,0])
                    Pm_i = Pn_i = num.zeros(Ni, num.Float)
                    for state in range(states):
                        Pm_i += rotation[m,state] * nucleus.P_uni[spin,state]
                        Pn_i += rotation[n,state] * nucleus.P_uni[spin,state]
                    D_ii = num.outerproduct(num.conjugate(Pm_i),Pn_i)
                    D_p  = packNEW(D_ii)

                    # add compensation charges to exchange density
                    Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    gt_aL[a].add(n_g, Q_L)

                    # add atomic contribution to exchange energy
                    C_pp  = nucleus.setup.M_pp
                    Exxa = -num.dot(D_p, num.dot(C_pp, D_p)) * DC
                    ExxVal += Exxa

                # add the nm contribution to exchange energy
                Exxs = exx_single(n_g, ewald=ewald, method=method) * DC
                ExxVal += Exxs
    # double up if spin compensated
    ExxVal *= wf.nspins % 2 + 1
    return ExxVal
    
def __valence_core_core__(nuclei, nspins):
    """Determine the valence-core and core-core contributions for each
     spin and nucleus"""

    ExxCore = ExxValCore = 0.0

    for nucleus in nuclei:
        # error handling for old setup files
        if nucleus.setup.ExxC == None:
            print 'Warning no exact exchange information in setup file'
            print 'Value of exact exchange may be incorrect'
            print 'Please regenerate setup file to correct error'
            break
      
        # add core-core contribution from current nucleus
        ExxCore += nucleus.setup.ExxC

        # add val-core contribution from current nucleus
        for spin in range(nspins):
            D_p = nucleus.D_sp[spin]
            ExxValCore += - num.dot(D_p, nucleus.setup.X_p)

    return ExxValCore, ExxCore

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
    X_p = packNEW2(X_ii, symmetric = True)
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
                if error > 1e-6:
                    print 'Error not symmetric by:', error, '=',\
                          error/M2[r,c]*100, '%'
    assert p == len(M)
    return M

def packNEW2(M2, symmetric = False):
    """new pack method"""
    
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] =  (M2[r, c] + num.conjugate(M2[c,r])) / 2. # <- divide by 2!!
            p += 1
            if symmetric:
                error = abs(M2[r, c] - num.conjugate(M2[c, r]))
                if error > 1e-6:
                    print 'Error not symmetric by:', error, '=',\
                          error/M2[r,c]*100, '%'
    assert p == len(M)
    return M

if __name__ == '__main__':
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor

    d  = Domain((20,20,20))   # domain object
    N  = 2**6                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    gd = GridDescriptor(d,Nc) # grid-descriptor object
    r2 = rSquared(gd)         # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2*r)/pi     # density of the hydrogen atom

    exx = ExxSingle(gd).get_single_exchange(nH, method = 'recip')
    print 'Numerical result: ', exx
    print 'Analytic result:  ', -5/16.
