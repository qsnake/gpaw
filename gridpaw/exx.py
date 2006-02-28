import Numeric as num
from Numeric import pi
from gridpaw.utilities.complex import real
from FFT import fftnd, inverse_fftnd
from gridpaw.poisson_solver import PoissonSolver
from gridpaw.utilities import DownTheDrain

class Translate:
    """Class used to translate wave functions / densities."""
    def __init__(self, sgd, lgd, type = num.Complex):
        self.Ns = sgd.N_c
        self.Nl = lgd.N_c
        self.Nr = 1. * self.Nl / self.Ns

        # ensure that the large grid-descriptor is an integer number of times
        # bigger than the small grid-descriptor
        assert num.alltrue(self.Nr == num.around(self.Nr))
        self.tmp = num.zeros(self.Nl, type)

    def translate(self, w, R):
        """Translate input array 'w' defined in the large grid-descriptor 'lgd'
           distance 'R' measured in units of the small grid-descriptor 'sgd'.
        """
        R = num.array(R)
        tmp = self.tmp

        # do nothing, if array is not moved
        if num.alltrue(R == 0): return
        
        # ensure that R is within allowed range and of correct type
        assert num.alltrue(R > 0 and R < self.Nr)

        # determine the size of the blocks to be moved
        B = R * self.Ns
        A = self.Nl - B

        # translate 1. axis
        tmp[:] = w
        w[:A[0]] = tmp[B[0]:]
        w[A[0]:] = tmp[:B[0]]
        
        # translate 2. axis
        tmp[:] = w
        w[:, :A[1]] = tmp[:, B[1]:]
        w[:, A[1]:] = tmp[:, :B[1]]
        
        # translate 3. axis
        tmp[:] = w
        w[:, :, :A[2]] = tmp[:, :, B[2]:]
        w[:, :, A[2]:] = tmp[:, :, :B[2]]

class Coulomb:
    """Class used to evaluate coulomb integrals, and exchange energies of given
       single orbital electron densities
    """
    def __init__(self, gd):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gridpaw module
        """        
        self.gd = gd
        self.ng1, self.vg1, self.Eg1 = self.construct_gauss(gd)

        # calculate reciprocal lattice vectors
        dim = num.reshape(gd.N_c, (3,1,1,1))
        dk = 2*pi / gd.domain.cell_c
        dk.shape = (3, 1, 1, 1)
        k = ((num.indices(self.gd.N_c)+dim/2)%dim - dim/2)*dk
        self.k2 = sum(k**2)
        self.k2[0,0,0] = 1.0

        # Ewald corection
        rc = .5 * num.average(gd.domain.cell_c)
        self.ewald = num.ones(gd.N_c) - num.cos(num.sqrt(self.k2) * rc)
        # lim k ->0 ewald / k2 
        self.ewald[0,0,0] = .5 * rc**2

        # determine N^3
        self.N3 = self.gd.N_c[0]*self.gd.N_c[1]*self.gd.N_c[2]

    def rSquared(gd):
        """constructs and returns a matrix containing the square of the
           distance from the origin which is placed in the center of the box
           described by the given grid-descriptor 'gd'.
        """

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
    rSquared = staticmethod(rSquared)

    def erf3D(M):
        """return matrix with the value of the error function evaluated for
           each element in input matrix 'M'.
        """
        from gridpaw.utilities import erf

        dim = M.shape
        res = num.zeros(dim,num.Float)
        for k in range(dim[0]):
            for l in range(dim[1]):
                for m in range(dim[2]):
                    res[k,l,m] = erf(M[k,l,m])
        return res
    erf3D = staticmethod(erf3D)
        
    def construct_gauss(gd):
        """Construct gaussian density, potential and self-energy"""
        # determine r^2 and r matrices
        r2 = Coulomb.rSquared(gd)
        r  = num.sqrt(r2)

        # 'width' of gaussian distribution
        # a=a0/... => ng~exp(-a0/4) on the boundary of the domain
        a = 25./min(gd.domain.cell_c)**2

        # gaussian density for Z=1
        ng1 = num.exp(-a*r2)*(a/pi)**(1.5)

        # gaussian potential for Z=1
        vg1 = Coulomb.erf3D(num.sqrt(a)*r)/r

        # gaussian self energy for Z=1
        Eg1 = -num.sqrt(a/2/pi)

        return ng1, vg1, Eg1
    construct_gauss = staticmethod(construct_gauss)

    def neutralize(self, n, Z=None):
        """Method for neutralizing input density 'n' with nonzero total
           charge 'Z'. Returns energy correction caused by making 'n' neutral
        """
        if Z == None: Z = self.gd.integrate(n)
        if type(Z) == complex: print Z; assert abs(Z.imag) < 1e-6
        if abs(Z) < 1e-8: return 0.0
        else:
            # construct gauss density array
            ng = Z*self.ng1 # gaussian density
            
            # calculate energy corrections
            EGaussN    = -0.5 * num.conjugate(Z)\
                         * self.gd.integrate(n * self.vg1)
            EGaussSelf = num.absolute(Z)**2 * self.Eg1
            
            # neutralize density
            n -= ng
            
            # return correctional energy contribution due to neutralization
            return - EGaussSelf + 2 * real(EGaussN)

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
            # make density charge neutral, and get energy correction
            Ecorr = self.neutralize(n, Z)

            # determine potential
            solve = PoissonSolver(self.gd, out=DownTheDrain()).solve
            v = self.gd.new_array()
            solve(v,n)

            # determine exchange energy
            exx = -0.5*self.gd.integrate(v * n)

            # return resulting exchange energy
            return exx + Ecorr
        elif method=='recip':
            if ewald: return -.5 * self.single_coulomb(n)
            else:
                Ecorr = self.neutralize(n, Z)
                I = num.absolute(fftnd(n))**2 * 4 * pi / self.k2
                exx = -0.5*self.gd.integrate(I) / self.N3
                
                # return resulting exchange energy
                return exx + Ecorr
        else:
            raise RunTimeError('method name ', method, 'not recognized')
    
    def single_coulomb(self, n):
        """Evaluates the coulomb integral:
                                           *
                            /    /      n(r)  n(r')
          ((n)) = (n | n) = | dr | dr'  -----------
	                    /    /        |r - r'|
	   where n could be complex.
	"""
        nk = fftnd(n)
        I = num.absolute(nk)**2 * self.ewald / self.k2
        return self.gd.integrate(I) * 4 * pi  / self.N3

    def dual_coulomb(self, n1, n2):
        """Evaluates the coulomb integral:
                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------
	              /    /         |r - r'|
	   where n1 and n2 could be complex.
	"""
        n1k = fftnd(n1)
        n2k = fftnd(n2)

        I = num.conjugate(n1k) * n2k * self.ewald / self.k2
        return self.gd.integrate(I) * 4 * pi  / self.N3
   
    def dual_coulomb_old(self, n1, n2):
        """Evaluates the coulomb integral:
                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------
	              /    /         |r - r'|
	   where n1 and n2 could be complex.
	"""
        from FFT import inverse_fftnd
        
        # Construct gauss density and potential for density 2
	Z1  = self.gd.integrate(n1)
        if type(Z1) == complex: assert abs(Z1.imag) < 1e-6
	ng1 = Z1*self.ng1

	# Neutralize density n1
	n1_neutral = n1 - ng1

        # Construct gauss density and potential for density 2
	Z2  = self.gd.integrate(n2)
        if type(Z2) == complex: assert abs(Z2.imag) < 1e-6
	ng2 = Z2*self.ng1

	# Neutralize density n2
	n2_neutral = n2 - ng2

        n2_neutral_k = fftnd(n2_neutral)
	v2_neutral_k = (4*pi*n2_neutral_k)/self.k2
	v2_neutral   = inverse_fftnd(v2_neutral_k)

	exx   = num.conjugate(n1_neutral) * v2_neutral
	corr1 = (Z1 * n2 + Z2 * n2) * self.vg1
	corr2 = -2 * Z1 * Z2 * self.Eg1
	return self.gd.integrate(exx + corr1) - corr2

class PawExx:
    """Class offering methods for non-selfconsistent evaluation of the
       exchange energy of a gridPAW calculation.
    """
    def __init__(self, calc):
        # store options in local varibles
        self.calc = calc
        self.wannier = None
        self.ewald = None
        self.method = None

        # allocate space for fine grid density
        self.n_g = calc.paw.finegd.new_array()

        # load single exchange calculator
        self.exx_single = Coulomb(calc.paw.finegd).get_single_exchange
        
    def get_exact_exchange(self,
                           decompose = False,
                           wannier   = False,
                           ewald     = True,
                           method    = 'recip'):
        """Control method for the calculation of exact exchange energy"""

        # check if desired calculation type is possible and setup interpolator
        paw = self.calc.paw
        if wannier != self.wannier:
            if paw.wf.typecode == num.Float:
                self.interpolate = paw.interpolate
                #Interpolator(paw.gd, 5, num.Float).apply
            elif wannier:
                from gridpaw.transformers import Interpolator
                self.interpolate = Interpolator(paw.gd, 5, num.Complex).apply
            else:
                raise RuntimeError('Cannot do k-point calculation without' + \
                                   'wannier keyword')

        # only do calculation if not previously done
        if wannier != self.wannier or method != self.method or \
               (method == 'recip' and ewald != self.ewald):
            
            # update calculation parameters
            self.wannier = wannier
            self.ewald   = ewald
            self.method  = method

            # Get valence-valence contribution using specified method
            if not self.wannier:
                self.ExxVal = self.valence_kohn_sham()
            elif paw.wf.typecode == num.Float:
                self.ExxVal = self.valence_wannier_gamma()
            else:
                self.ExxVal = self.valence_wannier_kpoints()

        if not hasattr(self, 'ExxCore'):
            # Get valence-core and core-core exact exchange contributions
            self.ExxValCore, self.ExxCore = self.valence_core_core(paw.nuclei,
                                                                 paw.wf.nspins)

        # add all contributions, to get total exchange energy
        self.Exx = self.ExxVal + self.ExxValCore + self.ExxCore    

        # return result
        if decompose:
            return num.array([self.Exx, self.ExxVal,
                              self.ExxValCore, self.ExxCore])
        else: return self.Exx

    def gauss_functions(gd, nuclei, R_Rc=num.zeros((1,3)),
                        Nk_c=num.ones(3), type=num.Float):
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
            for R in R_Rc:
                spos_c = (nucleus.spos_c + R) / Nk_c
                gt_L = create_localized_functions(gSpline, gd,
                                                  spos_c, typecode=type)
#                gt_L.set_phase_factors([[0, 0, 0]])
                gt_aL.append(gt_L)
        return gt_aL
    gauss_functions = staticmethod(gauss_functions)

    def get_kpoint_dimensions(kpts):
        """returns number of kpoints along each axis of input Monkhorst pack"""
        nkpts = len(kpts)
        print nkpts, kpts.shape
        if nkpts == 1: return num.ones(3)
        tol = 1e-5
        Nk_c = num.zeros(3)
        for c in range(3):
            # sort kpoints in ascending order along current axis
            slist = num.argsort(kpts[:,c])
            skpts = num.take(kpts, slist)

            # determine increment between kpoints along current axis
            DeltaK = max([skpts[n+1,c] - skpts[n,c] for n in range(nkpts-1)])

            # determine number of kpoints as inverse of distance between kpoints
            if DeltaK > tol: Nk_c[c] = int(round(1/DeltaK)) 
            else: Nk_c[c] = 1
        return Nk_c
    get_kpoint_dimensions= staticmethod(get_kpoint_dimensions)

    def valence_kohn_sham(self):
        """Calculate valence-valence contribution to exact exchange
           energy using Kohn-Sham orbitals
        """
        wf = self.calc.paw.wf
        nuclei = self.calc.paw.nuclei
        finegd = self.calc.paw.finegd

        # get gauss functions
        gt_aL = self.gauss_functions(finegd, nuclei)

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
                    self.interpolate(n_G, self.n_g)

                    # determine for each nucleus, the atomic correction
                    for a, nucleus in enumerate(nuclei):
                        # generate density matrix
                        Pm_i = nucleus.P_uni[spin,m]
                        Pn_i = nucleus.P_uni[spin,n]
                        D_ii = num.outerproduct(Pm_i,Pn_i)
                        D_p  = packNEW(D_ii)

                        # add compensation charges to exchange density
                        Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                        gt_aL[a].add(self.n_g, Q_L)

                        # add atomic contribution to exchange energy
                        C_pp  = nucleus.setup.M_pp
                        Exxa = -fnm*num.dot(D_p, num.dot(C_pp, D_p)) * DC
                        ExxVal += Exxa

                    # determine total charge of exchange density
                    if n == m: Z = 1
                    else: Z = 0

                    # add the nm contribution to exchange energy
                    Exxs = fnm * self.exx_single(self.n_g, Z=Z,
                                                 ewald=self.ewald,
                                                 method=self.method) * DC
                    ExxVal += Exxs
        return ExxVal
    
    def valence_wannier_gamma(self):
        """Calculate valence-valence contribution to exact exchange
           energy using Wannier function
        """
        # load additional packages for wannier calculation
        from ASE.Utilities.Wannier import Wannier
        from gridpaw.utilities.blas import gemm
        from gridpaw.transformers import Interpolator

        wf = self.calc.paw.wf
        nuclei = self.calc.paw.nuclei
        finegd = self.calc.paw.finegd

        # get gauss functions
        gt_aL = self.gauss_functions(finegd, nuclei)

        # initialize variable for the list of wannier wave functions
        wannierwave_nG = None

        # calculate exact exchange
        ExxVal = 0.0
        for spin in range(wf.nspins):
            # determine number of occupied orbitals for current spin
            # Note! Cannot handle spin compensated with odd numbered electrons
            # e.g spin compensated H with 1/2 up electron and 1/2 down electron
            states = int(round(num.sum(wf.kpt_u[spin].f_n))) * wf.nspins / 2

            if states < 1: break # do not proceed if no orbitals are occupied

            # allocate space for wannier wave function if necessary
            if wannierwave_nG == None:
                if wf.kpt_u[spin].Htpsit_nG == None:
                    wannierwave_nG = num.zeros((states,) + 
                                               tuple(self.calc.paw.gd.N_c),
                                               num.Float)
                else: wannierwave_nG = wf.kpt_u[spin].Htpsit_nG[:states]

            # determine the wannier rotation matrix
            wannier = Wannier(numberofwannier=states,
                              calculator=self.calc,
                              spin=spin)
            wannier.Localize()
            U_knn = wannier.GetListOfRotationMatrices()
            rotation = U_knn[0].real.copy()

            # apply rotation to old wavefunctions and get wannier wavefunctions
            psit_nG = wf.kpt_u[spin].psit_nG[:states]
            gemm(1.0, psit_nG, rotation, 0.0, wannierwave_nG)

            # apply rotation to expansion coeff. P = <ptilde|psitilde>
            P_ani = [num.matrixmultiply(rotation, nucleus.P_uni[spin,:states])
                     for nucleus in nuclei]

            # determine Exx contribution from each valence-valence state-pair
            for n in range(states):
                for m in range(n, states):
                    # determine double count factor:
                    DC = 2 - (n == m)

                    # determine current exchange density
                    n_G = num.conjugate(wannierwave_nG[m]) * \
                          wannierwave_nG[n]

                    # and interpolate to the fine grid
                    self.interpolate(n_G, self.n_g)

                    # determine for each nucleus, the atomic correction
                    for a, nucleus in enumerate(nuclei):
                        # generate density matrix
                        D_ii = num.outerproduct(num.conjugate(P_ani[a][m]),
                                                P_ani[a][n])
                        D_p = packNEW(D_ii)

                        # add compensation charges to exchange density
                        Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                        gt_aL[a].add(self.n_g, Q_L)

                        # add atomic contribution to exchange energy
                        C_pp  = nucleus.setup.M_pp
                        Exxa = - num.dot(D_p, num.dot(C_pp, D_p)) * DC
                        ExxVal += Exxa

                    # add the nm contribution to exchange energy
                    Exxs = self.exx_single(self.n_g, ewald=self.ewald,
                                           method=self.method) * DC
                    ExxVal += Exxs
        # double up if spin compensated
        ExxVal *= wf.nspins % 2 + 1
        return ExxVal
    
    def valence_wannier_kpoints(self):
        """Calculate valence-valence contribution to exact exchange
           energy using Wannier function
        """
        raise NotImplementedError
    
        # load additional packages for wannier calculation
        from ASE.Utilities.Wannier import Wannier
        from gridpaw.utilities.blas import gemm
        from gridpaw.transformers import Interpolator
        from gridpaw.domain import Domain
        from gridpaw.grid_descriptor import GridDescriptor

        wf = self.calc.paw.wf
        nuclei = self.calc.paw.nuclei
        gd = self.calc.paw.gd
        finegd = self.calc.paw.finegd

        assert len(wf.bzk_kc) == len(wf.ibzk_kc)
        
        # get information on the number of kpoints
        Nk = wf.nkpts
        Nk_c = self.get_kpoint_dimensions(num.array(wf.ibzk_kc))

        # construct large grid-descriptor of repeated unitcell
        ldomain = Domain(gd.domain.cell_c * Nk_c)
        lgd = GridDescriptor(ldomain, gd.N_c * Nk_c)

        # load single exchange calculator and translator
        exx_single = Coulomb(lgd).get_single_exchange
        translate = Translate(gd, lgd).translate

        # construct translation vectors
        R_Rc = num.zeros((Nk, 3))
        for i in range(Nk_c[0]):
            for j in range(Nk_c[1]):
                for k in range(Nk_c[2]):
                    tmp = [i, j, k]
                    R_Rc[num.dot(tmp, Nk_c - 1)] = tmp

        # get gauss functions
        gt_AL = self.gauss_functions(lgd, nuclei, R_Rc, Nk_c, type=num.Complex)

        # initialize variable for the list of wannier wave functions
        wannierwave_nG = None

        # calculate exact exchange
        ExxVal = 0.0
        for spin in range(wf.nspins):
            # determine number of occupied orbitals for current spin
            # Note! Cannot handle spin compensated with odd numbered electrons
            # e.g spin compensated H with 1/2 up electron and 1/2 down electron
            # Note! This only works if there is a band gap, i.e. sum(f_n) must
            # be identical for each kpoint
            states = int(round(num.sum(wf.kpt_u[spin].f_n))) * wf.nspins / 2

            if states < 1: break # do not proceed if no orbitals are occupied

            # allocate space for wannier wave function if necessary
            if wannierwave_nG == None:
                wannierwave_nG = num.zeros((states,) + tuple(lgd.N_c),
                                           num.Complex)

            # determine the wannier rotation matrix
            wannier = Wannier(numberofwannier=states,
                              calculator=self.calc,
                              spin=spin)
            wannier.Localize()
            U_knn = wannier.GetListOfRotationMatrices()

            # apply rotation to old wavefunctions and get wannier wavefunctions
            wannierwave_nG = [wannier.GetGrid(n).GetArray()
                              for n in range(states)]

            # apply rotation to expansion coeff. P = <ptilde|psitilde>
            P_Ani = []
            for nucleus in nuclei:
                rot_k = [num.matrixmultiply(U_knn[k],
                                   nucleus.P_uni[spin * Nk + k, :states])
                         for k in range(Nk)]
                for R in R_Rc:
                    P_Ani.append(num.zeros(rot_k[0].shape, num.Complex))
                    for k in range(Nk):
                        phase = num.dot( -2*pi*wf.ibzk_kc[k], R)
                        # perhaps R should be multiplied by domain size
                        P_Ani[-1] += num.exp(1.j * phase) * rot_k[k]
                    P_Ani[-1] /= Nk**.5

            # determine Exx contribution from each valence-valence state-pair
            for n in range(states):
                for m in range(n, states):
                    for Rn in R_Rc:
                        # determine double count factor:
                        DC = 2 - (n == m)

                        translate(wannierwave_nG[n], Rn)
                        # determine current exchange density
                        n_G = num.conjugate(wannierwave_nG[m]) * \
                              wannierwave_nG[n]

                        # determine for each nucleus, the atomic correction
                        for a, nucleus in enumerate(nuclei):
                            for R in range(Nk):
                                # generate density matrix
                                D_ii = num.outerproduct(num.conjugate( \
                                    P_Ani[a*Nk+R][m]), P_Ani[a*Nk+R][n])
                                D_p = packNEW(D_ii)

                                # add compensation charges to exchange density
                                Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                                gt_AL[a*Nk+R].add(n_G, Q_L)

                                # add atomic contribution to exchange energy
                                C_pp  = nucleus.setup.M_pp
                                Exxa  = - num.dot(D_p, num.dot(C_pp, D_p)) * DC
                                ExxVal += Exxa

                        # add the nm contribution to exchange energy
                        Exxs = exx_single(n_G, ewald=self.ewald,
                                          method=self.method) * DC
                        ExxVal += Exxs
        # double up if spin compensated
        ExxVal *= wf.nspins % 2 + 1
        return Nk * ExxVal

    def valence_core_core(nuclei, nspins):
        """Determine the valence-core and core-core contributions for each
           spin and nucleus
        """

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
    valence_core_core = staticmethod(valence_core_core)

def atomic_exact_exchange(atom, type = 'all'):
    """Returns the exact exchange energy of the atom defined by the
       instantiated AllElectron object 'atom'
    """

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

def translate_test():
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor
    from ASE.Visualization.VTK import VTKPlotArray
    
    d  = Domain((4,4,4))   # domain object
    N  = 2**4                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    gd = GridDescriptor(d,Nc) # grid-descriptor object

    N  *= 4                   # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    lgd = GridDescriptor(d,Nc)# grid-descriptor object

    r2 = Coulomb.rSquared(lgd)# matrix with the square of the radial coordinate
    g  = num.exp(-r2)/ pi     # gaussian density 

    trans = Translate(gd, lgd, num.Float).translate

    g2 = g.copy()
    trans(g2,(2,2,3))
    a = 8 * num.identity(3, num.Float)
    VTKPlotArray(g2, a)

def single_exchange_test():
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor

    d  = Domain((20,20,20))   # domain object
    N  = 2**6                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    gd = GridDescriptor(d,Nc) # grid-descriptor object
    r2 = Coulomb.rSquared(gd) # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2*r)/pi     # density of the hydrogen atom

    exx = Coulomb(gd).get_single_exchange(nH, method = 'recip')
    print 'Numerical result: ', exx
    print 'Analytic result:  ', -5/16.
    print -.5 * Coulomb(gd).dual_coulomb(nH, nH.copy())
    print -.5 * Coulomb(gd).dual_coulomb_old(nH, nH.copy())
    print -.5 * Coulomb(gd).single_coulomb(nH)

if __name__ == '__main__':
    single_exchange_test()
    #translate_test()    
