import Numeric as num
from Numeric import pi
from gridpaw.utilities.complex import real
from gridpaw.coulomb import Coulomb

class Translate:
    """Class used to translate wave functions / densities."""
    def __init__(self, sgd, lgd, type=num.Complex):
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

class PawExx:
    """Class offering methods for non-selfconsistent evaluation of the
       exchange energy of a gridPAW calculation.
    """
    def __init__(self, paw):
        # store options in local varibles
        self.paw = paw
        self.calc = None
        self.wannier = None
        self.ewald = None
        self.method = None
        #print 'Doing Exx on processor %s of %s' %(paw.domain.comm.rank,
        #                                          paw.domain.comm.size)

        # allocate space for fine grid density
        self.n_g = paw.finegd.new_array()

        # load single exchange calculator
        self.exx_single = Coulomb(paw.finegd).get_single_exchange

        # load interpolator
        if paw.wf.typecode == num.Float:
            self.interpolate = paw.interpolate
            #Interpolator(paw.gd, 5, num.Float).apply
        else:
            from gridpaw.transformers import Interpolator
            self.interpolate = Interpolator(paw.gd, 5, num.Complex).apply
        
    def get_exact_exchange(self,
                           decompose = False,
                           wannier   = False,
                           ewald     = True,
                           method    = 'recip',
                           calc      = None):
        """Control method for the calculation of exact exchange energy"""
        paw = self.paw

        # only do calculation if not previously done
        if wannier != self.wannier or method != self.method or \
               (method == 'recip' and ewald != self.ewald):
            
            # update calculation parameters
            self.calc    = calc
            self.wannier = wannier
            self.ewald   = ewald
            self.method  = method

            # Get valence-valence contribution using specified method
            if paw.wf.typecode == num.Float:
                if wannier and calc != None:
                    self.Exxt = self.valence_wannier_gamma()
                elif not wannier:
                    self.Exxt = self.valence_valence_gamma()
                else:
                    raise RuntimeError('Must give calculator object for ' + \
                                       'calculations using wannier functions')
            else:
                if wannier and calc != None:
                    self.Exxt = self.valence_wannier_kpoints()
                elif not wannier:
                    self.Exxt = self.valence_valence_kpoints()
                else:
                    raise RuntimeError('Must give calculator object for ' + \
                                       'calculations using wannier functions')

        if not hasattr(self, 'ExxVV'):
            self.ExxVV, self.ExxVC, self.ExxCC = self.atomic_corrections()
        
        # sum contributions from all processors
        ksum = paw.wf.kpt_comm.sum
        dsum = paw.domain.comm.sum
        self.Exxt  = ksum(self.Exxt)
        self.ExxVV = ksum(dsum(self.ExxVV))
        self.ExxVC = ksum(dsum(self.ExxVC))
        self.ExxCC = ksum(dsum(self.ExxCC))

        # add all contributions, to get total exchange energy
        Exx = num.array([self.Exxt, self.ExxVV, self.ExxVC, self.ExxCC])

        # return result
        if decompose:
            return Exx
        else: return sum(Exx)

    def gauss_functions(gd, nuclei, R_Rc=num.zeros((1, 3)),
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
                #gt_L.set_phase_factors([[0, 0, 0]])
                gt_aL.append(gt_L)
        return gt_aL
    gauss_functions = staticmethod(gauss_functions)

    def get_kpoint_dimensions(kpts):
        """Returns number of kpoints along each axis of input Monkhorst pack"""
        nkpts = len(kpts)
        if nkpts == 1: return num.ones(3)
        tol = 1e-5
        Nk_c = num.zeros(3)
        for c in range(3):
            # sort kpoints in ascending order along current axis
            slist = num.argsort(kpts[:, c])
            skpts = num.take(kpts, slist)

            # determine increment between kpoints along current axis
            DeltaK = max([skpts[n+1, c] - skpts[n, c] for n in range(nkpts-1)])

            #determine number of kpoints as inverse of distance between kpoints
            if DeltaK > tol: Nk_c[c] = int(round(1/DeltaK))
            else: Nk_c[c] = 1
        return Nk_c
    get_kpoint_dimensions = staticmethod(get_kpoint_dimensions)

    def valence_valence_gamma(self):
        """Calculate valence-valence contribution to exact exchange
           energy using Kohn-Sham orbitals
        """
        wf = self.paw.wf
        ghat_nuclei = self.paw.ghat_nuclei
        finegd = self.paw.finegd
        parallel = (self.paw.domain.comm.size != 1)

        # get gauss functions
        if parallel:
            assert not self.paw.nuclei[0].setup.softgauss
        else:
            gt_aL = self.gauss_functions(finegd, ghat_nuclei)

        # calculate exact exchange of smooth wavefunctions
        Exxt = 0.0
        for u in range(len(wf.myspins)): # local spin index
            #spin = wf.myspins[u] # global spin index XXX
            for n in range(wf.nbands):
                for m in range(n, wf.nbands):
                    # determine double count factor:
                    DC = 2 - (n == m)

                    # calculate joint occupation number
                    fnm = (wf.kpt_u[u].f_n[n] *
                           wf.kpt_u[u].f_n[m]) * wf.nspins / 2.

                    # determine current exchange density
                    n_G = wf.kpt_u[u].psit_nG[m] * \
                          wf.kpt_u[u].psit_nG[n] 

                    # and interpolate to the fine grid
                    self.interpolate(n_G, self.n_g)

                    # determine the compensation charges for each nucleus
                    for a, nucleus in enumerate(ghat_nuclei):
                        if nucleus.in_this_domain:
                            # generate density matrix
                            Pm_i = nucleus.P_uni[u, m] # spin ??
                            Pn_i = nucleus.P_uni[u, n] # spin ??
                            D_ii = num.outerproduct(Pm_i,Pn_i)
                            D_p  = packNEW(D_ii)
                            
                            # determine compensation charge coefficients
                            Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                        else:
                            Q_L = None

                        # add compensation charges to exchange density
                        if parallel:
                            nucleus.ghat_L.add(self.n_g, Q_L, communicate=True)
                        else:
                            gt_aL[a].add(self.n_g, Q_L)

                    # determine total charge of exchange density
                    if n == m: Z = 1.
                    else: Z = 0.
                    # Z = float(n == m)

                    # add the nm contribution to exchange energy
                    Exxt += fnm * DC * self.exx_single(self.n_g, Z=Z,
                                                       ewald=self.ewald,
                                                       method=self.method)
        return Exxt
    
    def valence_wannier_gamma(self):
        """Calculate valence-valence contribution to exact exchange
           energy using Wannier function
        """
        # load additional packages for wannier calculation
        from ASE.Utilities.Wannier import Wannier
        from gridpaw.utilities.blas import gemm
        from gridpaw.transformers import Interpolator

        wf = self.paw.wf
        nuclei = self.paw.nuclei
        finegd = self.paw.finegd

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
                                               tuple(self.paw.gd.N_c),
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
    
    def valence_valence_kpoints(self):
        """Calculate valence-valence contribution to exact exchange
           energy using kohn-sham orbitals...
        """
        raise NotImplementedError
    
##         # load additional packages
##         from gridpaw.domain import Domain
##         from gridpaw.grid_descriptor import GridDescriptor

##         wf = self.paw.wf
##         nuclei = self.paw.nuclei
##         gd = self.paw.gd
##         finegd = self.paw.finegd

##         assert len(wf.bzk_kc) == len(wf.ibzk_kc)
        
##         # get information on the number of kpoints
##         Nk = wf.nkpts
##         Nk_c = self.get_kpoint_dimensions(num.array(wf.ibzk_kc))

##         # construct large grid-descriptor of repeated unitcell
##         ldomain = Domain(gd.domain.cell_c * Nk_c)
##         lgd = GridDescriptor(ldomain, gd.N_c * Nk_c)

##         # load single exchange calculator and translator
##         exx_single = Coulomb(lgd).get_single_exchange
##         translate = Translate(gd, lgd).translate

##         # construct translation vectors
##         R_Rc = num.zeros((Nk, 3))
##         for i in range(Nk_c[0]):
##             for j in range(Nk_c[1]):
##                 for k in range(Nk_c[2]):
##                     tmp = [i, j, k]
##                     R_Rc[num.dot(tmp, Nk_c - 1)] = tmp

##         # get gauss functions
##         gt_AL = self.gauss_functions(lgd, nuclei, R_Rc, Nk_c, type=num.Complex)

    def valence_wannier_kpoints(self):
        """Calculate valence-valence contribution to exact exchange
           energy using Wannier function
        """
        raise NotImplementedError
    
##         # load additional packages for wannier calculation
##         from ASE.Utilities.Wannier import Wannier
##         from gridpaw.utilities.blas import gemm
##         from gridpaw.transformers import Interpolator
##         from gridpaw.domain import Domain
##         from gridpaw.grid_descriptor import GridDescriptor

##         wf = self.paw.wf
##         nuclei = self.paw.nuclei
##         gd = self.paw.gd
##         finegd = self.paw.finegd

##         assert len(wf.bzk_kc) == len(wf.ibzk_kc)
        
##         # get information on the number of kpoints
##         Nk = wf.nkpts
##         Nk_c = self.get_kpoint_dimensions(num.array(wf.ibzk_kc))

##         # construct large grid-descriptor of repeated unitcell
##         ldomain = Domain(gd.domain.cell_c * Nk_c)
##         lgd = GridDescriptor(ldomain, gd.N_c * Nk_c)

##         # load single exchange calculator and translator
##         exx_single = Coulomb(lgd).get_single_exchange
##         translate = Translate(gd, lgd).translate

##         # construct translation vectors
##         R_Rc = num.zeros((Nk, 3))
##         for i in range(Nk_c[0]):
##             for j in range(Nk_c[1]):
##                 for k in range(Nk_c[2]):
##                     tmp = [i, j, k]
##                     R_Rc[num.dot(tmp, Nk_c - 1)] = tmp

##         # get gauss functions
##         gt_AL = self.gauss_functions(lgd, nuclei, R_Rc, Nk_c, type=num.Complex)

##         # initialize variable for the list of wannier wave functions
##         wannierwave_nG = None

##         # calculate exact exchange
##         ExxVal = 0.0
##         for spin in range(wf.nspins):
##             # determine number of occupied orbitals for current spin
##             # Note! Cannot handle spin compensated with odd numbered electrons
##             # e.g spin compensated H with 1/2 up electron and 1/2 down electron
##             # Note! This only works if there is a band gap, i.e. sum(f_n) must
##             # be identical for each kpoint
##             states = int(round(num.sum(wf.kpt_u[spin].f_n))) * wf.nspins / 2

##             if states < 1: break # do not proceed if no orbitals are occupied

##             # allocate space for wannier wave function if necessary
##             if wannierwave_nG == None:
##                 wannierwave_nG = num.zeros((states,) + tuple(lgd.N_c),
##                                            num.Complex)

##             # determine the wannier rotation matrix
##             wannier = Wannier(numberofwannier=states,
##                               calculator=self.calc,
##                               spin=spin)
##             wannier.Localize()
##             U_knn = wannier.GetListOfRotationMatrices()

##             # apply rotation to old wavefunctions and get wannier wavefunctions
##             wannierwave_nG = [wannier.GetGrid(n).GetArray()
##                               for n in range(states)]

##             # apply rotation to expansion coeff. P = <ptilde|psitilde>
##             P_Ani = []
##             for nucleus in nuclei:
##                 rot_k = [num.matrixmultiply(U_knn[k],
##                                    nucleus.P_uni[spin * Nk + k, :states])
##                          for k in range(Nk)]
##                 for R in R_Rc:
##                     P_Ani.append(num.zeros(rot_k[0].shape, num.Complex))
##                     for k in range(Nk):
##                         phase = num.dot( -2*pi*wf.ibzk_kc[k], R)
##                         # perhaps R should be multiplied by domain size
##                         P_Ani[-1] += num.exp(1.j * phase) * rot_k[k]
##                     P_Ani[-1] /= Nk**.5

##             # determine Exx contribution from each valence-valence state-pair
##             for n in range(states):
##                 for m in range(n, states):
##                     for Rn in R_Rc:
##                         # determine double count factor:
##                         DC = 2 - (n == m)

##                         translate(wannierwave_nG[n], Rn)
##                         # determine current exchange density
##                         n_G = num.conjugate(wannierwave_nG[m]) * \
##                               wannierwave_nG[n]

##                         # determine for each nucleus, the atomic correction
##                         for a, nucleus in enumerate(nuclei):
##                             for R in range(Nk):
##                                 # generate density matrix
##                                 D_ii = num.outerproduct(num.conjugate( \
##                                     P_Ani[a*Nk+R][m]), P_Ani[a*Nk+R][n])
##                                 D_p = packNEW(D_ii)

##                                 # add compensation charges to exchange density
##                                 Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
##                                 gt_AL[a*Nk+R].add(n_G, Q_L)

##                                 # add atomic contribution to exchange energy
##                                 C_pp  = nucleus.setup.M_pp
##                                 Exxa  = - num.dot(D_p, num.dot(C_pp, D_p)) * DC
##                                 ExxVal += Exxa

##                         # add the nm contribution to exchange energy
##                         Exxs = exx_single(n_G, ewald=self.ewald,
##                                           method=self.method) * DC
##                         ExxVal += Exxs
##         # double up if spin compensated
##         ExxVal *= wf.nspins % 2 + 1
##         return Nk * ExxVal

    def atomic_corrections(self):
        """Determine the atomic corrections to the valence-valence exchange
           interaction, the valence-core contribution, and the core-core
           contributions.
        """
        wf = self.paw.wf
        ExxVV = ExxVC = ExxCC = 0.0
        for nucleus in self.paw.my_nuclei:
            # error handling for old setup files
            if nucleus.setup.ExxC == None:
                print 'Warning no exact exchange information in setup file'
                print 'Value of exact exchange may be incorrect'
                print 'Please regenerate setup file to correct error'
                break

            for spin in wf.myspins: # global spin index
                """Add core-core contribution:"""
                if spin == 0:
                    ExxCC += nucleus.setup.ExxC

                """Add val-core contribution:
                      vc,a     -    a    a
                     E     = - >   D  * X
                      xx       - p  p    p
                """
                D_p = nucleus.D_sp[spin]
                ExxVC += - num.dot(D_p, nucleus.setup.X_p)

            """Determine the atomic corrections to the val-val interaction:
                       -- 
                vv,a   \     a        a              a
               E     = /    D      * C            * D
                xx     --    i1,i3    i1,i2,i3,i4    i2,i4
                    i1,i2,i3 ,4
                 
                       -- 
                vv,a   \             a      a      a      a      a
               E     = /    f * f * P    * P    * P    * P    * C    
                xx     --    n   m   n,i1   m,i2   n,i3   m,i4   i1,i2,i3,i4
                    n,m,i1,i2,i3,i4
            """
            for u in range(len(wf.myspins)): # local spin index
                #spin = wf.myspins[u] # global spin index XXX
                for n in range(wf.nbands):
                    for m in range(n, wf.nbands):
                        # determine double count factor:
                        DC = 2 - (n == m)

                        # calculate joint occupation number
                        fnm = (wf.kpt_u[u].f_n[n] *
                               wf.kpt_u[u].f_n[m]) * wf.nspins / 2.

                        # generate density matrix
                        Pm_i = nucleus.P_uni[u, m] # spin ??
                        Pn_i = nucleus.P_uni[u, n] # spin ??
                        D_ii = num.outerproduct(Pm_i,Pn_i)
                        D_p  = packNEW(D_ii)

                        # C_iiii from setup file
                        C_pp  = nucleus.setup.M_pp

                        # add atomic contribution to val-val interaction
                        ExxVV += - fnm * num.dot(D_p, num.dot(C_pp, D_p)) * DC

        return ExxVV, ExxVC, ExxCC

##     def valence_core_core(nuclei, nspins):
##         """Determine the valence-core and core-core contributions for each
##            spin and nucleus
##         """

##         ExxCore = ExxValCore = 0.0
##         for nucleus in nuclei:
##             # error handling for old setup files
##             if nucleus.setup.ExxC == None:
##                 print 'Warning no exact exchange information in setup file'
##                 print 'Value of exact exchange may be incorrect'
##                 print 'Please regenerate setup file to correct error'
##                 break

##             # add core-core contribution from current nucleus
##             ExxCore += nucleus.setup.ExxC

##             # add val-core contribution from current nucleus
##             for spin in range(nspins):
##                 D_p = nucleus.D_sp[spin]
##                 ExxValCore += - num.dot(D_p, nucleus.setup.X_p)
        
##         return ExxValCore, ExxCore
##     valence_core_core = staticmethod(valence_core_core)
    
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
    else: raise RuntimeError('Unknown type of exchange: ', type)

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
                    # integrate density * potential
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

if __name__ == '__main__':
    from gridpaw.domain import Domain
    from gridpaw.grid_descriptor import GridDescriptor
    from ASE.Visualization.VTK import VTKPlotArray
    
    d  = Domain((4,4,4))      # domain object
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
