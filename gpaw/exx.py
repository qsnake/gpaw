import Numeric as num
from Numeric import pi
from gpaw.utilities.complex import real
from gpaw.coulomb import Coulomb
from gpaw.utilities.tools import pack, pack2, core_states
from gpaw.gaunt import make_gaunt
from gpaw.utilities import hartree, unpack

class XCHandler:
    """Exchange correlation handler.
    
    Handle the set of exchange and correlation functionals, of the form::

                 name     parameters
      xcdict = {'xLDA':  {'coeff': 0.2, 'scalarrel': True},
                'xEXX':  {'coeff': 0.8, 'screened': False}}
              
    """

    hooks = {'LDA': {'xLDA': {}, 'cLDA': {}},
             'PBE': {'xPBE': {}, 'cPBE': {}},
             'EXX': {'xEXX': {}}}
    
    def __init__(self, xcdict):
        self.set_xcdict(xcdict)

    def set_xcdict(self, xcdict):
        # ensure correct type
        if type(xcdict) == str:
            xcdict = hooks[xcdict]
        else:
            assert type(xcdict) == dict

        # Check if it is hybrid calculation
        if 'xEXX' in xcdict:
            self.hybrid = xcdict['xExx'].get('coeff', 1.0)
        else:
            self.hybrid = 0.0

        # make list of xc-functionals
        self.functional_list = []
        for xc, par in xcdict.items():
            self.functional_list.append(XCFunctional(xc, **par))
        
    def calculate_spinpaired(self, *args):
        E = 0.0
        for xc in self.functional_list:
            E += xc.calculate_spinpaired(*args)
        return E
    
    def calculate_spinpolarized(self, *args):
        E = 0.0
        for xc in self.functional_list:
            E += xc.calculate_spinpolarized(*args)
        return E

class XXFunctional:
    def calculate_spinpaired(self, *args):
        return 0.0
    def calculate_spinpolarized(self, *args):
        return 0.0    

def get_exx(xcname, softgauss, typecode, gd, finegd, interpolate,
            my_nuclei, ghat_nuclei, nspins):
    if xcname != 'EXX':
        return None
    
    else:
        # ensure that calculation is a Gamma point calculation
        if typecode == num.Complex:
            msg = 'k-point calculations with exact exchange has not yet\n'\
                  'been implemented. Please use gamma point only.'
            raise NotImplementedError(msg)
        
        # ensure that softgauss option is false
        if softgauss:
            msg = 'Exact exchange is currently not compatible with extra\n'\
                  'soft compensation charges.\n'\
                  'Please set keyword softgauss=False'
            raise NotImplementedError(msg)

        return SelfConsistentExx(gd, finegd, interpolate,
                                 my_nuclei, ghat_nuclei, nspins)

class SelfConsistentExx:
    """Class offering methods for selfconsistent evaluation of the
       exchange energy of a gridPAW calculation.
    """
    def __init__(self, gd, finegd, interpolate,
                 my_nuclei, ghat_nuclei, nspins):
        self.my_nuclei = my_nuclei
        self.ghat_nuclei = ghat_nuclei
        self.nspins = nspins
        self.interpolate = interpolate

        # allocate space for matrices
        self.n_G = gd.new_array()
        self.v_G = gd.new_array()
        self.n_g = finegd.new_array()
        self.v_g = finegd.new_array()

        self.integrate = finegd.integrate
        self.Exx = 0.0

    def adjust_hamiltonian(self, psit_nG, Htpsit_nG, nbands,
                           f_n, u, s, poisson, restrict):
        """                  ~  ~
           Adjust values of  H psi due to inclusion of exact exchange.
           Called from kpoint.hamilton.diagonalize.
        """
        if s == 0:
            self.Exx = 0.0
        for n in range(nbands):
            for m in range(nbands):
                # determine current exchange density
                self.n_G[:] = psit_nG[m] * psit_nG[n] 

                # and interpolate to the fine grid
                self.interpolate(self.n_G, self.n_g)

                # determine the compensation charges for each nucleus
                for a, nucleus in enumerate(self.ghat_nuclei):
                    if nucleus.in_this_domain:
                        # generate density matrix
                        Pm_i = nucleus.P_uni[u, m]
                        Pn_i = nucleus.P_uni[u, n]
                        D_ii = num.outerproduct(Pm_i, Pn_i)
                        D_p  = pack(D_ii, symmetric=False)

                        # determine compensation charge coefficients
                        Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    else:
                        Q_L = None

                    # add compensation charges to exchange density
                    nucleus.ghat_L.add(self.n_g, Q_L, communicate=True)

                # determine total charge of exchange density
                Z = float(n == m)

                # determine exchange potential
                poisson.solve(self.v_g, -self.n_g, charge=-Z)

                # update hamiltonian
                restrict(self.v_g, self.v_G)
                
                Htpsit_nG[n] += f_n[m] / (self.nspins % 2 + 1) *\
                                self.v_G * psit_nG[m]

                # add the nm contribution to exchange energy
                self.Exx -= .5 * f_n[n] * f_n[m] / (self.nspins % 2 + 1) *\
                            self.integrate(self.v_g * self.n_g)

                # update the vxx_sni vector of the nuclei, used to determine
                # the atomic hamiltonian
                for nucleus in self.my_nuclei:
                    v_L = num.zeros((nucleus.setup.lmax + 1)**2, num.Float)
                    nucleus.ghat_L.integrate(self.v_g, v_L)
                    nucleus.vxx_sni[s, n] += num.dot(
                        unpack(num.dot(nucleus.setup.Delta_pL, v_L)),
                        nucleus.P_uni[u, m])
        print 'Exchange energy:', self.Exx

    def adjust_hamitonian_matrix(self, H_nn, P_ni, nucleus, s):
        """Called from kpoint.diagonalize"""
        H_nn += num.dot(P_ni, num.transpose(nucleus.vxx_sni[s]))

    def adjust_residual(self, R_nG):
        """from the nucleus class"""
        pass

    def adjust_residual2(self, R_G):
        """from the nucleus class"""
        pass

class PerturbativeExx:
    """Class offering methods for non-selfconsistent evaluation of the
       exchange energy of a gridPAW calculation.
    """
    def __init__(self, paw):
        # store options in local varibles
        self.paw = paw
        self.method = None

        # allocate space for fine grid density
        self.n_g = paw.finegd.new_array()

        # load single exchange calculator
        self.coulomb = Coulomb(paw.finegd,
                                  paw.hamiltonian.poisson).coulomb
                 ## ---------->>> get_single_exchange <<<------

        # load interpolator
        self.interpolate = paw.density.interpolate

        # ensure that calculation is a Gamma point calculation
        if paw.typecode == num.Complex:
            msg = 'k-point calculations with exact exchange has not yet\n'\
                  'been implemented. Please use gamma point only.'
            raise NotImplementedError(msg)

        # ensure that softgauss option is false
        if paw.nuclei[0].setup.softgauss:
            msg = 'Exact exchange is currently not compatible with extra\n'\
                  'soft compensation charges.\n'\
                  'Please set keyword softgauss=False'
            raise NotImplementedError(msg)
            
    def get_exact_exchange(self,
                           decompose = False,
                           method    = None
                           ):
        """Control method for the calculation of exact exchange energy.
           Allowed method names are 'real', 'recip_gauss', and 'recip_ewald'
        """
        paw = self.paw

        if method == None:
            if paw.domain.comm.size == 1: # serial computation
                method = 'recip_gauss'
            else: # parallel computation
                method = 'real'
        self.method = method

        # Get smooth pseudo exchange energy contribution
        self.Exxt = self.get_pseudo_exchange()

        # Get atomic corrections
        self.ExxVV, self.ExxVC, self.ExxCC = self.atomic_corrections()
        
        # sum contributions from all processors
        ksum = paw.kpt_comm.sum
        dsum = paw.domain.comm.sum
        self.Exxt  = ksum(self.Exxt)
        self.ExxVV = ksum(dsum(self.ExxVV))
        self.ExxVC = ksum(dsum(self.ExxVC))
        self.ExxCC = ksum(dsum(self.ExxCC))

        # add all contributions, to get total exchange energy
        Exx = num.array([self.Exxt, self.ExxVV, self.ExxVC, self.ExxCC])

        # return result, decompose if desired
        if decompose: return Exx
        else: return sum(Exx)

    def get_pseudo_exchange(self):
        """Calculate smooth contribution to exact exchange energy"""
        paw = self.paw
        ghat_nuclei = paw.ghat_nuclei
        finegd = paw.finegd

        # calculate exact exchange of smooth wavefunctions
        Exxt = 0.0
        for u, kpt in enumerate(paw.kpt_u):
            #spin = paw.myspins[u] # global spin index XXX
            for n in range(paw.nbands):
                for m in range(n, paw.nbands):
                    # determine double count factor:
                    DC = 2 - (n == m)

                    # calculate joint occupation number
                    fnm = (kpt.f_n[n] * kpt.f_n[m]) * paw.nspins / 2.

                    # determine current exchange density
                    n_G = kpt.psit_nG[m] * kpt.psit_nG[n] 

                    # and interpolate to the fine grid
                    self.interpolate(n_G, self.n_g)

                    # determine the compensation charges for each nucleus
                    for a, nucleus in enumerate(ghat_nuclei):
                        if nucleus.in_this_domain:
                            # generate density matrix
                            Pm_i = nucleus.P_uni[u, m] # spin ??
                            Pn_i = nucleus.P_uni[u, n] # spin ??
                            D_ii = num.outerproduct(Pm_i,Pn_i)
                            D_p  = pack(D_ii, symmetric=False)
                            
                            # determine compensation charge coefficients
                            Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                        else:
                            Q_L = None

                        # add compensation charges to exchange density
                        nucleus.ghat_L.add(self.n_g, Q_L, communicate=True)

                    # determine total charge of exchange density
                    Z = float(n == m)

                    # add the nm contribution to exchange energy
                    Exxt += -.5 * fnm * DC * self.coulomb(self.n_g, Z1=Z,
                                                       method=self.method)
        return Exxt
    
    def atomic_corrections(self):
        """Determine the atomic corrections to the valence-valence exchange
           interaction, the valence-core contribution, and the core-core
           contributions.
        """
        ExxVV = ExxVC = ExxCC = 0.0
        for nucleus in self.paw.my_nuclei:
            # error handling for old setup files
            if nucleus.setup.ExxC == None:
                print 'Warning no exact exchange information in setup file'
                print 'Value of exact exchange may be incorrect'
                print 'Please regenerate setup file  with "-x" option,'
                print 'to correct error'
                break

            for kpt in self.paw.kpt_u:
                # Add core-core contribution:
                s = kpt.s
                if s == 0:
                    ExxCC += nucleus.setup.ExxC

                # Add val-core contribution:
                #              __
                #     vc,a    \     a    a
                #    E     = - )   D  * X
                #     xx      /__   p    p
                #              p

                D_p = nucleus.D_sp[s]
                ExxVC += - num.dot(D_p, nucleus.setup.X_p)

            """Determine the atomic corrections to the val-val interaction:
                 
                       -- 
                vv,a   \     a        a              a
               E     = /    D      * C            * D
                xx     --    i1,i3    i1,i2,i3,i4    i2,i4
                    i1,i2,i3,i4

                       -- 
                vv,a   \             a      a      a      a      a
               E     = /    f * f * P    * P    * P    * P    * C    
                xx     --    n   m   n,i1   m,i2   n,i3   m,i4   i1,i2,i3,i4
                    n,m,i1,i2,i3,i4
            """
            for u, kpt in enumerate(self.paw.kpt_u):
                #spin = paw.myspins[u] # global spin index XXX
                for n in range(self.paw.nbands):
                    for m in range(n, self.paw.nbands):
                        # determine double count factor:
                        DC = 2 - (n == m)

                        # calculate joint occupation number
                        fnm = (kpt.f_n[n] * kpt.f_n[m]) * self.paw.nspins / 2.

                        # generate density matrix
                        Pm_i = nucleus.P_uni[u, m] # spin ??
                        Pn_i = nucleus.P_uni[u, n] # spin ??
                        D_ii = num.outerproduct(Pm_i,Pn_i)
                        D_p  = pack(D_ii, symmetric=False)

                        # C_iiii from setup file
                        C_pp  = nucleus.setup.M_pp

                        # add atomic contribution to val-val interaction
                        ExxVV += - fnm * num.dot(D_p, num.dot(C_pp, D_p)) * DC

        return ExxVV, ExxVC, ExxCC

def atomic_exact_exchange(atom, type = 'all'):
    """Returns the exact exchange energy of the atom defined by the
       instantiated AllElectron object 'atom'
    """

    # make gaunt coeff. list
    gaunt = make_gaunt(lmax=max(atom.l_j))

    # number of valence, Nj, and core, Njcore, orbitals
    Nj     = len(atom.n_j)
    Njcore = core_states(atom.symbol)

    # determine relevant states for chosen type of exchange contribution
    if type == 'all': nstates = mstates = range(Nj)
    elif type == 'val-val': nstates = mstates = range(Njcore,Nj)
    elif type == 'core-core': nstates = mstates = range(Njcore)
    elif type == 'val-core':
        nstates = range(Njcore,Nj)
        mstates = range(Njcore)
    else: raise RuntimeError('Unknown type of exchange: ', type)

    vr = num.zeros(atom.N, num.Float)
    vrl = num.zeros(atom.N, num.Float)
    
    # do actual calculation of exchange contribution
    Exx = 0.0
    for j1 in nstates:
        # angular momentum of first state
        l1 = atom.l_j[j1]

        for j2 in mstates:
            # angular momentum of second state
            l2 = atom.l_j[j2]

            # joint occupation number
            f12 = .5 * atom.f_j[j1] / (2. * l1 + 1) * \
                       atom.f_j[j2] / (2. * l2 + 1)

            # electron density times radius times length element
            nrdr = atom.u_j[j1] * atom.u_j[j2] * atom.dr
            nrdr[1:] /= atom.r[1:]

            # potential times radius
            vr[:] = 0.0

            # L summation
            for l in range(l1 + l2 + 1):
                # get potential for current l-value
                hartree(l, nrdr, atom.beta, atom.N, vrl)

                # take all m1 m2 and m values of Gaunt matrix of the form
                # G(L1,L2,L) where L = {l,m}
                G2 = gaunt[l1**2:(l1+1)**2, l2**2:(l2+1)**2, l**2:(l+1)**2]**2

                # add to total potential
                vr += vrl * num.sum(G2.copy().flat)

            # add to total exchange the contribution from current two states
            Exx += -.5 * f12 * num.dot(vr, nrdr)

    # double energy if mixed contribution
    if type == 'val-core': Exx *= 2.

    # return exchange energy
    return Exx

def constructX(gen):
    """Construct the X_p^a matrix for the given atom"""
    # make gaunt coeff. list
    gaunt = make_gaunt(lmax=max(gen.l_j))

    uv_j = gen.vu_j    # soft valence states * r:
    lv_j = gen.vl_j    # their repective l quantum numbers
    Nvi  = 0 
    for l in lv_j:
        Nvi += 2 * l + 1   # total number of valence states (including m)

    # number of core and valence orbitals (j only, i.e. not m-number)
    Njcore = gen.njcore
    Njval  = len(lv_j)

    # core states * r:
    uc_j = gen.u_j[:Njcore]
    r, dr, N, beta = gen.r, gen.dr, gen.N, gen.beta

    # potential times radius
    vr = num.zeros(N, num.Float)
        
    # initialize X_ii matrix
    X_ii = num.zeros((Nvi, Nvi), num.Float)

    # sum over core states
    for jc in range(Njcore):
        lc = gen.l_j[jc]

        # sum over first valence state index
        i1 = 0
        for jv1 in range(Njval):
            lv1 = lv_j[jv1] 

            # electron density 1 times radius times length element
            n1c = uv_j[jv1] * uc_j[jc] * dr
            n1c[1:] /= r[1:]

            # sum over second valence state index
            i2 = 0
            for jv2 in range(Njval):
                lv2 = lv_j[jv2]
                
                # electron density 2
                n2c = uv_j[jv2] * uc_j[jc] * dr
                n2c[1:] /= r[1:]
            
                # sum expansion in angular momenta
                for l in range(min(lv1,lv2) + lc + 1):
                    # Int density * potential * r^2 * dr:
                    hartree(l, n2c, beta, N, vr)
                    nv = num.dot(n1c, vr)
                    
                    # expansion coefficients
                    A_mm = X_ii[i1:i1 + 2 * lv1 + 1, i2:i2 + 2 * lv2 + 1]
                    for mc in range(2*lc+1):
                        for m in range(2*l+1):
                            G1c = gaunt[lv1**2:(lv1 + 1)**2,
                                        lc**2+mc,l**2 + m]
                            G2c = gaunt[lv2**2:(lv2 + 1)**2,
                                        lc**2+mc,l**2 + m]
                            A_mm += nv * num.outerproduct(G1c, G2c)
                            
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = pack2(X_ii, symmetric=True, tol=1e-8)
    return X_p
