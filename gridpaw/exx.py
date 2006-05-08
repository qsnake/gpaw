import Numeric as num
from Numeric import pi
from gridpaw.utilities.complex import real
from gridpaw.coulomb import Coulomb
from gridpaw.utilities.tools import pack

class PawExx:
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
        self.exx_single = Coulomb(paw.finegd).get_single_exchange

        # load interpolator
        self.interpolate = paw.interpolate

        # ensure that calculation is a Gamma point calculation
        if paw.wf.typecode == num.Complex:
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
                           method    = 'recip_gauss'
                           ):
        """Control method for the calculation of exact exchange energy.
           Allowed method names are 'real', 'recip_gauss', and 'recip_ewald'
        """
        paw = self.paw

        # only do calculation if not previously done
        if method != self.method:            
            # update calculation method
            self.method  = method

            # Get smooth pseudo exchange energy contribution
            self.Exxt = self.get_pseudo_exchange()

        # Get atomic corrections
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

        # return result, decompose if desired
        if decompose: return Exx
        else: return sum(Exx)

    def get_pseudo_exchange(self):
        """Calculate smooth contribution to exact exchange energy"""
        wf = self.paw.wf
        ghat_nuclei = self.paw.ghat_nuclei
        finegd = self.paw.finegd

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
                    Exxt += fnm * DC * self.exx_single(self.n_g, Z=Z,
                                                       method=self.method)
        return Exxt
    
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
                print 'Please regenerate setup file  with "-x" option,'
                print 'to correct error'
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
    # get Gaunt coefficients
    from gridpaw.gaunt import gaunt

    # get Hartree potential calculator
    from gridpaw.setup import Hartree

    # get core state counter
    from gridpaw.utilities.tools import core_states

    # maximum angular momentum
    Lmax = 2 * max(atom.l_j) + 1

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

    # get revised pack2 module
    from gridpaw.utilities.tools import pack2

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
    X_p = pack2(X_ii, symmetric=True, tol=1e-4)
    return X_p
