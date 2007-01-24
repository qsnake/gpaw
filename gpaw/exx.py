# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange."""

import Numeric as num
from Numeric import pi
from gpaw.coulomb import Coulomb
from gpaw.utilities.tools import core_states
from gpaw.gaunt import make_gaunt
from gpaw.utilities import hartree, packed_index, unpack, unpack2, pack, pack2
from gpaw.ae import AllElectronSetup

class XXFunctional:
    """Dummy EXX functional"""
    def calculate_spinpaired(self, e_g, n_g, v_g):
        e_g[:] = 0.0    

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        e_g[:] = 0.0    
        
class EXX:
    """EXact eXchange.

    Class offering methods for selfconsistent evaluation of the
    exchange energy."""
    
    def __init__(self, gd, finegd, interpolate, restrict, poisson,
                 my_nuclei, ghat_nuclei, nspins, nbands, kcomm, dcomm,
                 energy_only=False):

        # Initialize class-attributes
        self.nspins      = nspins
        self.my_nuclei   = my_nuclei
        self.ghat_nuclei = ghat_nuclei
        self.interpolate = interpolate
        self.restrict    = restrict
        self.poisson     = poisson
        self.rank        = dcomm.rank
        self.psum        = lambda x: kcomm.sum(dcomm.sum(x))
        self.energy_only = energy_only
        self.integrate   = gd.integrate
        self.fineintegrate = finegd.integrate
        
        # Allocate space for matrices
        self.nt_G = gd.empty()
        self.vt_G = gd.empty()
        self.nt_g = finegd.empty()
        self.vt_g = finegd.empty()
        if not energy_only:
            self.vt_nG = gd.empty(nbands)

    def calculate_energy(self, kpt, Htpsit_nG, H_nn, hybrid):
        """Apply exact exchange operator."""

        # Initialize method-attributes
        psit_nG = kpt.psit_nG[:]  # wave functions
        Exx = Ekin = 0.0          # Energy of eXact eXchange and kinetic energy
        deg = 2 / self.nspins     # spin degeneracy
        f_n = kpt.f_n             # occupation number
        s   = kpt.s               # global spin index
        u   = kpt.u               # local spin/kpoint index
        if not self.energy_only:
            for nucleus in self.my_nuclei:
                nucleus.vxx_uni[:] = 0.0

        # Determine pseudo-exchange
        for n1, psit1_G in enumerate(psit_nG):
            for n2, psit2_G in enumerate(psit_nG):
                # Determine current exchange density ...
                self.nt_G[:] = psit1_G * psit2_G 

                # and interpolate to the fine grid:
                self.interpolate(self.nt_G, self.nt_g)

                # Determine the compensation charges for each nucleus:
                for nucleus in self.ghat_nuclei:
                    if nucleus.in_this_domain:
                        # Generate density matrix
                        P1_i = nucleus.P_uni[u, n1]
                        P2_i = nucleus.P_uni[u, n2]
                        D_ii = num.outerproduct(P1_i, P2_i)
                        D_p  = pack(D_ii, tolerance=1e3)

                        # Determine compensation charge coefficients:
                        Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    else:
                        Q_L = None

                    # Add compensation charges to exchange density:
                    nucleus.ghat_L.add(self.nt_g, Q_L, communicate=True)

                # Determine total charge of exchange density:
                Z = float(n1 == n2)

                # Determine exchange potential:
                self.vt_g[:] = 0.0
                self.poisson.solve(self.vt_g, -self.nt_g, charge=-Z)
                self.restrict(self.vt_g, self.vt_G)

                # Integrate the potential on fine and coarse grids
                int_fine = self.fineintegrate(self.vt_g * self.nt_g)
                int_coarse = self.integrate(self.vt_G * self.nt_G)
                if self.rank == 0: # Only add to energy on master CPU
                    Exx += 0.5 * f_n[n1] * f_n[n2] * hybrid / deg * int_fine
                    Ekin -= f_n[n1] * f_n[n2] * hybrid / deg * int_coarse

                if not self.energy_only:
                    Htpsit_nG[n1] += f_n[n2] * hybrid / deg * \
                                     self.vt_G * psit2_G
                    if n1 == n2:
                        self.vt_nG[n1] = f_n[n1] * hybrid / deg * self.vt_G
                    
                    # Update the vxx_uni and vxx_unii vectors of the nuclei,
                    # used to determine the atomic hamiltonian, and the 
                    # residuals
                    for nucleus in self.ghat_nuclei:
                        v_L = num.zeros((nucleus.setup.lmax + 1)**2, num.Float)
                        nucleus.ghat_L.integrate(self.vt_g, v_L)

                        if nucleus in self.my_nuclei:
                            v_ii = unpack(num.dot(nucleus.setup.Delta_pL, v_L))
                            nucleus.vxx_uni[u, n1] += (
                                f_n[n2] * hybrid / deg * num.dot(
                                v_ii, nucleus.P_uni[u, n2]))

                            if n1 == n2:
                                nucleus.vxx_unii[u, n1] = (
                                    f_n[n2] * hybrid / deg * v_ii)
        
        # Apply the atomic corrections to the energy and the Hamiltonian matrix
        for nucleus in self.my_nuclei:
            # Ensure that calculation does not use extra soft comp. charges
            setup = nucleus.setup
            assert not setup.softgauss or isinstance(setup, AllElectronSetup)

            # Add non-trivial corrections the Hamiltonian matrix
            if not self.energy_only:
                H_nn += num.innerproduct(nucleus.vxx_uni[u], nucleus.P_uni[u])

            # Get atomic density and Hamiltonian matrices
            D_p  = nucleus.D_sp[s]
            D_ii = unpack2(D_p)
            H_p  = nucleus.H_sp[s]
            ni = len(D_ii)
            
            # Add atomic corrections to the valence-valence exchange energy
            # --
            # >  D   C     D
            # --  ii  iiii  ii
            C_pp = setup.M_pp
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += C_pp[p13, p24] * D_ii[i3, i4]
                    if not self.energy_only and i1 > i2:
                        p12 = packed_index(i1, i2, ni)
                        H_p[p12] -= 2 * hybrid * A # XXX: No '/ deg' ???
                    Exx -= hybrid / deg * D_ii[i1, i2] * A
            
            # Add valence-core exchange energy
            # --
            # >  X   D
            # --  ii  ii
            Exx -= hybrid * num.dot(D_p, setup.X_p)
            if not self.energy_only:
                H_p -= hybrid * setup.X_p

            # Add core-core exchange energy
            if s == 0:
                Exx += hybrid * nucleus.setup.ExxC

        # Update the class attributes
        if u == 0:
            self.Exx = self.Ekin = 0
        self.Exx += self.psum(Exx)
        self.Ekin += self.psum(Ekin)
    
class PerturbativeExx:
    """Class offering methods for non-selfconsistent evaluation of the
       exchange energy of a *gpaw* calculation.
    """
    def __init__(self, paw):
        # store options in local varibles
        self.paw = paw
        self.method = None

        # allocate space for fine grid density
        self.n_g = paw.finegd.new_array()

        # load single exchange calculator
        self.coulomb = Coulomb(paw.finegd, paw.hamiltonian.poisson).coulomb

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
                    for nucleus in ghat_nuclei:
                        if nucleus.in_this_domain:
                            # generate density matrix
                            Pm_i = nucleus.P_uni[u, m]
                            Pn_i = nucleus.P_uni[u, n]
                            D_ii = num.outerproduct(Pm_i, Pn_i)
                            D_p  = pack(D_ii, tolerance=1e3)
                            
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
                for n in range(self.paw.nbands):
                    for m in range(n, self.paw.nbands):
                        # determine double count factor:
                        DC = 2 - (n == m)

                        # calculate joint occupation number
                        fnm = (kpt.f_n[n] * kpt.f_n[m]) * self.paw.nspins / 2.

                        # generate density matrix
                        Pm_i = nucleus.P_uni[u, m]
                        Pn_i = nucleus.P_uni[u, n]
                        D_ii = num.outerproduct(Pm_i,Pn_i)
                        D_p  = pack(D_ii, tolerance=1e3)

                        # C_iiii from setup file
                        C_pp  = nucleus.setup.M_pp

                        # add atomic contribution to val-val interaction
                        ExxVV += - fnm * num.dot(D_p, num.dot(C_pp, D_p)) * DC

        return ExxVV, ExxVC, ExxCC
        #---------------------- TEST STUFF ------------------------
        ExxVV_TEST = 0.0
        for nucleus in self.paw.my_nuclei:
            for s in range(self.paw.nspins):
                D_p  = nucleus.D_sp[s]
                D_ii = unpack2(D_p)
                C_pp = nucleus.setup.M_pp
                ni   = len(D_ii)
                for i1 in range(ni):
                    for i2 in range(ni):
                        A = 0.0
                        for i3 in range(ni):
                            p13 = packed_index(i1, i3, ni)
                            for i4 in range(ni):
                                p24 = packed_index(i2, i4, ni)
                                A += C_pp[p13, p24] * D_ii[i3, i4]
                        ExxVV_TEST -= D_ii[i1, i2] * A * self.paw.nspins / 2.
        print 'Test of D*C*D summation: %0.7f %0.7f %0.7f' % tuple(num.array(
            [ExxVV, ExxVV_TEST, ExxVV - ExxVV_TEST]) * 27.211395655517311)
        #---------------------- TEST STUFF ------------------------
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
    """Construct the X_p^a matrix for the given atom.

    The X_p^a matrix describes the valence-core interactions of the
    partial waves.
    """
    # initialize attributes
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

    # make gaunt coeff. list
    if Njcore == 0:
        lmax = 0
    else:
        lmax = max(gen.l_j[:Njcore])
    gaunt = make_gaunt(lmax=lmax)

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
                for l in range(min(lv1, lv2) + lc + 1):
                    # Int density * potential * r^2 * dr:
                    hartree(l, n2c, beta, N, vr)
                    nv = num.dot(n1c, vr)
                    
                    # expansion coefficients
                    A_mm = X_ii[i1:i1 + 2 * lv1 + 1, i2:i2 + 2 * lv2 + 1]
                    for mc in range(2 * lc + 1):
                        for m in range(2 * l + 1):
                            G1c = gaunt[lv1**2:(lv1 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            G2c = gaunt[lv2**2:(lv2 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            print G1c.shape, G2c.shape, A_mm.shape
                            A_mm += nv * num.outerproduct(G1c, G2c)
                            
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = pack2(X_ii, tolerance=1e-8)
    return X_p
