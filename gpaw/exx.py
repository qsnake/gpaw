# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange.

The eXact-eXchange energy functional is::

                                         *  _       _     * _        _
           __                /        phi  (r) phi (r) phi (r') phi (r')
       -1 \                  |  _  _      n       m       m        n
 E   = --  ) delta     f  f  | dr dr' ---------------------------------
  xx    2 /__     s s   n  m |                    _   _
           nm      n m       /                   |r - r'|
         
The action of the non-local exchange potential on an orbital is::

               /                         __
 ^             | _    _  _        _     \      _       _
 V   phi (r) = |dr' V(r, r') phi (r') =  ) V  (r) phi (r)
  xx    n      |                n       /__ nm       m
               /                         m

where::

                          _     * _
              __     psi (r) psi (r')
   _  _      \          m       m
 V(r, r') = - )  f   ----------------
             /__  m       _   _
              m          |r - r'|
              
and::

                        * _       _
                /    psi (r) psi (r')
     _          | _     m       n
 V  (r) = -  f  |dr' ----------------
  nm          m |         _   _
                /        |r - r'|

"""

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!

from gpaw.coulomb import Coulomb
from gpaw.utilities.tools import core_states, symmetrize
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
        self.nt_G = gd.empty()    # Pseudo density on coarse grid
        self.vt_G = gd.empty()    # Pot. of comp. pseudo density on coarse grid
        self.nt_g = finegd.empty()# Comp. pseudo density on fine grid
        self.vt_g = finegd.empty()# Pot. of comp. pseudo density on fine grid
        if not energy_only:
            nmyu = nspins # XXX only correct for serial calculation!
            self.vt_snG = gd.empty((nmyu, nbands))  

    def calculate_energy(self, kpt, Htpsit_nG, H_nn, hybrid):
        """Apply exact exchange operator."""

        # Initialize method-attributes
        psit_nG = kpt.psit_nG     # Wave functions
        Exx = Ekin = 0.0          # Energy of eXact eXchange and kinetic energy
        deg = 2 / self.nspins     # Spin degeneracy
        f_n = kpt.f_n             # Occupation number
        s   = kpt.s               # Global spin index
        u   = kpt.u               # Local spin/kpoint index
        if not self.energy_only:
            for nucleus in self.my_nuclei:
                nucleus.vxx_uni[u] = 0.0

        # Determine pseudo-exchange
        for n1, psit1_G in enumerate(psit_nG):
            f1 = f_n[n1]
            for n2, psit2_G in enumerate(psit_nG):
                f2 = f_n[n2]
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
                    Exx += 0.5 * f1 * f2 * hybrid / deg * int_fine
                    Ekin -= f1 * f2 * hybrid / deg * int_coarse

                if not self.energy_only:
                    Htpsit_nG[n1] += f2 * hybrid / deg * \
                                     self.vt_G * psit2_G
                    if n1 == n2:
                        self.vt_snG[u, n1] = f1 * hybrid / deg * self.vt_G
                    
                    # Update the vxx_uni and vxx_unii vectors of the nuclei,
                    # used to determine the atomic hamiltonian, and the 
                    # residuals
                    for nucleus in self.ghat_nuclei:
                        v_L = num.zeros((nucleus.setup.lmax + 1)**2, num.Float)
                        nucleus.ghat_L.integrate(self.vt_g, v_L)

                        if nucleus in self.my_nuclei:
                            v_ii = unpack(num.dot(nucleus.setup.Delta_pL, v_L))
                            nucleus.vxx_uni[u, n1] += (
                                f2 * hybrid / deg * num.dot(
                                v_ii, nucleus.P_uni[u, n2]))

                            if n1 == n2:
                                # XXX Check this:
                                nucleus.vxx_unii[u, n1] = (
                                    f2 * hybrid / deg * v_ii)
        
        # Apply the atomic corrections to the energy and the Hamiltonian matrix
        for nucleus in self.my_nuclei:
            # Ensure that calculation does not use extra soft comp. charges
            setup = nucleus.setup
            assert not setup.softgauss or isinstance(setup, AllElectronSetup)

            # error handling for old setup files
            if nucleus.setup.ExxC == None:
                print 'Warning no exact exchange information in setup file'
                print 'Value of exact exchange may be incorrect'
                print 'Please regenerate setup file  with "-x" option,'
                print 'to correct error'
                break

            # Add non-trivial corrections the Hamiltonian matrix
            if not self.energy_only:
                h_nn = symmetrize(inner(nucleus.P_uni[u], nucleus.vxx_uni[u]))
                H_nn += h_nn
                Ekin -= num.dot(f_n, num.diagonal(h_nn))

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
                    A = 0.0 # = C * D
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += C_pp[p13, p24] * D_ii[i3, i4]
                    if not self.energy_only:
                        p12 = packed_index(i1, i2, ni)
                        H_p[p12] -= 2 * hybrid / deg * A / ((i1!=i2) + 1)
                        Ekin += 2 * hybrid / deg * D_ii[i1, i2] * A
                    Exx -= hybrid / deg * D_ii[i1, i2] * A
            
            # Add valence-core exchange energy
            # --
            # >  X   D
            # --  ii  ii
            Exx -= hybrid * num.dot(D_p, setup.X_p)
            if not self.energy_only:
                H_p -= hybrid * setup.X_p
                Ekin += hybrid * num.dot(D_p, setup.X_p)

            # Add core-core exchange energy
            if s == 0:
                Exx += hybrid * nucleus.setup.ExxC

        # Update the class attributes
        if u == 0:
            self.Exx = self.Ekin = 0
        self.Exx += self.psum(Exx)
        self.Ekin += self.psum(Ekin)


def valence_valence_corrections(paw):
    """Determine the atomic corrections to the valence-valence interaction
    using method 1::

               -- 
        vv,a   \             a      a      a      a      a
       E     = /    f * f * P    * P    * P    * P    * C    
        xx     --    n   m   n,i1   m,i2   n,i3   m,i4   i1,i2,i3,i4
            n,m,i1,i2,i3,i4

    or method 2::
    
               -- 
        vv,a   \     a        a              a
       E     = /    D      * C            * D
        xx     --    i1,i3    i1,i2,i3,i4    i2,i4
            i1,i2,i3,i4

    This should be the same. (This is not always the case, see e.g. He)
    """
    ExxVV1 = ExxVV2 = 0.0
    #---------------------- METHOD 1 ------------------------
    for nucleus in paw.my_nuclei:
        for u, kpt in enumerate(paw.kpt_u):
            for n in range(paw.nbands):
                for m in range(n, paw.nbands):
                    # determine double count factor:
                    DC = 2 - (n == m)

                    # calculate joint occupation number
                    fnm = (kpt.f_n[n] * kpt.f_n[m]) * paw.nspins / 2.

                    # generate density matrix
                    Pm_i = nucleus.P_uni[u, m]
                    Pn_i = nucleus.P_uni[u, n]
                    D_ii = num.outerproduct(Pm_i,Pn_i)
                    D_p  = pack(D_ii, tolerance=1e3)

                    # C_iiii from setup file
                    C_pp  = nucleus.setup.M_pp

                    # add atomic contribution to val-val interaction
                    ExxVV1 += - fnm * num.dot(D_p, num.dot(C_pp, D_p)) * DC

    #---------------------- METHOD 2 ------------------------
    for nucleus in paw.my_nuclei:
        for s in range(paw.nspins):
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
                    ExxVV2 -= D_ii[i1, i2] * A * paw.nspins / 2.

    # Return result of the two different methods
    return ExxVV1, ExxVV2


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
    lmax = max(gen.l_j[:Njcore] + gen.vl_j)
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
                            A_mm += nv * num.outerproduct(G1c, G2c)
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = pack2(X_ii, tolerance=1e-8)
    return X_p
