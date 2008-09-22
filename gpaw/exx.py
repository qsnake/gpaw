# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange.

The eXact-eXchange energy functional is::

                                         *  _       _     * _        _
           __                /        phi  (r) phi (r) phi (r') phi (r')
       -1 \                  |  _  _      n       m       m        n
 E   = --  ) delta     f  f  | dr dr' ---------------------------------  (5.1)
  xx    2 /__     s s   n  m |                    _   _
           nm      n m       /                   |r - r'|
         
The action of the non-local exchange potential on an orbital is::

               /                         __
 ^             | _    _  _        _     \      _       _
 V   phi (r) = |dr' V(r, r') phi (r') =  ) V  (r) phi (r)               (5.3)
  xx    n      |                n       /__ nm       m
               /                         m

where::

                          _     * _
              __     psi (r) psi (r')
   _  _      \          m       m
 V(r, r') = - )  f   ----------------                                   (5.4a)
             /__  m       _   _
              m          |r - r'|
              
and::

                        * _       _
                /    psi (r) psi (r')
     _          | _     m       n
 V  (r) = -  f  |dr' ----------------                                   (5.4b)
  nm          m |         _   _
                /        |r - r'|

Equation numbers as in
'Exact Exchange in Density Functional Calculations'
Masters Thesis by Carsten Rostgaard, CAMP 2006
"""

import numpy as npy

from gpaw.utilities.complex import real
from gpaw.utilities.tools import core_states, symmetrize
from gpaw.gaunt import make_gaunt
from gpaw.utilities import hartree, packed_index, unpack, unpack2, pack, pack2
from gpaw.ae import AllElectronSetup
from gpaw.utilities.blas import gemm
from gpaw.pair_density import PairDensity2 as PairDensity
from gpaw.poisson import PoissonSolver, PoissonFFTSolver

usefft = False
verbose = False

## for debug
## from gpaw.mpi.mpiprint import mpiprint

def dummy_interpolate(nt_G, nt_g):
    nt_g[:] = nt_G


class EXX:
    """EXact eXchange.

    Class offering methods for selfconsistent evaluation of the
    exchange energy."""
    
    def __init__(self, paw, energy_only=False, use_finegrid=True):
        # Initialize class-attributes
        self.nspins        = paw.nspins
        self.nbands        = paw.nbands
        self.my_nuclei     = paw.my_nuclei
        self.interpolate   = paw.density.interpolate
        self.restrict      = paw.hamiltonian.restrict
        self.integrate     = paw.gd.integrate
        self.fineintegrate = paw.finegd.integrate
        self.pair_density  = PairDensity(paw, finegrid=True)
        self.ghat_nuclei   = paw.ghat_nuclei
        self.rank          = paw.domain.comm.rank
        ksum, dsum = paw.kpt_comm.sum, paw.domain.comm.sum
        self.psum          = lambda x: ksum(dsum(x)) # Sum over all processors
        self.energy_only   = energy_only
        self.Na            = len(paw.nuclei)
        self.use_finegrid  = use_finegrid
        self.pair_density  = PairDensity(paw, use_finegrid)
        self.poisson_solve = paw.hamiltonian.poisson.solve
        self.Exx = 0.
        self.Ekin = 0.

        # Set correct Poisson solver
        if usefft:
            if use_finegrid:
                solver = PoissonFFTSolver()
                solver.initialize(paw.finegd)
                self.poisson_solve = solver.solve
            else:
                solver = PoissonFFTSolver()
                solver.initialize(paw.gd)
                self.poisson_solve = solver.solve
        elif not use_finegrid:
            solver = PoissonSolver(nn=paw.hamiltonian.poisson.nn)
            solver.initialize(paw.gd)
            self.poisson_solve = solver.solve
            
        # Allocate space for matrices
        self.nt_G = paw.gd.empty()# Pseudo density on coarse grid
        self.rhot_g = paw.finegd.empty()# Comp. pseudo density on fine grid
        self.vt_G = paw.gd.empty()# Pot. of comp. pseudo density on coarse grid
        self.vt_g = paw.finegd.empty()# Pot. of comp. pseudo dens. on fine grid

        # Overwrites in case of coarse grid Poisson solver
        if not use_finegrid:
            self.fineintegrate = paw.gd.integrate
            self.interpolate = dummy_interpolate
            self.rhot_g = paw.gd.empty()
            self.vt_g = self.vt_G
        
        # For rotating the residuals we need the diagonal Fock potentials
        if not energy_only:
            self.vt_unG = paw.gd.zeros((paw.nmyu, paw.nbands))

    def apply(self, kpt, Htpsit_nG, H_nn, hybrid):
        """Apply exact exchange operator."""

        # Initialize method-attributes
        psit_nG = kpt.psit_nG   # Wave functions
        Exx = Ekin = 0.0        # Energy of eXact eXchange and kinetic energy
        deg = 2 / self.nspins   # Spin degeneracy
        f_n = kpt.f_n           # Occupation number
        s   = kpt.s             # Global spin index
        u   = kpt.u             # Local spin/kpoint index
        pd  = self.pair_density # Class for handling pair densities
        fmin= 1e-9              # Occupations less than this counts as empty

        if not self.energy_only:
            for nucleus in self.my_nuclei:
                nucleus.vxx_uni[u] = 0.0

        # Determine pseudo-exchange
        for n1 in range(self.nbands):
            psit1_G = psit_nG[n1]
            f1 = f_n[n1] * hybrid / deg
            for n2 in range(n1, self.nbands):
                psit2_G = psit_nG[n2]
                f2 = f_n[n2] * hybrid / deg
                if f1 < fmin and f2 < fmin:
                    continue # Don't do anything if both occupations are small
                
                dc = (1 + (n1 != n2)) * deg / hybrid  # double count factor
                pd.initialize(kpt, n1, n2)

                # Determine current exchange density
                pd.get_coarse(self.nt_G)
                pd.add_compensation_charges(self.nt_G, self.rhot_g)

                # Re-use stored potential if possible
                zero_phi = True
                if 0: #n1 == n2 and hasattr(self, 'vt_unG'):
                    zero_phi = False
                    if self.use_finegrid:
                        self.interpolate(self.vt_unG[u, n2] / f2, self.vt_g)
                    else:
                        npy.divide(self.vt_unG[u, n2], f2, self.vt_g)
                    
                # Determine exchange potential:
                iter = self.poisson_solve(self.vt_g, -self.rhot_g,
                                          charge=-float(n1 == n2), eps=1e-12,
                                          zero_initial_phi=zero_phi)
                if verbose:
                    print 'EXX: n1=%s, n2=%s, iterations=%s' % (n1, n2, iter)

                if self.use_finegrid:
                    self.restrict(self.vt_g, self.vt_G)
                else:
                    assert self.vt_G is self.vt_g

                # Integrate the potential on fine and coarse grids
                int_fine = self.fineintegrate(self.vt_g * self.rhot_g)
                int_coarse = self.integrate(self.vt_G * self.nt_G)
                if self.rank == 0: # Only add to energy on master CPU
                    Exx += 0.5 * f1 * f2 * dc * int_fine
                    Ekin -= f1 * f2 * dc * int_coarse

                if not self.energy_only:
                    Htpsit_nG[n1] += f2 * self.vt_G * psit2_G
                    if n1 != n2:
                        Htpsit_nG[n2] += f1 * self.vt_G * psit1_G
                    else:
                        self.vt_unG[u, n2] = f2 * self.vt_G

                    # Update the vxx_uni and vxx_unii vectors of the nuclei,
                    # used to determine the atomic hamiltonian, and the 
                    # residuals
                    for nucleus in self.ghat_nuclei:
                        v_L = npy.zeros((nucleus.setup.lmax + 1)**2)
                        if self.use_finegrid:
                            nucleus.ghat_L.integrate(self.vt_g, v_L)
                        else:
                            if nucleus.Ghat_L is not None:
                                nucleus.Ghat_L.integrate(self.vt_G, v_L)

                        if nucleus.in_this_domain:
                            v_ii = unpack(npy.dot(nucleus.setup.Delta_pL, v_L))
                            v_ni = nucleus.vxx_uni[u]
                            v_nii = nucleus.vxx_unii[u]

                            v_ni[n1] += f2 * npy.dot(v_ii,nucleus.P_uni[u, n2])
                            if n1 != n2:
                                v_ni[n2] += f1 * npy.dot(v_ii,
                                                         nucleus.P_uni[u, n1])
                            else:
                                # XXX Check this:
                                v_nii[n1] = f2 * v_ii

        # Apply the atomic corrections to the energy and the Hamiltonian matrix
        for nucleus in self.my_nuclei:
            setup = nucleus.setup

            # Add non-trivial corrections the Hamiltonian matrix
            if not self.energy_only:
                h_nn = symmetrize(npy.inner(nucleus.P_uni[u],
                                            nucleus.vxx_uni[u]))
                H_nn += h_nn
                Ekin -= npy.dot(f_n, npy.diagonal(h_nn))

            # Get atomic density and Hamiltonian matrices
            D_p  = nucleus.D_sp[s]
            D_ii = unpack2(D_p)
            H_p  = nucleus.H_sp[s]
            ni = len(D_ii)
            
            # Add atomic corrections to the valence-valence exchange energy
            # --
            # >  D   C     D
            # --  ii  iiii  ii
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                    if not self.energy_only:
                        p12 = packed_index(i1, i2, ni)
                        H_p[p12] -= 2 * hybrid / deg * A / ((i1 != i2) + 1)
                        Ekin += 2 * hybrid / deg * D_ii[i1, i2] * A
                    Exx -= hybrid / deg * D_ii[i1, i2] * A
            
            # Add valence-core exchange energy
            # --
            # >  X   D
            # --  ii  ii
            Exx -= hybrid * npy.dot(D_p, setup.X_p)
            if not self.energy_only:
                H_p -= hybrid * setup.X_p
                Ekin += hybrid * npy.dot(D_p, setup.X_p)

            # Add core-core exchange energy
            if s == 0:
                Exx += hybrid * setup.ExxC

        # Update the class attributes
        if u == 0:
            self.Exx = self.Ekin = 0
        self.Exx += self.psum(Exx)
        self.Ekin += self.psum(Ekin)

    def force_kpoint(self, kpt, hybrid):
        """Force due to exact exchange operator"""

        deg = 2 / self.nspins
        u = kpt.u
        F_ac = npy.zeros((self.Na, 3))
        fmin = 1.e-10
        pd = self.pair_density

        for n1 in range(self.nbands):
            psit1_G = kpt.psit_nG[n1]
            f1 = kpt.f_n[n1] * hybrid / deg
            for n2 in range(n1, self.nbands):
                psit2_G = kpt.psit_nG[n2]
                f2 = kpt.f_n[n2] * hybrid / deg
                if f1 < fmin and f2 < fmin:
                    continue

                # Re-determine all of the exhange potentials
                dc = (1 + (n1 != n2)) * deg / hybrid
                pd.initialize(kpt, n1, n2)
                pd.get_coarse(self.nt_G)
                pd.add_compensation_charges(self.nt_G, self.rhot_g)
                self.poisson_solve(self.vt_g, -self.rhot_g,
                                   charge=-float(n1 == n2), eps=1e-12,
                                   zero_initial_phi=True)
                if self.use_finegrid:
                    self.restrict(self.vt_g, self.vt_G)
                else:
                    assert self.vt_G is self.vt_g

                # Determine force contribution from exchange potential
                for nucleus in self.ghat_nuclei:
                    if self.use_finegrid:
                        ghat_L = nucleus.ghat_L
                    else:
                        ghat_L = nucleus.Ghat_L
                    
                    if nucleus.in_this_domain:
                        lmax = nucleus.setup.lmax
                        F_Lc = npy.zeros(((lmax + 1)**2, 3))
                        ghat_L.derivative(self.vt_g, F_Lc)
                        D_ii = npy.outer(nucleus.P_uni[u, n1],
                                         nucleus.P_uni[u, n2])
                        D_p = pack(D_ii, tolerance=1e30)
                        Q_L = npy.dot(D_p, nucleus.setup.Delta_pL)
                        F_ac[nucleus.a] -= (f1 * f2 * dc * npy.dot(Q_L, F_Lc))
                    else:
                        ghat_L.derivative(self.vt_g, None)

                # Add force contribution from the change in projectors
                for nucleus in self.ghat_nuclei:
                    v_L = npy.zeros((nucleus.setup.lmax + 1)**2)
                    if self.use_finegrid:
                        nucleus.ghat_L.integrate(self.vt_g, v_L)
                    else:
                        nucleus.Ghat_L.integrate(self.vt_G, v_L)
                    
                    if nucleus.in_this_domain:
                        v_ii = unpack(npy.dot(nucleus.setup.Delta_pL, v_L))

                        ni = nucleus.setup.ni
                        F_ic = npy.zeros((ni, 3))
                        nucleus.pt_i.derivative(psit1_G, F_ic)
                        F_ic.shape = (ni * 3,)
                        F_iic = npy.dot(v_ii, npy.outer(
                            nucleus.P_uni[u, n2], F_ic))

                        F_ic[:] = 0.0
                        F_ic.shape =(ni, 3)
                        nucleus.pt_i.derivative(psit2_G, F_ic)
                        F_ic.shape = (ni * 3,)
                        F_iic += npy.dot(v_ii, npy.outer(
                            nucleus.P_uni[u, n1], F_ic))

                        # F_iic *= 2.0
                        F_iic.shape = (ni, ni, 3)
                        for i in range(ni):
                            F_ac[nucleus.a] -= f1 * f2 * dc * real(F_iic[i, i])

                    else:
                        nucleus.pt_i.derivative(psit1_G, None)
                        nucleus.pt_i.derivative(psit2_G, None)
        return F_ac

    def adjust_residual(self, pR_G, dR_G, u, n):
        dR_G += self.vt_unG[u, n] * pR_G

    def rotate(self, u, U_nn):
        # Rotate EXX related stuff
        vt_nG = self.vt_unG[u]
        gemm(1.0, vt_nG.copy(), U_nn, 0.0, vt_nG)
        for nucleus in self.my_nuclei:
            v_ni = nucleus.vxx_uni[u]
            gemm(1.0, v_ni.copy(), U_nn, 0.0, v_ni)
            v_nii = nucleus.vxx_unii[u]
            gemm(1.0, v_nii.copy(), U_nn, 0.0, v_nii)


def atomic_exact_exchange(atom, type = 'all'):
    """Returns the exact exchange energy of the atom defined by the
       instantiated AllElectron object 'atom'
    """
    gaunt = make_gaunt(lmax=max(atom.l_j)) # Make gaunt coeff. list
    Nj = len(atom.n_j)                     # The total number of orbitals

    # determine relevant states for chosen type of exchange contribution
    if type == 'all':
        nstates = mstates = range(Nj)
    else:
        Njcore = core_states(atom.symbol) # The number of core orbitals
        if type == 'val-val':
            nstates = mstates = range(Njcore, Nj)
        elif type == 'core-core':
            nstates = mstates = range(Njcore)
        elif type == 'val-core':
            nstates = range(Njcore,Nj)
            mstates = range(Njcore)
        else:
            raise RuntimeError('Unknown type of exchange: ', type)

    # Arrays for storing the potential (times radius)
    vr = npy.zeros(atom.N)
    vrl = npy.zeros(atom.N)
    
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
                vr += vrl * npy.sum(G2.copy().ravel())

            # add to total exchange the contribution from current two states
            Exx += -.5 * f12 * npy.dot(vr, nrdr)

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
    vr = npy.zeros(N)
        
    # initialize X_ii matrix
    X_ii = npy.zeros((Nvi, Nvi))

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
                    nv = npy.dot(n1c, vr)
                    
                    # expansion coefficients
                    A_mm = X_ii[i1:i1 + 2 * lv1 + 1, i2:i2 + 2 * lv2 + 1]
                    for mc in range(2 * lc + 1):
                        for m in range(2 * l + 1):
                            G1c = gaunt[lv1**2:(lv1 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            G2c = gaunt[lv2**2:(lv2 + 1)**2,
                                        lc**2 + mc, l**2 + m]
                            A_mm += nv * npy.outer(G1c, G2c)
                i2 += 2 * lv2 + 1
            i1 += 2 * lv1 + 1

    # pack X_ii matrix
    X_p = pack2(X_ii, tolerance=1e-8)
    return X_p

def get_valence_core_exx(paw, hybrid=None):
    if hybrid is None:
        hybrid = paw.xcfunc.hybrid

    Exx_vc = 0.0
    for nucleus in paw.my_nuclei:
        D_p = npy.sum(nucleus.D_sp, axis=0)
        Exx_vc -= hybrid * npy.dot(D_p, nucleus.setup.X_p)

    return paw.gd.comm.sum(Exx_vc) * paw.Ha

def H_coulomb_val_core(paw, u=0):
    """Short description here.

    ::

                     core   *    *
             //       --   i(r) k(r') k(r) j (r')
       H   = || drdr' >   ----------------------
        ij   //       --          |r - r'|
                      k
                      
    """
    H_nn = npy.zeros((paw.nbands, paw.nbands), dtype=paw.dtype)
    for nucleus in paw.nuclei:
        X_ii = unpack(nucleus.setup.X_p)
        P_ni = nucleus.P_uni[u]
        H_nn += 2 * npy.dot(P_ni.conj(), npy.dot(X_ii, P_ni.T))

    return paw.gd.comm.sum(Exx_vc) * paw.Ha
