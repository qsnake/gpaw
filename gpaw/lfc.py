from math import pi

import numpy as np

from gpaw import debug, extra_parameters
from gpaw.spherical_harmonics import Y
import _gpaw

"""

===  =================================================
 M   Global localized function number.
 W   Global volume number.
 G   Global grid point number.
 g   Local (inside sphere) grid point number.
 i   Index into list of current spheres for current G.
===  =================================================

l
m
b
w

Global grid point number (*G*) for a 7*6 grid::

   -------------
  |5 . . . . . .|
  |4 . . . . . .|
  |3 9 . . . . .|
  |2 8 . . . . .|
  |1 7 . . . . .|
  |0 6 . . . . .|
   -------------

For this example *G* runs from 0 to 41.

Here is a sphere inside the box with grid points (*g*) numbered from 0
to 7::

   -------------
  |. . . . . . .|
  |. . . . 5 . .|
  |. . . 1 4 7 .|
  |. . . 0 3 6 .|
  |. . . . 2 . .|
  |. . . . . . .|
   -------------

~  _  ^  ~  ~
p  v  g  n  F  
 i     L  c  M

i
d  d  d  d  d
s     s
   s     s
"""

class Sphere:
    def __init__(self, spline_j):
        self.spline_j = spline_j
        self.spos_c = None
        self.rank = None
        self.ranks = None
        self.Mmax = None
        self.A_wgm = None
        self.G_wb = None
        self.M_w = None
        self.sdisp_wc = None
        self.normalized = False
        
    def set_position(self, spos_c, gd, cut):
        if self.spos_c is not None and not (self.spos_c - spos_c).any():
            return False

        self.A_wgm = []
        self.G_wb = []
        self.M_w = []
        self.sdisp_wc = []
        
        ng = 0
        M = 0
        for spline in self.spline_j:
            rcut = spline.get_cutoff()
            l = spline.get_angular_momentum_number()
            for beg_c, end_c, sdisp_c in gd.get_boxes(spos_c, rcut, cut):
                A_gm, G_b = self.spline_to_grid(spline, gd, beg_c, end_c,
                                                spos_c - sdisp_c)
                if len(G_b) > 0:
                    self.A_wgm.append(A_gm)
                    self.G_wb.append(G_b)
                    self.M_w.append(M)
                    self.sdisp_wc.append(sdisp_c)
                    ng += A_gm.shape[0]
                    assert A_gm.shape[0] > 0
            M += 2 * l + 1

        self.Mmax = M
        
        if ng > 0:
            self.rank = gd.get_rank_from_position(spos_c)
        else:
            self.rank = None
            self.ranks = None
            self.A_wgm = None
            self.G_wb = None
            self.M_w = None
            self.sdisp_wc = None
            
        self.spos_c = spos_c
        self.normalized = False
        return True

    def spline_to_grid(self, spline, gd, start_c, end_c, spos_c):
        dom = gd
        h_cv = dom.cell_cv / gd.N_c[:, None]
        pos_v = np.dot(spos_c, dom.cell_cv)
        return _gpaw.spline_to_grid(spline.spline, start_c, end_c, pos_v, h_cv,
                                    gd.n_c, gd.beg_c)

    def get_function_count(self):
        return sum([2 * spline.get_angular_momentum_number() + 1
                    for spline in self.spline_j])

    def normalize(self, integral, a, dv, comm):
        """Normalize localized functions."""
        if self.normalized:
            yield None
            yield None
            yield None
            return
        
        I_M = np.zeros(self.Mmax)
        
        nw = len(self.A_wgm) // len(self.spline_j)
        assert nw * len(self.spline_j) == len(self.A_wgm)

        for M, A_gm in zip(self.M_w, self.A_wgm):
            I_m = A_gm.sum(axis=0)
            I_M[M:M + len(I_m)] += I_m * dv

        requests = []
        if len(self.ranks) > 0:
            I_rM = np.empty((len(self.ranks), self.Mmax))
            for r, J_M in zip(self.ranks, I_rM):
                requests.append(comm.receive(J_M, r, a, False))
        if self.rank != comm.rank:
            requests.append(comm.send(I_M, self.rank, a, False))

        yield None

        for request in requests:
            comm.wait(request)
            
        requests = []
        if len(self.ranks) > 0:
            I_M += I_rM.sum(axis=0)
            for r in self.ranks:
                requests.append(comm.send(I_M, r, a, False))
        if self.rank != comm.rank:
            requests.append(comm.receive(I_M, self.rank, a, False))

        yield None

        for request in requests:
            comm.wait(request)
        
        w = 0
        for M, A_gm in zip(self.M_w, self.A_wgm):
            if M == 0 and integral > 1e-15:
                A_gm *= integral / I_M[0]
            else:
                A_gm -= I_M[M:M + A_gm.shape[1]] * self.A_wgm[w % nw]
            w +=1
        self.normalized = True
        yield None
        
# Quick hack: base class to share basic functionality across LFC classes
class BaseLFC:
    def dict(self, shape=(), derivative=False, zero=False):
        if isinstance(shape, int):
            shape = (shape,)
        if derivative:
            assert not zero
            c_axiv = {}
            for a in self.my_atom_indices:
                ni = self.get_function_count(a)
                c_axiv[a] = np.empty(shape + (ni, 3), self.dtype)
            return c_axiv
        else:
            c_axi = {}
            for a in self.my_atom_indices:
                ni = self.get_function_count(a)
                c_axi[a] = np.empty(shape + (ni,), self.dtype)
                if zero:
                    c_axi[a].fill(0.0)
            return c_axi

class NewLocalizedFunctionsCollection(BaseLFC):
    """New LocalizedFunctionsCollection

    Utilizes that localized functions can be stored on a spherical subset of
    the uniform grid, as opposed to LocalizedFunctionsCollection which is just
    a wrapper around the old localized_functions which use rectangular grids.

    Methods missing before LocalizedFunctionsCollection is obsolete:

    add1, add2, derivative
    """
    def __init__(self, gd, spline_aj, kpt_comm=None, cut=False, dtype=float,
                 integral=None, forces=None):
        self.gd = gd
        self.sphere_a = [Sphere(spline_j) for spline_j in spline_aj]
        self.cut = cut
        self.ibzk_qc = None
        self.gamma = True
        self.dtype = dtype

        # Global or local M-indices?
        self.use_global_indices = False
        
        if isinstance(integral, (float, int)):
            self.integral_a = np.empty(len(self.sphere_a))
            self.integral_a.fill(integral)
        else:
            self.integral_a = integral
        
    def set_k_points(self, ibzk_qc):
        self.ibzk_qc = ibzk_qc
        self.gamma = False
        self.dtype = complex
                
    def set_positions(self, spos_ac):
        movement = False
        for spos_c, sphere in zip(spos_ac, self.sphere_a):
            movement |= sphere.set_position(spos_c, self.gd, self.cut)

        if movement:
            self._update(spos_ac)
    
    def _update(self, spos_ac):
        nB = 0
        nW = 0
        self.my_atom_indices = []
        self.atom_indices = []
        M = 0
        self.M_a = []
        for a, sphere in enumerate(self.sphere_a):
            self.M_a.append(M)
            G_wb = sphere.G_wb
            if G_wb:
                nB += sum([len(G_b) for G_b in G_wb])
                nW += len(G_wb)
                self.atom_indices.append(a)
                if sphere.rank == self.gd.comm.rank:
                    self.my_atom_indices.append(a)

                if not self.use_global_indices:
                    M += sphere.Mmax

            if self.use_global_indices:
                M += sphere.Mmax

        self.Mmax = M

        natoms = len(spos_ac)
        if debug:
            # Holm-Nielsen check:
            assert (self.gd.comm.sum(float(sum(self.my_atom_indices))) ==
                    natoms * (natoms - 1) // 2)

        self.M_W = np.empty(nW, np.intc)
        self.G_B = np.empty(nB, np.intc)
        self.W_B = np.empty(nB, np.intc)
        self.A_Wgm = []
        sdisp_Wc = np.empty((nW, 3), int)
        self.pos_Wv = np.empty((nW, 3))        
        
        B1 = 0
        W = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            self.A_Wgm.extend(sphere.A_wgm)
            nw = len(sphere.M_w)
            self.M_W[W:W + nw] = self.M_a[a] + np.array(sphere.M_w)
            sdisp_Wc[W:W + nw] = sphere.sdisp_wc
            self.pos_Wv[W:W + nw] = np.dot(spos_ac[a] -
                                           np.array(sphere.sdisp_wc),
                                           self.gd.cell_cv)
            for G_b in sphere.G_wb:
                B2 = B1 + len(G_b)
                self.G_B[B1:B2] = G_b
                self.W_B[B1:B2:2] = W
                self.W_B[B1 + 1:B2 + 1:2] = -W - 1
                B1 = B2
                W += 1
        assert B1 == nB

        if self.gamma:
            self.phase_qW = np.empty((0, nW), complex)
        else:
            self.phase_qW = np.exp(2j * pi * np.inner(self.ibzk_qc, sdisp_Wc))

        indices = np.argsort(self.G_B, kind='mergesort')
        self.G_B = self.G_B[indices]
        self.W_B = self.W_B[indices]

        self.lfc = _gpaw.LFC(self.A_Wgm, self.M_W, self.G_B, self.W_B,
                             self.gd.dv, self.phase_qW)

        # Find out which ranks have a piece of the
        # localized functions:
        x_a = np.zeros(natoms, bool)
        x_a[self.atom_indices] = True
        x_a[self.my_atom_indices] = False
        x_ra = np.empty((self.gd.comm.size, natoms), bool)
        self.gd.comm.all_gather(x_a, x_ra)
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            if sphere.rank == self.gd.comm.rank:
                sphere.ranks = x_ra[:, a].nonzero()[0]
            else:
                sphere.ranks = []

        if self.integral_a is not None:
            iterators = []
            for a in self.atom_indices:
                iterator = self.sphere_a[a].normalize(self.integral_a[a], a,
                                                       self.gd.dv,
                                                       self.gd.comm)
                iterators.append(iterator)
            for i in range(3):
                for iterator in iterators:
                    iterator.next()
            
    def M_to_ai(self, src_xM, dst_axi):
        xshape = src_xM.shape[:-1]
        src_xM = src_xM.reshape(np.prod(xshape), self.Mmax)        
        for a in self.my_atom_indices:
            M1 = self.M_a[a]
            M2 = M1 + self.sphere_a[a].Mmax
            dst_axi[a] = src_xM[:, M1:M2].copy()

    def ai_to_M(self, src_axi, dst_xM):
        xshape = dst_xM.shape[:-1]
        dst_xM = dst_xM.reshape(np.prod(xshape), self.Mmax)
        for a in self.my_atom_indices:
            M1 = self.M_a[a]
            M2 = M1 + self.sphere_a[a].Mmax
            dst_xM[:, M1:M2] = src_axi[a]
    
    def add(self, a_xG, c_axi=1.0, q=-1):
        """Add localized functions to extended arrays.

        ::
        
                   --  a     a
          a (G) += >  c   Phi (G)
           x       --  xi    i
                   a,i
        """
        assert not self.use_global_indices
        
        if isinstance(c_axi, float):
            assert q == -1
            c_xi = np.array([c_axi])
            c_axi = dict([(a, c_xi) for a in self.my_atom_indices])

        comm = self.gd.comm
        dtype = a_xG.dtype
        xshape = a_xG.shape[:-3]
        requests = []
        M1 = 0
        b_axi = {}
        for a in self.atom_indices:
            c_xi = c_axi.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xi is None:
                c_xi = np.empty(xshape + (sphere.Mmax,), dtype)
                b_axi[a] = c_xi
                requests.append(comm.receive(c_xi, sphere.rank, a, False))
            else:
                for r in sphere.ranks:
                    requests.append(comm.send(c_xi, r, a, False))
            M1 = M2

        for request in requests:
            comm.wait(request)

        c_xM = np.empty(xshape + (self.Mmax,), dtype)
        M1 = 0
        for a in self.atom_indices:
            c_xi = c_axi.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xi is None:
                c_xi = b_axi[a]
            c_xM[..., M1:M2] = c_xi
            M1 = M2
            
        self.lfc.add(c_xM, a_xG, q)
    
    def add1(self, n_g, scale, I_a):
        """What should this do? XXX"""
        raise NotImplementedError

    def add2(self, n_g, D_asp, s, I_a):
        """Add atomic electron density to extended density array and integrate.

        ::

                   ---
                   \    a         a        a
           n(g) +=  )  D (s)   Phi (g)  Phi (g)
                   /    i1,i2    i1       i2
                   ---
                 a,i1,i2

        also at the same time::

               /    __
           a   |   \     a         a        a   
          I  = | dg >   D (s)   Phi (g)  Phi (g)
               |   /__   i1,i2     i1       i2
               /   i1,i2
        
        where s is the spin index, and D_ii' is the unpacked version of D_p
        """
        raise NotImplementedError

    def integrate(self, a_xG, c_axi, q=-1):
        """Calculate integrals of arrays times localized functions.

        ::
        
                   /             a
          c_axi =  | dG a (G) Phi (G)
                   /     x       i
        """
        assert not self.use_global_indices

        dtype = a_xG.dtype
        
        xshape = a_xG.shape[:-3]
        c_xM = np.zeros(xshape + (self.Mmax,), dtype)
        self.lfc.integrate(a_xG, c_xM, q)

        comm = self.gd.comm
        rank = comm.rank
        srequests = []
        rrequests = []
        c_arxi = {}
        b_axi = {}
        M1 = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if sphere.rank != rank:
                c_xi = c_xM[..., M1:M2].copy()
                b_axi[a] = c_xi
                srequests.append(comm.send(c_xi,
                                           sphere.rank, a, False))
            else:
                if len(sphere.ranks) > 0:
                    c_rxi = np.empty(sphere.ranks.shape + xshape + (M2 - M1,),
                                     dtype)
                    c_arxi[a] = c_rxi
                    for r, b_xi in zip(sphere.ranks, c_rxi):
                        rrequests.append(comm.receive(b_xi, r, a, False))
            M1 = M2

        for request in rrequests:
            comm.wait(request)

        M1 = 0
        for a in self.atom_indices:
            c_xi = c_axi.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xi is not None:
                if len(sphere.ranks) > 0:
                    c_xi[:] = c_xM[..., M1:M2] + c_arxi[a].sum(axis=0)
                else:
                    c_xi[:] = c_xM[..., M1:M2]
            M1 = M2

        for request in srequests:
            comm.wait(request)

    def derivative(self, a_xG, c_axiv, q=-1):
        """Calculate x-, y-, and z-derivatives of localized function integrals.

        ::
        
                    d   /             a
          c_axiv =  --  | dG a (G) Phi (G)
                    dv  /     x       i

        where v is either x, y, or z.
        """
        assert not self.use_global_indices

        dtype = a_xG.dtype
        
        xshape = a_xG.shape[:-3]
        c_xMv = np.zeros(xshape + (self.Mmax, 3), dtype)

        cspline_M = []
        for a in self.atom_indices:
            for spline in self.sphere_a[a].spline_j:
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)
        gd = self.gd
        h_cv = gd.cell_cv / gd.N_c
        self.lfc.derivative(a_xG, c_xMv, h_cv, gd.n_c, cspline_M,
                            gd.beg_c, self.pos_Wv, q)

        comm = self.gd.comm
        rank = comm.rank
        srequests = []
        rrequests = []
        c_arxiv = {}
        b_axiv = {}
        M1 = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if sphere.rank != rank:
                c_xiv = c_xMv[..., M1:M2, :].copy()
                b_axiv[a] = c_xiv
                srequests.append(comm.send(c_xiv,
                                           sphere.rank, a, False))
            else:
                if len(sphere.ranks) > 0:
                    c_rxiv = np.empty(sphere.ranks.shape + xshape +
                                      (M2 - M1, 3), dtype)
                    c_arxiv[a] = c_rxiv
                    for r, b_xiv in zip(sphere.ranks, c_rxiv):
                        rrequests.append(comm.receive(b_xiv, r, a, False))
            M1 = M2

        for request in rrequests:
            comm.wait(request)

        M1 = 0
        for a in self.atom_indices:
            c_xiv = c_axiv.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xiv is not None:
                if len(sphere.ranks) > 0:
                    c_xiv[:] = c_xMv[..., M1:M2, :] + c_arxiv[a].sum(axis=0)
                else:
                    c_xiv[:] = c_xMv[..., M1:M2, :]
            M1 = M2

        for request in srequests:
            comm.wait(request)

    def griditer(self):
        """Iterate over grid points."""
        self.g_W = np.zeros(len(self.M_W), np.intc)
        self.current_lfindices = []
        G1 = 0
        for W, G in zip(self.W_B, self.G_B):
            G2 = G

            yield G1, G2
            
            self.g_W[self.current_lfindices] += G2 - G1

            if W >= 0:
                self.current_lfindices.append(W)
            else:
                self.current_lfindices.remove(-1 - W)

            G1 = G2

    def get_function_count(self, a):
        return self.sphere_a[a].get_function_count()


class BasisFunctions(NewLocalizedFunctionsCollection):
    def __init__(self, gd, spline_aj, kpt_comm=None, cut=False, dtype=float,
                 integral=None, forces=None):
        NewLocalizedFunctionsCollection.__init__(self, gd, spline_aj,
                                                 kpt_comm, cut,
                                                 dtype, integral,
                                                 forces)
        self.use_global_indices = True
        
    def add_to_density(self, nt_sG, f_asi):
        """Add linear combination of squared localized functions to density.

        ::

          ~        --   a  /    a    \2
          n (r) += >   f   | Phi (r) |
            s      --   si \    i    /
                   a,i
        """
        nspins = len(nt_sG)
        f_sM = np.empty((nspins, self.Mmax))
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M1 = self.M_a[a]
            M2 = M1 + sphere.Mmax
            f_sM[:, M1:M2] = f_asi[a]

        for nt_G, f_M in zip(nt_sG, f_sM):
            self.lfc.construct_density1(f_M, nt_G)

    def construct_density(self, rho_MM, nt_G, q):
        """Calculate electron density from density matrix.

        rho_MM: ndarray
            Density matrix.
        nt_G: ndarray
            Pseudo electron density.

        ::
                  
          ~        --      *
          n(r) +=  >    Phi (r) rho     Phi (r)
                   --     M1       M1M2   M2
                  M1,M2 
        """
        self.lfc.construct_density(rho_MM, nt_G, q)

    def integrate2(self, a_xG, c_xM, q=-1):
        """Calculate integrals of arrays times localized functions.

        ::
        
                  /
          c_xM += | dG a (G) Phi (G)
                  /     x       M
        """
        xshape, Gshape = a_xG.shape[:-3], a_xG.shape[-3:]
        Nx = int(np.prod(xshape))
        a_xG = a_xG.reshape((Nx,) + Gshape)
        c_xM = c_xM.reshape(Nx, -1)
        for a_G, c_M in zip(a_xG, c_xM):
            self.lfc.integrate(a_G, c_M, q)

    def calculate_potential_matrix(self, vt_G, Vt_MM, q):
        """Calculate lower part of potential matrix.

        ::

                      /
            ~         |     *  _  ~ _        _   _
            V      =  |  Phi  (r) v(r) Phi  (r) dr    for  mu >= nu
             mu nu    |     mu            nu
                      /

        Overwrites the elements of the target matrix Vt_MM. """
        Vt_MM[:] = 0.0
        self.lfc.calculate_potential_matrix(vt_G, Vt_MM, q)

    def lcao_to_grid(self, C_nM, psit_nG, q):
        """Deploy basis functions onto grids according to coefficients.

        ::

                       ----
             ~   _     \                 _
            psi (r) +=  )    C     Phi  (r)
               n       /      n mu    mu
                       ----
                        mu
        """
        for C_M, psit_G in zip(C_nM, psit_nG):
            self.lfc.lcao_to_grid(C_M, psit_G, q)

    def calculate_potential_matrix_derivative(self, vt_G, DVt_MMc, q):
        """Calculate derivatives of potential matrix elements.

        ::

                     /     *  _
                    |   Phi  (r)
           ~c       |      mu    ~ _        _   _
          DV      = |   -------- v(r) Phi  (r) dr
            mu nu   |     dr             nu
                   /        c

        Results are added to DVt_MMc.
        """
        cspline_M = []
        for a, sphere in enumerate(self.sphere_a):
            for j, spline in enumerate(sphere.spline_j):
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)
        gd = self.gd
        h_cv = gd.cell_cv / gd.N_c
        self.lfc.calculate_potential_matrix_derivative(vt_G, DVt_MMc, h_cv,
                                                       gd.n_c, q,
                                                       np.array(cspline_M),
                                                       gd.beg_c, self.pos_Wv)

    # Python implementations:
    if 0:
        def add_to_density(self, nt_sG, f_sM):
            nspins = len(nt_sG)
            nt_sG = nt_sG.reshape((nspins, -1))
            for G1, G2 in self.griditer():
                for W in self.current_lfindices:
                    M = self.M_W[W]
                    A_gm = self.A_Wgm[W][self.g_W[W]:self.g_W[W] + G2 - G1]
                    nm = A_gm.shape[1]
                    nt_sG[0, G1:G2] += np.dot(A_gm**2, f_sM[0, M:M + nm])

        def construct_density(self, rho_MM, nt_G, k):
            nt_G = nt_G.ravel()

            for G1, G2 in self.griditer():
                for W1 in self.current_lfindices:
                    M1 = self.M_W[W1]
                    f1_gm = self.A_Wgm[W1][self.g_W[W1]:self.g_W[W1] + G2 - G1]
                    nm1 = f1_gm.shape[1]
                    for W2 in self.current_lfindices:
                        M2 = self.M_W[W2]
                        f2_gm = self.A_Wgm[W2][self.g_W[W2]:
                                               self.g_W[W2] + G2 - G1]
                        nm2 = f2_gm.shape[1]
                        rho_mm = rho_MM[M1:M1 + nm1, M2:M2 + nm2]
                        if self.ibzk_qc is not None:
                            rho_mm = (rho_mm *
                                      self.phase_qW[k, W1] *
                                      self.phase_qW[k, W2].conj()).real
                        nt_G[G1:G2] += (np.dot(f1_gm, rho_mm) * f2_gm).sum(1)

        def calculate_potential_matrix(self, vt_G, Vt_MM, k):
            vt_G = vt_G.ravel()
            Vt_MM[:] = 0.0
            dv = self.gd.dv

            for G1, G2 in self.griditer():
                for W1 in self.current_lfindices:
                    M1 = self.M_W[W1]
                    f1_gm = self.A_Wgm[W1][self.g_W[W1]:self.g_W[W1] + G2 - G1]
                    nm1 = f1_gm.shape[1]
                    for W2 in self.current_lfindices:
                        M2 = self.M_W[W2]
                        f2_gm = self.A_Wgm[W2][self.g_W[W2]:
                                               self.g_W[W2] + G2 - G1]
                        nm2 = f2_gm.shape[1]
                        Vt_mm = np.dot(f1_gm.T,
                                       vt_G[G1:G2, None] * f2_gm) * dv
                        if self.ibzk_qc is not None:
                            Vt_mm = (Vt_mm *
                                     self.phase_qW[k, W1].conj() *
                                     self.phase_qW[k, W2])
                        Vt_MM[M1:M1 + nm1, M2:M2 + nm2] += Vt_mm

        def lcao_to_grid(self, C_nM, psit_nG, k):
            for C_M, psit_G in zip(C_nM, psit_nG):
                self._lcao_band_to_grid(C_M, psit_G, k)

        def _lcao_band_to_grid(self, C_M, psit_G, k):
            psit_G = psit_G.ravel()
            for G1, G2 in self.griditer():
                for W in self.current_lfindices:
                    A_gm = self.A_Wgm[W][self.g_W[W]:self.g_W[W] + G2 - G1]
                    M1 = self.M_W[W]
                    M2 = M1 + A_gm.shape[1]
                    if self.ibzk_qc is None:
                        psit_G[G1:G2] += np.dot(A_gm, C_M[M1:M2])
                    else:
                        psit_G[G1:G2] += np.dot(A_gm,
                                                C_M[M1:M2] /
                                                self.phase_qW[k, W])


from gpaw.localized_functions import LocFuncs, LocFuncBroadcaster
from gpaw.mpi import run

class LocalizedFunctionsCollection(BaseLFC):
    def __init__(self, gd, spline_aj, kpt_comm=None,
                 cut=False, forces=False, dtype=float,
                 integral=None):
        self.gd = gd
        self.spline_aj = spline_aj
        self.cut = cut
        self.forces = forces
        self.dtype = dtype
        self.integral_a = integral

        self.spos_ac = None
        self.lfs_a = {}
        self.ibzk_qc = None
        self.gamma = True
        self.kpt_comm = kpt_comm

        self.my_atom_indices = None

    def set_k_points(self, ibzk_qc):
        self.ibzk_qc = ibzk_qc
        self.gamma = False

    def set_positions(self, spos_ac):
        if self.kpt_comm:
            lfbc = LocFuncBroadcaster(self.kpt_comm)
        else:
            lfbc = None

        for a, spline_j in enumerate(self.spline_aj):
            if self.spos_ac is None or (self.spos_ac[a] != spos_ac[a]).any():
                lfs = LocFuncs(spline_j, self.gd, spos_ac[a],
                               self.dtype, self.cut, self.forces, lfbc)
                if len(lfs.box_b) > 0:
                    if not self.gamma:
                        lfs.set_phase_factors(self.ibzk_qc)
                    self.lfs_a[a] = lfs
                elif a in self.lfs_a:
                    del self.lfs_a[a]

        if lfbc:
            lfbc.broadcast()

        rank = self.gd.comm.rank
        self.my_atom_indices = [a for a, lfs in self.lfs_a.items()
                                if lfs.root == rank]
        self.my_atom_indices.sort()
        self.atom_indices = [a for a, lfs in self.lfs_a.items()]
        self.atom_indices.sort()

        if debug:
            # Holm-Nielsen check:
            natoms = len(spos_ac)
            assert (self.gd.comm.sum(float(sum(self.my_atom_indices))) ==
                    natoms * (natoms - 1) // 2)

        if self.integral_a is not None:
            if isinstance(self.integral_a, (float, int)):
                integral = self.integral_a
                for a in self.atom_indices:
                    self.lfs_a[a].normalize(integral)
            else:
                for a in self.atom_indices:
                    lfs = self.lfs_a[a]
                    integral = self.integral_a[a]
                    if abs(integral) > 1e-15:
                        lfs.normalize(integral)
        self.spos_ac = spos_ac

    def get_dtype(self): # old LFC uses the dtype attribute for dicts
        return self.dtype

    def add(self, a_xG, c_axi=1.0, q=-1):
        if isinstance(c_axi, float):
            assert q == -1
            c_xi = np.array([c_axi])
            run([lfs.iadd(a_xG, c_xi) for lfs in self.lfs_a.values()])
        else:
            run([self.lfs_a[a].iadd(a_xG, c_axi.get(a), q, True)
                 for a in self.atom_indices])

    def integrate(self, a_xG, c_axi, q=-1):
        for c_xi in c_axi.values():
            c_xi.fill(0.0)
        run([self.lfs_a[a].iintegrate(a_xG, c_axi.get(a), q)
             for a in self.atom_indices])

    def derivative(self, a_xG, c_axiv, q=-1):
        for c_xiv in c_axiv.values():
            c_xiv.fill(0.0)
        run([self.lfs_a[a].iderivative(a_xG, c_axiv.get(a), q)
             for a in self.atom_indices])

    def add1(self, n_g, scale, I_a):
        scale_i = np.array([scale], float)
        for lfs in self.lfs_a.values():
            lfs.add(n_g, scale_i)
        for a, lfs in self.lfs_a.items():
            I_ic = np.zeros((1, 4))
            for box in lfs.box_b:
                box.norm(I_ic)
            I_a[a] += I_ic[0, 0] * scale

    def add2(self, n_g, D_asp, s, scale, I_a):
        for a, lfs in self.lfs_a.items():
            I_a[a] += lfs.add_density2(n_g, scale * D_asp[a][s])

    def get_function_count(self, a):
        return self.lfs_a[a].ni

if extra_parameters.get('usenewlfc'):
    LocalizedFunctionsCollection = NewLocalizedFunctionsCollection

def test():
    from gpaw.grid_descriptor import GridDescriptor

    ngpts = 40
    h = 1.0 / ngpts
    N_c = (ngpts, ngpts, ngpts)
    a = h * ngpts
    gd = GridDescriptor(N_c, (a, a, a))
    
    from gpaw.spline import Spline
    a = np.array([1, 0.9, 0.8, 0.0])
    s = Spline(0, 0.2, a)
    x = LocalizedFunctionsCollection(gd, [[s], [s]])
    x.set_positions([(0.5, 0.45, 0.5), (0.5, 0.55, 0.5)])
    n_G = gd.zeros()
    x.add(n_G)
    #xy.f(np.array(([(2.0,)])), n_G)
    import pylab as plt
    plt.contourf(n_G[20, :, :])
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    test()
