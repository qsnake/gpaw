# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import numpy as npy

from gpaw.utilities.blas import axpy, gemm, gemv, gemmdot
from gpaw import extra_parameters
from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import nablaYL

from gpaw.utilities.blas import gemm, gemv, axpy

# load points and weights for the angular integration
from gpaw.sphere import Y_nL, points, weights

from itertools import izip

"""
                           3
             __   dn       __   __    dY
   __  2    \       L  2  \    \        L  2
  (\/n) = (  ) Y  --- )  + ) (  )  n  --- )
            /__ L dr      /__  /__  L dr
                                        c
             L            c=1    L


        dY
          L
  A   = --- r
   Lc   dr
          c

"""
# A_ncL is defined as above, n is an expansion point index (50 Lebedev points).
A_ncL = npy.empty((len(points), 3, 25))
for A_cL, Y_L, R_c in zip(A_ncL, Y_nL, points):
    for L, Y in enumerate(Y_L):
        l = int(sqrt(L))
        A_cL[:, L] = nablaYL(L, R_c)  - l * R_c * Y

# Make A_Liy as a view into A_ncL, thus being contiguous in the order as
# A_ncL, i.e. A_Liy[:, :, y].T and A_Liy[:, c, y] are contiguous
# A_Liy used in the old XCCorrection class
A_Liy = A_ncL.T


class YLExpansion:
    def __init__(self, n_sLg, Y_nL, rgd):
        self.n_sLg = n_sLg
        self.Y_nL = Y_nL
        self.rgd = rgd
        self.nspins, self.Lmax, self.ng = n_sLg.shape

    def __iter__(self):
        raise NotImplementedError


class DensityExpansion(YLExpansion):
    def __iter__(self):
        """Expand the density on angular slices.

        ::

          n_g = \sum_L Y_L n_Lg
        """
        for Y_L in self.Y_nL:
            yield npy.dot(Y_L, self.n_sLg)

    def get_gradient_expansion(self):
        """Return GradiendExpansion object.

        This is a generator which will yield the gradient of this density.
        """
        # Calculate the radial derivatives of the density expansion dn/dr
        dndr_sLg = npy.zeros((self.nspins, self.Lmax, self.ng))
        for n_Lg, dndr_Lg in zip(self.n_sLg, dndr_sLg):
            for n_g, dndr_g in zip(n_Lg, dndr_Lg):
                self.rgd.derivative(n_g, dndr_g)

        # Create GradientExpansion object
        return GradientExpansion(self.n_sLg, self.Y_nL, dndr_sLg, self.rgd)


class GradientExpansion(YLExpansion):
    def __init__(self, n_sLg, Y_nL, dndr_sLg, rgd):
        YLExpansion.__init__(self, n_sLg, Y_nL, rgd)
        self.dndr_sLg = dndr_sLg

    def __iter__(self):
        """Iterate through all gradient slices.

        If the density is::
        
          n(rad, ang) = \sum_L n_L(rad) Y_L(ang).

        The gradient of the density n_g is::
        
          __             __                              __
          \/ n = \sum_L (\/_rad n_Lg) Y_L + \sum_L n_Lg (\/_ang Y_L).


        We denote the radial part by::

          a1_g := \sum_L Y_L dndr_Lg,

        and define r times the angular part by::

                            __
          a1_cg = r \sum_L (\/ Y_L) n_Lg = \sum_L A_cL n_Lg,
                           __
          where A_cL = r * \/_c Y_L.
        
        We also determine the square norm of the gradient of n::

                  __          __               __
          a2_g = |\/n_g|^2 = (\/_rad n_g)^2 + (\/_ang n_g)^2

               = a1_g^2 + \sum_c a1_cg[c]^2 / r^2

        """
        for A_cL, Y_L in zip(A_ncL, self.Y_nL):
            A_cL = A_cL[:, :self.Lmax].copy()

            # Radial gradient
            a1_sg = npy.dot(Y_L, self.dndr_sLg)

            # Angular gradient
            a1_scg = npy.zeros((self.nspins, 3, self.ng))
            for a1_cg, n_Lg in zip(a1_scg, self.n_sLg):
                gemmdot(A_cL, n_Lg, out=a1_cg)

            # Square norm of gradient of individual spin channels
            a2_sg = npy.sum(a1_scg**2, 1)        # \
            a2_sg[:, 1:] /= self.rgd.r_g[1:]**2  # | angular contribution
            a2_sg[:, 0] = a2_sg[:, 1]            # /
            axpy(1.0, a1_sg**2, a2_sg)           # radial contribution   

            # Square norm of gradient of total density
            if self.nspins == 1:
                a2_g = a2_sg[0]
            else:
                a2_g = npy.sum(a1_scg.sum(0)**2, 0) # \                     
                a2_g[1:] /= self.rgd.r_g[1:]**2     # | angular contribution
                a2_g[0] = a2_g[1]                   # /                     
                axpy(1.0, a1_sg.sum(0)**2, a2_g)    # radial contribution
            yield GradientSlice(a1_sg, a1_scg, a2_sg, a2_g, A_cL)


class GradientSlice:
    """Storage container for gradient information.

    GradientSlice stores all possible information related to
    spin paired/polarized radial gradient slice.
    """
    def __init__(self, a1_sg, a1_scg, a2_sg, a2_g, A_cL):
        self.a1_sg = a1_sg       # Radial gradient of spin density
        self.a1_scg = a1_scg     # Angular gradient of spin density
        self.a2_sg = a2_sg       # Square norm of gradient of spin density
        self.a2_g = a2_g         # Square norm of gradient of total density
        self.A_cL = A_cL         # r * dY_l / dr_c
        self.nspins, self.ng = a1_sg.shape

        # energy gradient wrt. square norm gradient
        if self.nspins == 1:
            self.deda2_sg = npy.zeros((1, self.ng))
            self.a1_g = self.a1_sg[0]
            self.a1_cg = self.a1_scg[0]
        else:
            self.deda2_sg = npy.zeros((3, self.ng))
            self.a1_g = self.a1_sg.sum(0)
            self.a1_cg = self.a1_scg.sum(0)

    def radial_gradient(self, spin=None):
        """The radial gradient of the density.

        Returns::
        
          a1_g = \sum_L Y_L dn_Lg / dr,

        where n is the spin density corresponding to ``spin`` (or the total
        density if spin is None).
        """
        if spin is None:
            return self.a1_g
        else:
            return self.a1_sg[spin]

    def angular_gradient(self, spin=None):
        """The angular gradient of the density.

        Returns::

          a1_cg = r \sum_L n_Lg dY_L / dr_c

        where n is the spin density corresponding to ``spin`` (or the total
        density if spin is None).
        """
        if spin is None:
            return self.a1_cg
        else:
            return self.a1_scg[spin]

    def square_norm_gradient(self, spin=None):
        """The square norm of the density gradient.

        Returns::

          a2_g = | nabla n |^2 = nabla n . nabla n

        where n is the spin density corresponding to ``spin`` (or the total
        density if spin is None).
        """
        if spin is None:
            return self.a2_g
        else:
            return self.a2_sg[spin]
        
    def energy_gradient(self, spin=0):
        """Derivative of energy with respect to the density gradient norm.

        For spin polarized systems, this is a length 3 vector.
        For Libxc xc-functionals, the three dimensions are::

          0: de / d | nabla na |^2
          1: de / d | nabla nb |^2
          2: de / d ( nabla na . nabla nb)

        where na / nb are the alpha / beta spin channels of the density.

        For GPAW's builtin GGA functionals, the three dimensions are::
          0: de / d | nabla na |^2
          1: de / d | nabla nb |^2
          2: de / d | nabla (na + nb) |^2
        """
        return self.deda2_sg[spin]

    def get_A_cL(self):
        """The gradient of sperical harmonics.

        ::

          A_cL = r * dY_L / dr_c
        """
        return self.A_cL


class Integrator:
    def __init__(self, H_sp, weights, Y_nL, B_pqL, rgd, libxc=True):
        self.H_sp = H_sp
        self.weights = weights
        self.Y_nL = Y_nL
        self.B_pqL = B_pqL
        self.rgd = rgd
        self.dv_g = rgd.dv_g
        self.nspins, self.np = H_sp.shape
        self.libxc = libxc

    def __iter__(self):
        for self.weight, self.Y_L in zip(self.weights, self.Y_nL):
            yield self

    def integrate_H_sp(self, coeff, v_sg, n_qg, grad=None):
        """Integrates given potential on given radial slice and adds the
        result to H_sp
        
        coeff:   Multiply the integration result with this constant before
                 adding to H_sp
        i_slice: Slice definition given by integrator iterator.
        v_sg:    The potential to integrate.
        n_qg:    All possible pairs of partial waves
        grad:    GradientSlice object with all possible details needed for
                 gradient
        """
        BY_pq = npy.dot(self.B_pqL, self.Y_L)
        v_sq = gemmdot(v_sg, n_qg, trans='t')

        # The LDA part
        for H_p, v_g in zip(self.H_sp, v_sg):
            dEdD_q = npy.dot(n_qg, v_g * self.dv_g)
            axpy(coeff * self.weight, npy.dot(BY_pq, dEdD_q), H_p)

        if grad is None:
            return

        # The GGA part
        A_cL = grad.get_A_cL()
        BA_pqc = gemmdot(self.B_pqL, A_cL, trans='t').reshape(self.np, -1)
        def energy_gradient(coeff, a1_g, a1_cg, deda2_g, dEdD_p):
            """Determine the derivative of the energy wrt the density matrix D.

            More explanations here...
            """
            # Add contribution from derivative of radial density times Y_L
            x_g = a1_g * deda2_g * self.dv_g
            self.rgd.derivative2(x_g, x_g)
            gemv(coeff, BY_pq, npy.dot(n_qg, x_g), 1.0, dEdD_p, 't')

            # Add contribution from radial density times gradient of Y_L
            x_cg = a1_cg * deda2_g * self.rgd.dr_g
            gemv(-4.0 * pi * coeff, BA_pqc,
                 gemmdot(n_qg, x_cg, trans='t').reshape(-1),
                 1.0, dEdD_p, 't')

        if self.nspins == 1:
            energy_gradient(-2.0 * coeff * self.weight,
                            grad.radial_gradient(),
                            grad.angular_gradient(),
                            grad.energy_gradient(),
                            self.H_sp[0])
        elif self.libxc:
            # Libxc GGA routine
            # Here grad.get_energy_gradient() means
            # de / d (  nabla na . nabla nb )
            for s, H_p in enumerate(self.H_sp):
                # Cross terms between spin channels
                s2 = (s + 1) % 2 # opposite spin index
                energy_gradient(-1.0 * coeff * self.weight,
                                grad.radial_gradient(s2),
                                grad.angular_gradient(s2),
                                grad.energy_gradient(2),
                                H_p)
                
                # Individual spin contributions
                energy_gradient(-2.0 * coeff * self.weight,
                                grad.radial_gradient(s),
                                grad.angular_gradient(s),
                                grad.energy_gradient(s),
                                H_p)
        else:
            # GPAW's own GGA routine
            # Here grad.get_energy_gradient() means
            # de / d |nabla (na + nb)|^2

            # This is common to both spin channels
            dEdD_p = npy.zeros((self.np))
            energy_gradient(-2.0 * coeff * self.weight,
                            grad.radial_gradient(),
                            grad.angular_gradient(),
                            grad.energy_gradient(2),
                            dEdD_p)
            self.H_sp += dEdD_p

            # Individual spin contributions
            for s, H_p in enumerate(self.H_sp):
                energy_gradient(-4.0 * coeff * self.weight,
                                grad.radial_gradient(s),
                                grad.angular_gradient(s),
                                grad.energy_gradient(s),
                                H_p)
    
    def integrate_E(self,E):
        return E * self.weight

    def integrate_e_g(self, e_g):
        return npy.dot(self.dv_g, e_g) * self.weight


class BaseXCCorrection:
    def __init__(self,
                 xc,    # radial exchange-correlation object
                 w_jg,  # all-lectron partial waves
                 wt_jg, # pseudo partial waves
                 nc_g,  # core density
                 nct_g, # smooth core density
                 rgd,   # radial grid edscriptor
                 jl,    # ?
                 lmax,  # maximal angular momentum to consider
                 Exc0,  # xc energy of reference atom
                 phicorehole_g, # ?
                 fcorehole,     # ?
                 nspins,        # Number os spins
                 tauc_g=None,   # kinetic core energy array
                 tauct_g=None): # pseudo kinetic core energy array

        self.nc_g = nc_g
        self.nct_g = nct_g
        self.xc = xc
        self.Exc0 = Exc0
        self.Lmax = (lmax + 1)**2
        self.lmax = lmax
        if lmax == 0:
            self.weights = [1.0]
            self.Y_yL = npy.array([[1.0 / sqrt(4.0 * pi)]])
        else:
            self.weights = weights
            self.Y_yL = Y_nL[:, :self.Lmax].copy()
        jlL = []
        for j, l in jl:
            for m in range(2 * l + 1):
                jlL.append((j, l, l**2 + m))

        self.jlL = jlL
        ng = len(nc_g)
        self.ng = ng
        self.ni = ni = len(jlL)
        self.nj = nj = len(jl)
        self.np = np = ni * (ni + 1) // 2
        self.nq = nq = nj * (nj + 1) // 2
        self.B_Lqp = npy.zeros((self.Lmax, nq, np))
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL:
            for j2, l2, L2 in jlL[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                self.B_Lqp[:, q, p] = gaunt[L1, L2, :self.Lmax]
                p += 1
            i1 += 1
        self.B_pqL = npy.transpose(self.B_Lqp).copy()
        self.dv_g = rgd.dv_g
        self.n_qg = npy.zeros((nq, ng))
        self.nt_qg = npy.zeros((nq, ng))
        q = 0
        for j1, l1 in jl:
            for j2, l2 in jl[j1:]:
                rl1l2 = rgd.r_g**(l1 + l2)
                self.n_qg[q] = rl1l2 * w_jg[j1] * w_jg[j2]
                self.nt_qg[q] = rl1l2 * wt_jg[j1] * wt_jg[j2]
                q += 1
        self.rgd = rgd

        self.nspins = nspins
        if nspins == 1:
            self.nc_g = nc_g
        else:
            if fcorehole == 0.0:
                self.nca_g = self.ncb_g = 0.5 * nc_g
            else:
                ncorehole_g = fcorehole * phicorehole_g**2 / (4 * pi)
                self.nca_g = 0.5 * (nc_g - ncorehole_g)
                self.ncb_g = 0.5 * (nc_g + ncorehole_g)

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a=None):
        if self.xc.get_functional().is_gllb():
            # The coefficients for GLLB-functional are evaluated elsewhere
            return self.xc.xcfunc.xc.calculate_energy_and_derivatives(
                D_sp, H_sp, a)
        if self.xc.get_functional().mgga:
            if self.xc.get_functional().uses_libxc:
                return self.MGGA_libxc(D_sp, H_sp)
            else:
                return self.MGGA(D_sp, H_sp)
        if self.xc.get_functional().gga:
            if self.xc.get_functional().uses_libxc:
                return self.GGA_libxc(D_sp, H_sp)
            else:
                return self.GGA(D_sp, H_sp)
        return self.LDA(D_sp, H_sp)

    def two_phi_integrals(self, D_sp):
        """Evaluate the integral in the augmentation sphere.

        ::

                      /
          I_{i1 i2} = | d r [ phi_i1(r) phi_i2(r) v_xc[n](r) -
                      /       tphi_i1(r) tphi_i2(r) v_xc[tn](r) ]
                      a

        The input D_sp is the density matrix in packed(pack) form
        The result is given in packed(pack2) form.
        """
        I_sp = npy.zeros(D_sp.shape)
        self.calculate_energy_and_derivatives(D_sp, I_sp)
        return I_sp

    def four_phi_integrals(self, D_sp, fxc):
        """Calculate four-phi integrals.

        The density is given by the density matrix ``D_sp`` in packed(pack)
        form, and the resulting rank-four tensor is also returned in
        packed format. ``fxc`` is a radial object???
        """

        ns, np = D_sp.shape

        assert ns == 1 and not self.xc.get_functional().gga

        dot = npy.dot

        D_p = D_sp[0]
        D_Lq = npy.dot(self.B_Lqp, D_p)

        # Expand all-electron density in spherical harmonics:
        n_qg = self.n_qg
        n_Lg = dot(D_Lq, n_qg)
        n_Lg[0] += self.nc_g * sqrt(4 * pi)

        # Expand pseudo electron density in spherical harmonics:
        nt_qg = self.nt_qg
        nt_Lg = dot(D_Lq, nt_qg)
        nt_Lg[0] += self.nct_g * sqrt(4 * pi)

        # Allocate array for result:
        J_pp = npy.zeros((np, np))

        # Loop over 50 points on the sphere surface:
        for w, Y_L in zip(self.weights, self.Y_yL):
            B_pq = npy.dot(self.B_pqL, Y_L)

            fxcdv = fxc(dot(Y_L, n_Lg)) * self.dv_g
            dn2_qq = npy.inner(n_qg * fxcdv, n_qg)

            fxctdv = fxc(dot(Y_L, nt_Lg)) * self.dv_g
            dn2_qq -= npy.inner(nt_qg * fxctdv, nt_qg)

            J_pp += w * npy.dot(B_pq, npy.inner(dn2_qq, B_pq))

        return J_pp

    def create_kinetic(self,jlL,jl,ny,np,phi_jg,tau_ypg):
        """Short title here.
        
        kinetic expression is::

                                             __         __ 
          tau_s = 1/2 Sum_{i1,i2} D(s,i1,i2) \/phi_i1 . \/phi_i2 +tauc_s

        here the orbital dependent part is calculated::

          __         __         
          \/phi_i1 . \/phi_i2 = 
                      __    __
                      \/YL1.\/YL2 phi_j1 phi_j2 +YL1 YL2 dphi_j1 dphi_j2
                                                         ------  ------
                                                           dr     dr
          __    __
          \/YL1.\/YL2 [y] = Sum_c A[L1,c,y] A[L2,c,y] / r**2 
          
        """
        ng = self.ng
        Lmax = self.Lmax
        nj = len(jl)
        ni = len(jlL)
        np = ni * (ni + 1) // 2
        dphidr_jg = npy.zeros(npy.shape(phi_jg))
        for j in range(nj):
            phi_g = phi_jg[j]
            self.rgd.derivative(phi_g, dphidr_jg[j])
        ##second term
        for y in range(ny):
            i1 = 0
            p = 0
            Y_L = self.Y_yL[y]
            for j1, l1, L1 in jlL:
                for j2, l2, L2 in jlL[i1:]:
                    c = Y_L[L1]*Y_L[L2]
                    temp = c * dphidr_jg[j1] *  dphidr_jg[j2]
                    tau_ypg[y,p,:] += temp
                    p += 1
                i1 +=1
        ##first term
        for y in range(ny):
            i1 = 0
            p = 0
            A_Li = A_Liy[:self.Lmax, :, y]
            A_Lxg = A_Li[:, 0]
            A_Lyg = A_Li[:, 1]
            A_Lzg = A_Li[:, 2]
            for j1, l1, L1 in jlL:
                for j2, l2, L2 in jlL[i1:]:
                    temp = (A_Lxg[L1] * A_Lxg[L2] + A_Lyg[L1] * A_Lyg[L2]
                            + A_Lzg[L1] * A_Lzg[L2])
                    temp *=  phi_jg[j1] * phi_jg[j2] 
                    temp[1:] /= self.rgd.r_g[1:]**2                       
                    temp[0] = temp[1]
                    tau_ypg[y, p, :] += temp
                    p += 1
                i1 +=1
        tau_ypg *= 0.5
                    
        return 
        
    def set_nspins(self, nspins):
        """change number of spins"""
        if nspins != self.nspins:
            self.nspins = nspins
            if nspins == 1:
                self.nc_g = self.nca_g + self.ncb_g
            else:
                self.nca_g = self.ncb_g = 0.5 * self.nc_g
                
    def initialize_kinetic(self, data):
        r_g = self.rgd.r_g
        ny = len(points)
        ng = self.ng
        l_j = data.l_j
        nj = len(l_j)
        jl =  [(j, l_j[j]) for j in range(nj)]
        jlL = []
        for j, l in jl:
            for m in range(2 * l + 1):
                jlL.append((j, l, l**2 + m))
        ni = len(jlL)
        np = ni * (ni + 1) // 2
        self.tau_ypg = npy.zeros((ny, np, ng))
        self.taut_ypg = npy.zeros((ny, np, ng))
        phi_jg = data.phi_jg
        phit_jg = data.phit_jg
        phi_jg = npy.array([phi_g[:ng].copy() for phi_g in phi_jg])
        phit_jg = npy.array([phit_g[:ng].copy() for phit_g in phit_jg])
        self.create_kinetic(jlL,jl,ny, np,phit_jg, self.taut_ypg)
        self.create_kinetic(jlL,jl,ny, np,phi_jg, self.tau_ypg)            
        tauc_g = data.tauc_g
        tauct_g = data.tauct_g
        self.tauc_g = npy.array(tauc_g[:ng].copy())
        self.tauct_g = npy.array(tauct_g[:ng].copy())


class NewXCCorrection(BaseXCCorrection):
    def expand_density(self, D_sp, core=True):
        D_sLq = gemmdot(D_sp, self.B_Lqp, trans='t')
        n_sLg = npy.dot(D_sLq, self.n_qg)
        if core:
            if self.nspins == 1:
                axpy(sqrt(4 * pi), self.nc_g, n_sLg[0, 0])
            else:
                axpy(sqrt(4 * pi), self.nca_g, n_sLg[0, 0])
                axpy(sqrt(4 * pi), self.ncb_g, n_sLg[1, 0])
        return DensityExpansion(n_sLg, self.Y_yL, self.rgd)
    
    def expand_pseudo_density(self, D_sp, core=True):
        # TODO: when calling both expand pseudo_density
        # and expand_density the line below is redunant XXX
        D_sLq = gemmdot(D_sp, self.B_Lqp, trans='t')
        n_sLg = npy.dot(D_sLq, self.nt_qg)
        if core:
            n_sLg[:, 0] += sqrt(4 * pi) / self.nspins * self.nct_g
        return DensityExpansion(n_sLg, self.Y_yL, self.rgd)

    def get_integrator(self, H_sp):
        libxc = self.xc.get_functional().uses_libxc
        return Integrator(H_sp, self.weights, self.Y_yL, self.B_pqL,
                          self.rgd, libxc)

    def calculate_potential_slice(self, e_g, n_sg, vxc_sg, grad=None):
        xcfunc = self.xc.get_functional()
        vxc_sg[:] = 0.0
        if grad is None:
            if self.nspins == 1:
                xcfunc.calculate_spinpaired(e_g, n_sg[0], vxc_sg[0])
            else:
                xcfunc.calculate_spinpolarized(e_g,
                                               n_sg[0], vxc_sg[0],
                                               n_sg[1], vxc_sg[1])
        else:
            if self.nspins == 1:
                xcfunc.calculate_spinpaired(e_g, n_sg[0], vxc_sg[0],
                                            grad.square_norm_gradient(),
                                            grad.energy_gradient())
            else:
                xcfunc.calculate_spinpolarized(e_g,
                                               n_sg[0], vxc_sg[0],
                                               n_sg[1], vxc_sg[1],
                                               grad.square_norm_gradient(),
                                               grad.square_norm_gradient(0),
                                               grad.square_norm_gradient(1),
                                               grad.energy_gradient(2),
                                               grad.energy_gradient(0),
                                               grad.energy_gradient(1))

    def LDA(self, D_sp, H_sp):
        H_sp[:] = 0.0
        vxc_sg = npy.zeros((self.nspins, self.ng))
        e_g = npy.zeros((self.ng,))
        Etot = 0.0
        
        for n_sg, nt_sg, integrator in izip(self.expand_density(D_sp),
                                            self.expand_pseudo_density(D_sp),
                                            self.get_integrator(H_sp)):
            # ae-density
            self.calculate_potential_slice(e_g, n_sg, vxc_sg)
            Etot += integrator.integrate_e_g(e_g)
            integrator.integrate_H_sp(1.0, vxc_sg, self.n_qg)

            # pseudo-density
            self.calculate_potential_slice(e_g, nt_sg, vxc_sg)
            Etot -= integrator.integrate_e_g(e_g)
            integrator.integrate_H_sp(-1.0, vxc_sg, self.nt_qg)
        return Etot - self.Exc0

    def GGA(self, D_sp, H_sp):
        H_sp[:] = 0.0
        vxc_sg = npy.zeros((self.nspins, self.ng))
        e_g = npy.zeros((self.ng,))
        Etot = 0

        density_iter = self.expand_density(D_sp)
        pseudo_density_iter = self.expand_pseudo_density(D_sp)
        for n_sg, nt_sg, grad, gradt, integrator in izip(
            density_iter,
            pseudo_density_iter,
            density_iter.get_gradient_expansion(),
            pseudo_density_iter.get_gradient_expansion(),
            self.get_integrator(H_sp)):

            # ae-density
            self.calculate_potential_slice(e_g, n_sg, vxc_sg, grad)
            Etot += integrator.integrate_e_g(e_g)
            integrator.integrate_H_sp(1.0, vxc_sg, self.n_qg, grad=grad)

            # pseudo density
            self.calculate_potential_slice(e_g, nt_sg, vxc_sg, gradt)
            Etot -= integrator.integrate_e_g(e_g)
            integrator.integrate_H_sp(-1.0, vxc_sg, self.nt_qg, grad=gradt)
        return Etot - self.Exc0

    GGA_libxc = GGA
        
    def MGGA(self, D_sp, H_sp):
        raise NotImplementedError

    MGGA_libxc = MGGA


class XCCorrection(BaseXCCorrection):
    def quickdot(self, A_mn, x_n, y_m=None, trans='t'):
        if y_m is None and trans == 't':
            y_m = npy.zeros(A_mn.shape[:-1], dtype=A_mn.dtype)
        elif y_m is None and trans == 'n':
            y_m = npy.zeros(A_mn.shape[1:], dtype=A_mn.dtype)
        gemv(1.0, A_mn, x_n, 0.0, y_m, trans)
        return y_m
    
    def quickdotmm(self, A_mk, B_kn, C_mn=None):
        if C_mn is None:
            C_mn = npy.zeros((A_mk.shape[0],B_kn.shape[1]), A_mk.dtype)
        gemm(1.0, B_kn, A_mk, 0.0, C_mn, 'n')
        return C_mn
    
    def LDA(self, D_sp, H_sp):
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = self.quickdot(self.B_Lqp, D_p)
            n_Lg = self.quickdotmm(D_Lq, self.n_qg)
            axpy(sqrt(4 * pi), self.nc_g, n_Lg[0])
            nt_Lg = self.quickdotmm(D_Lq, self.nt_qg)
            axpy(sqrt(4 * pi), self.nct_g, nt_Lg[0])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            for w, Y_L in zip(self.weights, self.Y_yL):
                n_g = self.quickdot(n_Lg, Y_L, None, 'n')
                vxc_g = npy.zeros(self.ng)
                E += self.xc.get_energy_and_potential(n_g, vxc_g) * w
                dEdD_q = self.quickdot(self.n_qg, vxc_g * self.dv_g)
                nt_g = self.quickdot(nt_Lg, Y_L, None, 'n')
                vxct_g = npy.zeros(self.ng)
                E -= self.xc.get_energy_and_potential(nt_g, vxct_g) * w
                axpy(-1.0, self.quickdot(self.nt_qg, vxct_g * self.dv_g), dEdD_q)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, Y_L), dEdD_q), dEdD_p)
        else: 
            Da_p = D_sp[0]
            Da_Lq = self.quickdot(self.B_Lqp, Da_p)
            na_Lg = self.quickdotmm(Da_Lq, self.n_qg)
            axpy(sqrt(4 * pi), self.nca_g, na_Lg[0])
            nta_Lg = self.quickdotmm(Da_Lq, self.nt_qg)
            axpy(0.5*sqrt(4 * pi), self.nct_g, nta_Lg[0])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            Db_p = D_sp[1]
            Db_Lq = self.quickdot(self.B_Lqp, Db_p)
            nb_Lg = self.quickdotmm(Db_Lq, self.n_qg)
            axpy(sqrt(4 * pi), self.ncb_g, nb_Lg[0])
            ntb_Lg = self.quickdotmm(Db_Lq, self.nt_qg)
            axpy(0.5*sqrt(4 * pi), self.nct_g, ntb_Lg[0])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            for w, Y_L in zip(self.weights, self.Y_yL):
                na_g = self.quickdot(na_Lg, Y_L, None, 'n')
                vxca_g = npy.zeros(self.ng)
                nb_g = self.quickdot(nb_Lg, Y_L, None, 'n')
                vxcb_g = npy.zeros(self.ng)
                E += self.xc.get_energy_and_potential(na_g, vxca_g,
                                                      nb_g, vxcb_g) * w
                dEdDa_q = self.quickdot(self.n_qg, vxca_g * self.dv_g)
                dEdDb_q = self.quickdot(self.n_qg, vxcb_g * self.dv_g)
                nta_g = self.quickdot(nta_Lg, Y_L, None, 'n')
                vxcta_g = npy.zeros(self.ng)
                ntb_g = self.quickdot(ntb_Lg, Y_L, None, 'n')
                vxctb_g = npy.zeros(self.ng)
                E -= self.xc.get_energy_and_potential(nta_g, vxcta_g,
                                                      ntb_g, vxctb_g) * w
                axpy(-1.0, self.quickdot(self.nt_qg, vxcta_g * self.dv_g), dEdDa_q)
                axpy(-1.0, self.quickdot(self.nt_qg, vxctb_g * self.dv_g), dEdDb_q)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, Y_L), dEdDa_q), dEdDa_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, Y_L), dEdDb_q), dEdDb_p)
        return E - self.Exc0
        
    def GGA(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        xcfunc = self.xc.get_functional()
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = self.quickdot(self.B_Lqp, D_p)
            n_Lg = self.quickdotmm(D_Lq, self.n_qg)
            axpy(sqrt(4 * pi), self.nc_g, n_Lg[0])
            nt_Lg = self.quickdotmm(D_Lq, self.nt_qg)
            axpy(sqrt(4 * pi), self.nct_g, nt_Lg[0])
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                n_g = self.quickdot(n_Lg, Y_L, None, 'n')
                a1x_g = self.quickdot(n_Lg, A_Li[:, 0], None, 'n')
                a1y_g = self.quickdot(n_Lg, A_Li[:, 1], None, 'n')
                a1z_g = self.quickdot(n_Lg, A_Li[:, 2], None, 'n')
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = self.quickdot(dndr_Lg, Y_L, None, 'n')
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.n_qg, x_g)), dEdD_p)
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.n_qg, x_g * a1x_g)), dEdD_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.n_qg, x_g * a1y_g)), dEdD_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.n_qg, x_g * a1z_g)), dEdD_p)

                n_g = self.quickdot(nt_Lg, Y_L, None, 'n')
                a1x_g = self.quickdot(nt_Lg, A_Li[:, 0], None, 'n')
                a1y_g = self.quickdot(nt_Lg, A_Li[:, 1], None, 'n')
                a1z_g = self.quickdot(nt_Lg, A_Li[:, 2], None, 'n')
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = self.quickdot(dntdr_Lg, Y_L, None, 'n')
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.nt_qg, x_g)), dEdD_p)
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.nt_qg, x_g * a1x_g)), dEdD_p)
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.nt_qg, x_g * a1y_g)), dEdD_p)
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.nt_qg, x_g * a1z_g)), dEdD_p)

                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = self.quickdot(self.B_Lqp, Da_p)
            na_Lg = self.quickdotmm(Da_Lq, self.n_qg)
            axpy(sqrt(4 * pi), self.nca_g, na_Lg[0])
            nat_Lg = self.quickdotmm(Da_Lq, self.nt_qg)
            axpy(0.5*sqrt(4 * pi), self.nct_g, nat_Lg[0])
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0

            Db_p = D_sp[1]
            Db_Lq = self.quickdot(self.B_Lqp, Db_p)
            nb_Lg = self.quickdotmm(Db_Lq, self.n_qg)
            axpy(sqrt(4 * pi), self.ncb_g, nb_Lg[0])
            nbt_Lg = self.quickdotmm(Db_Lq, self.nt_qg)
            axpy(0.5*sqrt(4 * pi), self.nct_g, nbt_Lg[0])
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                na_g = self.quickdot(na_Lg, Y_L, None, 'n')
                aa1x_g = self.quickdot(na_Lg, A_Li[:, 0], None, 'n')
                aa1y_g = self.quickdot(na_Lg, A_Li[:, 1], None, 'n')
                aa1z_g = self.quickdot(na_Lg, A_Li[:, 2], None, 'n')
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = self.quickdot(dnadr_Lg, Y_L, None, 'n')
                aa2_g += aa1_g**2

                nb_g = self.quickdot(nb_Lg, Y_L, None, 'n')
                ab1x_g = self.quickdot(nb_Lg, A_Li[:, 0], None, 'n')
                ab1y_g = self.quickdot(nb_Lg, A_Li[:, 1], None, 'n')
                ab1z_g = self.quickdot(nb_Lg, A_Li[:, 2], None, 'n')
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = self.quickdot(dnbdr_Lg, Y_L, None, 'n')
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.n_qg, x_g * (aa1x_g + ab1x_g))), dEdD_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.n_qg, x_g * (aa1y_g + ab1y_g))), dEdD_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.n_qg, x_g * (aa1z_g + ab1z_g))), dEdD_p)
                dEdDa_p += dEdD_p
                dEdDb_p += dEdD_p

                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.n_qg, x_g)), dEdDa_p)
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.n_qg, x_g * aa1x_g)), dEdDa_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.n_qg, x_g * aa1y_g)), dEdDa_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.n_qg, x_g * aa1z_g)), dEdDa_p)

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.n_qg, x_g)), dEdDb_p)
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.n_qg, x_g * ab1x_g)), dEdDb_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.n_qg, x_g * ab1y_g)), dEdDb_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.n_qg, x_g * ab1z_g)), dEdDb_p)
                
                na_g = self.quickdot(nat_Lg, Y_L, None, 'n')
                aa1x_g = self.quickdot(nat_Lg, A_Li[:, 0], None, 'n')
                aa1y_g = self.quickdot(nat_Lg, A_Li[:, 1], None, 'n')
                aa1z_g = self.quickdot(nat_Lg, A_Li[:, 2], None, 'n')
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = self.quickdot(dnatdr_Lg, Y_L, None, 'n')
                aa2_g += aa1_g**2

                nb_g = self.quickdot(nbt_Lg, Y_L, None, 'n')
                ab1x_g = self.quickdot(nbt_Lg, A_Li[:, 0], None, 'n')
                ab1y_g = self.quickdot(nbt_Lg, A_Li[:, 1], None, 'n')
                ab1z_g = self.quickdot(nbt_Lg, A_Li[:, 2], None, 'n')
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = self.quickdot(dnbtdr_Lg, Y_L, None, 'n')
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                dEdD_p = w * self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.nt_qg, x_g * (aa1x_g + ab1x_g))), dEdD_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.nt_qg, x_g * (aa1y_g + ab1y_g))), dEdD_p)
                axpy(w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.nt_qg, x_g * (aa1z_g + ab1z_g))), dEdD_p)
                dEdDa_p -= dEdD_p
                dEdDb_p -= dEdD_p

                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.nt_qg, x_g)), dEdDa_p)
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.nt_qg, x_g * aa1x_g)), dEdDa_p)
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.nt_qg, x_g * aa1y_g)), dEdDa_p)
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.nt_qg, x_g * aa1z_g)), dEdDa_p)

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, Y_L), self.quickdot(self.nt_qg, x_g)), dEdDb_p)
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 0]), self.quickdot(self.nt_qg, x_g * ab1x_g)), dEdDb_p)
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 1]), self.quickdot(self.nt_qg, x_g * ab1y_g)), dEdDb_p)
                axpy(-w, self.quickdot(self.quickdot(self.B_pqL, A_Li[:, 2]), self.quickdot(self.nt_qg, x_g * ab1z_g)), dEdDb_p)

                y += 1
        return E - self.Exc0
    
    def GGA_libxc(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        xcfunc = self.xc.get_functional()
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = npy.dot(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                n_g = npy.dot(Y_L, n_Lg)
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * a1z_g))

                n_g = npy.dot(Y_L, nt_Lg)
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * a1z_g))

                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = npy.dot(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nat_Lg = npy.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0

            Db_p = D_sp[1]
            Db_Lq = npy.dot(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            nbt_Lg = npy.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]

                na_g = npy.dot(Y_L, na_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], na_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], na_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nb_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -deda2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * aa1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * aa1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * aa1z_g))
                dEdDb_p += dEdD_p

                x_g = -deda2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * ab1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * ab1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * ab1z_g))
                dEdDa_p += dEdD_p

                x_g = -2.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * aa1z_g))

                x_g = -2.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL, A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * ab1z_g))

                na_g = npy.dot(Y_L, nat_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nbt_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -deda2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * aa1z_g))
                dEdDb_p -= dEdD_p

                x_g = -deda2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * ab1z_g))
                dEdDa_p -= dEdD_p

                x_g = -2.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * aa1z_g))
                
                x_g = -2.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * ab1z_g))

                y += 1

        return E - self.Exc0

    def MGGA(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        E = 0.0
        xcfunc = self.xc.get_functional()
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = npy.dot(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0

            for w, Y_L in zip(self.weights, self.Y_yL):
                ## Calculate pseudo and all electron kinetic energy 
                ## from orbitals
                taut_pg = self.taut_ypg[y]
                taut_g = npy.dot(D_p,taut_pg)
                tau_pg = self.tau_ypg[y]
                tau_g = npy.dot(D_p,tau_pg)
                tau_g += self.tauc_g / sqrt(4. * pi)
                taut_g += self.tauct_g / sqrt(4. * pi)
                A_Li = A_Liy[:self.Lmax, :, y]
                
                n_g = npy.dot(Y_L, n_Lg)
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedtaua_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g,
                                            tau_g,dedtaua_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * a1z_g))
                dedtaua_g *= self.dv_g
                dEdD_p += w * npy.dot(tau_pg, dedtaua_g)

                n_g = npy.dot(Y_L, nt_Lg)
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g,
                                            taut_g,dedtaua_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL,
                                            A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL,
                                            A_Li[:, 1]),
                                     npy.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL,
                                            A_Li[:, 2]),
                                     npy.dot(self.nt_qg, x_g * a1z_g))
                dedtaua_g *= self.dv_g
                dEdD_p -= w * npy.dot(taut_pg,dedtaua_g)
                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = npy.dot(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nat_Lg = npy.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            
            Db_p = D_sp[1]
            Db_Lq = npy.dot(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            nbt_Lg = npy.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                taut_pg = self.taut_ypg[y]
                tauat_g = npy.dot(Da_p,taut_pg)
                taubt_g = npy.dot(Db_p,taut_pg)
                tau_pg = self.tau_ypg[y]
                taua_g = npy.dot(Da_p,tau_pg)
                taub_g = npy.dot(Db_p,tau_pg)
                taua_g += self.tauc_g * 0.5 / sqrt(4. * pi)
                taub_g += self.tauc_g * 0.5 / sqrt(4. * pi)
                tauat_g += self.tauct_g * 0.5 / sqrt(4. * pi)
                taubt_g += self.tauct_g * 0.5 / sqrt(4. * pi)
                A_Li = A_Liy[:self.Lmax, :, y]                

                na_g = npy.dot(Y_L, na_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], na_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], na_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nb_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2
                 
                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                         (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                dedtaua_g = npy.zeros(self.ng)
                dedtaub_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g,
                                               taua_g,taub_g,dedtaua_g,
                                               dedtaub_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp
                 
                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * (aa1x_g +
                                                                ab1x_g)))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * (aa1y_g +
                                                                ab1y_g)))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * (aa1z_g +
                                                                ab1z_g)))
                dEdDa_p += dEdD_p
                dEdDb_p += dEdD_p
                 
                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * aa1z_g))

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * ab1z_g))
                dedtaua_g *= self.dv_g
                dedtaub_g *= self.dv_g
                dEdDa_p += w * npy.dot(tau_pg,dedtaua_g)
                dEdDb_p += w * npy.dot(tau_pg,dedtaub_g)

                na_g = npy.dot(Y_L, nat_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nbt_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2
                 
                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g,
                                               tauat_g,taubt_g,dedtaua_g,
                                               dedtaub_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                 
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * (aa1x_g +
                                                                 ab1x_g)))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * (aa1y_g +
                                                                 ab1y_g)))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * (aa1z_g +
                                                                 ab1z_g)))
                dEdDa_p -= dEdD_p
                dEdDb_p -= dEdD_p
                
                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * aa1z_g))
                
                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * ab1z_g))
                dedtaua_g *= self.dv_g
                dedtaub_g *= self.dv_g
                dEdDa_p -= w * npy.dot(taut_pg,dedtaua_g)
                dEdDb_p -= w * npy.dot(taut_pg,dedtaub_g)
                y += 1

#        return 0.0
        return E - self.Exc0

    def MGGA_libxc(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        xcfunc = self.xc.get_functional()
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = npy.dot(self.B_Lqp, D_p)
            n_Lg = npy.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = npy.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = npy.zeros((self.Lmax, self.ng))
            dntdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0

            for w, Y_L in zip(self.weights, self.Y_yL):
                ## Calculate pseudo and all electron kinetic energy 
                ##from orbitals
                taut_pg = self.taut_ypg[y]
                taut_g = npy.dot(D_p,taut_pg)
                tau_pg = self.tau_ypg[y]
                tau_g = npy.dot(D_p,tau_pg)
                tau_g += self.tauc_g / sqrt(4. * pi)
                taut_g += self.tauct_g / sqrt(4. * pi)
                A_Li = A_Liy[:self.Lmax, :, y]
                
                n_g = npy.dot(Y_L, n_Lg)
                a1x_g = npy.dot(A_Li[:, 0], n_Lg)
                a1y_g = npy.dot(A_Li[:, 1], n_Lg)
                a1z_g = npy.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedtaua_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g,
                                            tau_g, dedtaua_g)
                E += w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * a1z_g))
                dedtaua_g *= self.dv_g
                dEdD_p += w * npy.dot(tau_pg,dedtaua_g)

                n_g = npy.dot(Y_L, nt_Lg)
                a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
                a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
                a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = npy.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g,
                                            taut_g,dedtaua_g)
                E -= w * npy.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * a1z_g))

                dedtaua_g *= self.dv_g
                dEdD_p -= w * npy.dot(taut_pg,dedtaua_g)
                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = npy.dot(self.B_Lqp, Da_p)
            na_Lg = npy.dot(Da_Lq, self.n_qg)
            na_Lg[0] += self.nca_g * sqrt(4 * pi)
            nat_Lg = npy.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = npy.zeros((self.Lmax, self.ng))
            dnatdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            
            Db_p = D_sp[1]
            Db_Lq = npy.dot(self.B_Lqp, Db_p)
            nb_Lg = npy.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += self.ncb_g * sqrt(4 * pi)
            nbt_Lg = npy.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = npy.zeros((self.Lmax, self.ng))
            dnbtdr_Lg = npy.zeros((self.Lmax, self.ng))
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                taut_pg = self.taut_ypg[y]
                tauat_g = npy.dot(Da_p,taut_pg)
                taubt_g = npy.dot(Db_p,taut_pg)
                tau_pg = self.tau_ypg[y]
                taua_g = npy.dot(Da_p,tau_pg)
                taub_g = npy.dot(Db_p,tau_pg)
                taua_g += self.tauc_g * 0.5 / sqrt(4. * pi)
                taub_g += self.tauc_g * 0.5 / sqrt(4. * pi)
                tauat_g += self.tauct_g * 0.5 / sqrt(4. * pi)
                taubt_g += self.tauct_g * 0.5 / sqrt(4. * pi)
                A_Li = A_Liy[:self.Lmax, :, y]                

                na_g = npy.dot(Y_L, na_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], na_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], na_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nb_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2
                 
                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                dedtaua_g = npy.zeros(self.ng)
                dedtaub_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g,
                                               taua_g, taub_g, dedtaua_g,
                                               dedtaub_g)
                E += w * npy.dot(e_g, self.dv_g)

                x_g = -deda2_g * self.dv_g * aa1_g 
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp
                 
                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L), #is there a +=??
                                     npy.dot(self.n_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * aa1x_g ))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * aa1y_g ))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * aa1z_g ))
                dEdDb_p += dEdD_p
                
                x_g = -deda2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p  = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.n_qg, x_g * ab1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.n_qg, x_g * ab1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.n_qg, x_g * ab1z_g))
                dEdDa_p += dEdD_p

                x_g = -2.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * aa1z_g))

                x_g = -2.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.n_qg, x_g * ab1z_g))
                dedtaua_g *= self.dv_g
                dedtaub_g *= self.dv_g
                dEdDa_p += w * npy.dot(tau_pg,dedtaua_g)
                dEdDb_p += w * npy.dot(tau_pg,dedtaub_g)

                na_g = npy.dot(Y_L, nat_Lg)
                aa1x_g = npy.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = npy.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = npy.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = npy.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = npy.dot(Y_L, nbt_Lg)
                ab1x_g = npy.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = npy.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = npy.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = npy.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2
                 
                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = npy.zeros(self.ng)
                vb_g = npy.zeros(self.ng)
                e_g = npy.zeros(self.ng)
                deda2_g = npy.zeros(self.ng)
                dedaa2_g = npy.zeros(self.ng)
                dedab2_g = npy.zeros(self.ng)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g,
                                               tauat_g, taubt_g, dedtaua_g,
                                               dedtaub_g)
                E -= w * npy.dot(e_g, self.dv_g)

                x_g = -deda2_g * self.dv_g * aa1_g 
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                     npy.dot(self.nt_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * aa1x_g ))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * aa1y_g ))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * aa1z_g ))
                dEdDb_p -= dEdD_p
                
                x_g = -deda2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp

                dEdD_p  = w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                      npy.dot(self.nt_qg, x_g))
                x_g = 4.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 0]),
                                      npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 1]),
                                      npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdD_p += w * npy.dot(npy.dot(self.B_pqL,
                                              A_Li[:, 2]),
                                      npy.dot(self.nt_qg, x_g * ab1z_g))
                dEdDa_p -= dEdD_p

                x_g = -2.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * aa1z_g))
                
                x_g = -2.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                       npy.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 0]),
                                       npy.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                            A_Li[:, 1]),
                                       npy.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * npy.dot(npy.dot(self.B_pqL,
                                               A_Li[:, 2]),
                                       npy.dot(self.nt_qg, x_g * ab1z_g))
                
                dedtaua_g *= self.dv_g
                dedtaub_g *= self.dv_g
                dEdDa_p -= w * npy.dot(taut_pg, dedtaua_g)
                dEdDb_p -= w * npy.dot(taut_pg, dedtaub_g)
                y += 1
        return E - self.Exc0

if extra_parameters.get('usenewxc'):
    XCCorrection = NewXCCorrection
