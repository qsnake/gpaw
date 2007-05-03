# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import Numeric as num
from multiarray import matrixproduct as dot3  # Avoid dotblas bug!

from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import YL

# load points and weights for the angular integration
from gpaw.sphere import Y_nL, points, weights

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


A_Liy = num.zeros((25, 3, len(points)), num.Float)

y = 0
for R in points:
    for l in range(5):
        for m in range(2 * l + 1):
            L = l**2 + m
            for c, n in YL[L]:
                for i in range(3):
                    ni = n[i]
                    if ni > 0:
                        a = ni * c * R[i]**(ni - 1)
                        for ii in range(3):
                            if ii != i:
                                a *= R[ii]**n[ii]
                        A_Liy[L, i, y] += a
            A_Liy[L, :, y] -= l * R * Y_nL[y, L]
    y += 1


class XCCorrection:
    def __init__(self,
                 xc,    # radial exchange-correlation object
                 w_j,   #
                 wt_j,  #
                 nc,    # core density 
                 nct,   # smooth core density
                 rgd,   # radial grid edscriptor
                 jl,    # ?
                 lmax,  # maximal angular momentum to consider
                 Exc0): # ?
        self.nc_g = nc
        self.nct_g = nct
        self.xc = xc
        self.Exc0 = Exc0
        self.Lmax = (lmax + 1)**2
        if lmax == 0:
            self.weights = [1.0]
            self.Y_yL = num.array([[1.0 / sqrt(4.0 * pi)]])
        else:
            self.weights = weights
            self.Y_yL = Y_nL[:, :self.Lmax].copy()
        jlL = []
        for j, l in jl:
            for m in range(2 * l + 1):
                jlL.append((j, l, l**2 + m))

        ng = len(nc)
        self.ng = ng
        ni = len(jlL)
        nj = len(jl)
        np = ni * (ni + 1) // 2
        nq = nj * (nj + 1) // 2
        self.B_Lqp = num.zeros((self.Lmax, nq, np), num.Float)
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
        self.B_pqL = num.transpose(self.B_Lqp).copy()
        self.dv_g = rgd.dv_g
        self.n_qg = num.zeros((nq, ng), num.Float)
        self.nt_qg = num.zeros((nq, ng), num.Float)
        q = 0
        for j1, l1 in jl:
            for j2, l2 in jl[j1:]:
                rl1l2 = rgd.r_g**(l1 + l2)
                self.n_qg[q] = rl1l2 * w_j[j1] * w_j[j2]
                self.nt_qg[q] = rl1l2 * wt_j[j1] * wt_j[j2]
                q += 1
        self.rgd = rgd
       
    def calculate_energy_and_derivatives(self, D_sp, H_sp):
        if self.xc.get_functional().gga:
            return self.GGA(D_sp, H_sp)
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = num.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = num.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            for w, Y_L in zip(self.weights, self.Y_yL):
                n_g = num.dot(Y_L, n_Lg)
                vxc_g = num.zeros(self.ng, num.Float)
                E += self.xc.get_energy_and_potential(n_g, vxc_g) * w
                dEdD_q = num.dot(self.n_qg, vxc_g * self.dv_g)
                nt_g = num.dot(Y_L, nt_Lg)
                vxct_g = num.zeros(self.ng, num.Float) 
                E -= self.xc.get_energy_and_potential(nt_g, vxct_g) * w
                dEdD_q -= num.dot(self.nt_qg, vxct_g * self.dv_g)
                dEdD_p += num.dot(dot3(self.B_pqL, Y_L),
                                  dEdD_q) * w
        else:
            Da_p = D_sp[0]
            Da_Lq = dot3(self.B_Lqp, Da_p)
            na_Lg = num.dot(Da_Lq, self.n_qg)
            na_Lg[0] += 0.5 * self.nc_g * sqrt(4 * pi)
            nta_Lg = num.dot(Da_Lq, self.nt_qg)
            nta_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            Db_p = D_sp[1]
            Db_Lq = dot3(self.B_Lqp, Db_p)
            nb_Lg = num.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += 0.5 * self.nc_g * sqrt(4 * pi)
            ntb_Lg = num.dot(Db_Lq, self.nt_qg)
            ntb_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            for w, Y_L in zip(self.weights, self.Y_yL):
                na_g = num.dot(Y_L, na_Lg)
                vxca_g = num.zeros(self.ng, num.Float) 
                nb_g = num.dot(Y_L, nb_Lg)
                vxcb_g = num.zeros(self.ng, num.Float) 
                E += self.xc.get_energy_and_potential(na_g, vxca_g,
                                                   nb_g, vxcb_g) * w
                dEdDa_q = num.dot(self.n_qg, vxca_g * self.dv_g)
                dEdDb_q = num.dot(self.n_qg, vxcb_g * self.dv_g)
                nta_g = num.dot(Y_L, nta_Lg)
                vxcta_g = num.zeros(self.ng, num.Float) 
                ntb_g = num.dot(Y_L, ntb_Lg)
                vxctb_g = num.zeros(self.ng, num.Float) 
                E -= self.xc.get_energy_and_potential(nta_g, vxcta_g,
                                                   ntb_g, vxctb_g) * w
                dEdDa_q -= num.dot(self.nt_qg, vxcta_g * self.dv_g)
                dEdDb_q -= num.dot(self.nt_qg, vxctb_g * self.dv_g)
                dEdDa_p += num.dot(dot3(self.B_pqL, Y_L),
                                  dEdDa_q) * w
                dEdDb_p += num.dot(dot3(self.B_pqL, Y_L),
                                  dEdDb_q) * w

        return E - self.Exc0

    def GGA(self, D_sp, H_sp):
        r_g = self.rgd.r_g
        xcfunc = self.xc.get_functional()
        E = 0.0
        if len(D_sp) == 1:
            D_p = D_sp[0]
            D_Lq = dot3(self.B_Lqp, D_p)
            n_Lg = num.dot(D_Lq, self.n_qg)
            n_Lg[0] += self.nc_g * sqrt(4 * pi)
            nt_Lg = num.dot(D_Lq, self.nt_qg)
            nt_Lg[0] += self.nct_g * sqrt(4 * pi)
            dndr_Lg = num.zeros((self.Lmax, self.ng), num.Float)
            dntdr_Lg = num.zeros((self.Lmax, self.ng), num.Float)
            for L in range(self.Lmax):
                self.rgd.derivative(n_Lg[L], dndr_Lg[L])
                self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
            dEdD_p = H_sp[0][:]
            dEdD_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]
                n_g = num.dot(Y_L, n_Lg)
                a1x_g = num.dot(A_Li[:, 0], n_Lg)
                a1y_g = num.dot(A_Li[:, 1], n_Lg)
                a1z_g = num.dot(A_Li[:, 2], n_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = num.dot(Y_L, dndr_Lg)
                a2_g += a1_g**2
                v_g = num.zeros(self.ng, num.Float) 
                e_g = num.zeros(self.ng, num.Float) 
                deda2_g = num.zeros(self.ng, num.Float)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E += w * num.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p += w * num.dot(dot3(self.B_pqL, Y_L),
                                      num.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.n_qg, x_g * a1x_g))
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.n_qg, x_g * a1y_g))
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.n_qg, x_g * a1z_g))

                n_g = num.dot(Y_L, nt_Lg)
                a1x_g = num.dot(A_Li[:, 0], nt_Lg)
                a1y_g = num.dot(A_Li[:, 1], nt_Lg)
                a1z_g = num.dot(A_Li[:, 2], nt_Lg)
                a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a1_g = num.dot(Y_L, dntdr_Lg)
                a2_g += a1_g**2
                v_g = num.zeros(self.ng, num.Float) 
                e_g = num.zeros(self.ng, num.Float) 
                deda2_g = num.zeros(self.ng, num.Float)
                xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
                E -= w * num.dot(e_g, self.dv_g)
                x_g = -2.0 * deda2_g * self.dv_g * a1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += v_g * self.dv_g
                B_Lqp = self.B_Lqp
                dEdD_p -= w * num.dot(dot3(self.B_pqL, Y_L),
                                      num.dot(self.nt_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.nt_qg, x_g * a1x_g))
                dEdD_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.nt_qg, x_g * a1y_g))
                dEdD_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.nt_qg, x_g * a1z_g))

                y += 1
        else:
            Da_p = D_sp[0]
            Da_Lq = dot3(self.B_Lqp, Da_p)
            na_Lg = num.dot(Da_Lq, self.n_qg)
            na_Lg[0] += 0.5 * self.nc_g * sqrt(4 * pi)
            nat_Lg = num.dot(Da_Lq, self.nt_qg)
            nat_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnadr_Lg = num.zeros((self.Lmax, self.ng), num.Float)
            dnatdr_Lg = num.zeros((self.Lmax, self.ng), num.Float)
            for L in range(self.Lmax):
                self.rgd.derivative(na_Lg[L], dnadr_Lg[L])
                self.rgd.derivative(nat_Lg[L], dnatdr_Lg[L])
            dEdDa_p = H_sp[0][:]
            dEdDa_p[:] = 0.0
            
            Db_p = D_sp[1]
            Db_Lq = dot3(self.B_Lqp, Db_p)
            nb_Lg = num.dot(Db_Lq, self.n_qg)
            nb_Lg[0] += 0.5 * self.nc_g * sqrt(4 * pi)
            nbt_Lg = num.dot(Db_Lq, self.nt_qg)
            nbt_Lg[0] += 0.5 * self.nct_g * sqrt(4 * pi)
            dnbdr_Lg = num.zeros((self.Lmax, self.ng), num.Float)
            dnbtdr_Lg = num.zeros((self.Lmax, self.ng), num.Float)
            for L in range(self.Lmax):
                self.rgd.derivative(nb_Lg[L], dnbdr_Lg[L])
                self.rgd.derivative(nbt_Lg[L], dnbtdr_Lg[L])
            dEdDb_p = H_sp[1][:]
            dEdDb_p[:] = 0.0
            y = 0
            for w, Y_L in zip(self.weights, self.Y_yL):
                A_Li = A_Liy[:self.Lmax, :, y]

                na_g = num.dot(Y_L, na_Lg)
                aa1x_g = num.dot(A_Li[:, 0], na_Lg)
                aa1y_g = num.dot(A_Li[:, 1], na_Lg)
                aa1z_g = num.dot(A_Li[:, 2], na_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = num.dot(Y_L, dnadr_Lg)
                aa2_g += aa1_g**2

                nb_g = num.dot(Y_L, nb_Lg)
                ab1x_g = num.dot(A_Li[:, 0], nb_Lg)
                ab1y_g = num.dot(A_Li[:, 1], nb_Lg)
                ab1z_g = num.dot(A_Li[:, 2], nb_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = num.dot(Y_L, dnbdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = num.zeros(self.ng, num.Float) 
                vb_g = num.zeros(self.ng, num.Float) 
                e_g = num.zeros(self.ng, num.Float) 
                deda2_g = num.zeros(self.ng, num.Float)
                dedaa2_g = num.zeros(self.ng, num.Float)
                dedab2_g = num.zeros(self.ng, num.Float)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E += w * num.dot(e_g, self.dv_g)

                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                B_Lqp = self.B_Lqp
                
                dEdD_p = w * num.dot(dot3(self.B_pqL, Y_L),
                                     num.dot(self.n_qg, x_g))
                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.n_qg, x_g * (aa1x_g +
                                                                ab1x_g)))
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.n_qg, x_g * (aa1y_g +
                                                                ab1y_g)))
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.n_qg, x_g * (aa1z_g +
                                                                ab1z_g)))
                dEdDa_p += dEdD_p
                dEdDb_p += dEdD_p

                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p += w * num.dot(dot3(self.B_pqL, Y_L),
                                      num.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.n_qg, x_g * aa1x_g))
                dEdDa_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.n_qg, x_g * aa1y_g))
                dEdDa_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.n_qg, x_g * aa1z_g))

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p += w * num.dot(dot3(self.B_pqL, Y_L),
                                      num.dot(self.n_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.n_qg, x_g * ab1x_g))
                dEdDb_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.n_qg, x_g * ab1y_g))
                dEdDb_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.n_qg, x_g * ab1z_g))

                na_g = num.dot(Y_L, nat_Lg)
                aa1x_g = num.dot(A_Li[:, 0], nat_Lg)
                aa1y_g = num.dot(A_Li[:, 1], nat_Lg)
                aa1z_g = num.dot(A_Li[:, 2], nat_Lg)
                aa2_g = aa1x_g**2 + aa1y_g**2 + aa1z_g**2
                aa2_g[1:] /= r_g[1:]**2
                aa2_g[0] = aa2_g[1]
                aa1_g = num.dot(Y_L, dnatdr_Lg)
                aa2_g += aa1_g**2

                nb_g = num.dot(Y_L, nbt_Lg)
                ab1x_g = num.dot(A_Li[:, 0], nbt_Lg)
                ab1y_g = num.dot(A_Li[:, 1], nbt_Lg)
                ab1z_g = num.dot(A_Li[:, 2], nbt_Lg)
                ab2_g = ab1x_g**2 + ab1y_g**2 + ab1z_g**2
                ab2_g[1:] /= r_g[1:]**2
                ab2_g[0] = ab2_g[1]
                ab1_g = num.dot(Y_L, dnbtdr_Lg)
                ab2_g += ab1_g**2

                a2_g = ((aa1x_g + ab1x_g)**2 +
                        (aa1y_g + ab1y_g)**2 +
                        (aa1z_g + ab1z_g)**2)
                a2_g[1:] /= r_g[1:]**2
                a2_g[0] = a2_g[1]
                a2_g += (aa1_g + ab1_g)**2

                va_g = num.zeros(self.ng, num.Float) 
                vb_g = num.zeros(self.ng, num.Float) 
                e_g = num.zeros(self.ng, num.Float) 
                deda2_g = num.zeros(self.ng, num.Float)
                dedaa2_g = num.zeros(self.ng, num.Float)
                dedab2_g = num.zeros(self.ng, num.Float)
                xcfunc.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                               a2_g, aa2_g, ab2_g,
                                               deda2_g, dedaa2_g, dedab2_g)
                E -= w * num.dot(e_g, self.dv_g)
                
                x_g = -2.0 * deda2_g * self.dv_g * (aa1_g + ab1_g)
                self.rgd.derivative2(x_g, x_g)
                dEdD_p = w * num.dot(dot3(self.B_pqL, Y_L),
                                     num.dot(self.nt_qg, x_g))

                x_g = 8.0 * pi * deda2_g * self.rgd.dr_g
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.nt_qg, x_g * (aa1x_g +
                                                                ab1x_g)))
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.nt_qg, x_g * (aa1y_g +
                                                                ab1y_g)))
                dEdD_p += w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.nt_qg, x_g * (aa1z_g +
                                                                ab1z_g)))
                dEdDa_p -= dEdD_p
                dEdDb_p -= dEdD_p

                x_g = -4.0 * dedaa2_g * self.dv_g * aa1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += va_g * self.dv_g
                dEdDa_p -= w * num.dot(dot3(self.B_pqL, Y_L),
                                      num.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedaa2_g * self.rgd.dr_g
                dEdDa_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.nt_qg, x_g * aa1x_g))
                dEdDa_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.nt_qg, x_g * aa1y_g))
                dEdDa_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.nt_qg, x_g * aa1z_g))

                x_g = -4.0 * dedab2_g * self.dv_g * ab1_g
                self.rgd.derivative2(x_g, x_g)
                x_g += vb_g * self.dv_g
                dEdDb_p -= w * num.dot(dot3(self.B_pqL, Y_L),
                                      num.dot(self.nt_qg, x_g))
                x_g = 16.0 * pi * dedab2_g * self.rgd.dr_g
                dEdDb_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 0]),
                                      num.dot(self.nt_qg, x_g * ab1x_g))
                dEdDb_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 1]),
                                      num.dot(self.nt_qg, x_g * ab1y_g))
                dEdDb_p -= w * num.dot(dot3(self.B_pqL,
                                              A_Li[:, 2]),
                                      num.dot(self.nt_qg, x_g * ab1z_g))

                y += 1

        return E - self.Exc0

    def two_phi_integrals(self,
                          D_sp # density matrix in packed form
                          ):
        """Evaluate the integral in the augmentation sphere
                    /
        I_{i1 i2} = | d r [ phi_i1(r) phi_i2(r) v_xc[n](r) -
                    /       tphi_i1(r) tphi_i2(r) v_xc[tn](r) ]
                    a
        The result is given in packed form
        """
        pass

    def four_phi_integrals(self,
                           D_sp, # density matrix in packed form
                           f_xc  # the fxc radial object
                           ):
        pass
    
