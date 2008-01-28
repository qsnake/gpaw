from gpaw.gllb import find_nucleus

import numpy as npy

from gpaw.gaunt import gaunt
from gpaw.sphere import Y_nL, points, weights
from gpaw.spherical_harmonics import YL



class DummyXC:
    def set_functional(self, xc):
        pass
        #print "GLLB: DummyXC::set_functional(xc) with ", xc.xcname

A_Liy = npy.zeros((25, 3, len(points)))

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


class XCNonLocalCorrection:
    def __init__(self,
                 xcfunc, # radial exchange-correlation object
                 w_j,    #
                 wt_j,   #
                 nc,     # core density
                 nct,    # smooth core density
                 rgd,    # radial grid edscriptor
                 jl,     # ?
                 lmax,   # maximal angular momentum to consider
                 Exc0,   # ?
                 extra_xc_data): # The response parts of core orbitals


        #print "Initializing XCNonLocalCorrection"
        # Some part's of code access xc.xcfunc.hydrid, this is to ensure
        # that is does not cause error
        self.xc = DummyXC()
        self.xc.xcfunc = DummyXC()
        self.xc.xcfunc.hybrid = 0.0

        self.extra_xc_data = extra_xc_data

        self.nc_g = nc
        self.nct_g = nct

        #from gpaw.xc_functional import XCRadialGrid, XCFunctional
        #self.slater_part = XCFunctional(SLATER_FUNCTIONAL, 1)

        self.motherxc = xcfunc

        self.Exc0 = Exc0
        self.Lmax = (lmax + 1)**2
        if lmax == 0:
            self.weights = [1.0]
            self.Y_yL = npy.array([[1.0 / npy.sqrt(4.0 * npy.pi)]])
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
        self.np = np
        nq = nj * (nj + 1) // 2
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
                self.n_qg[q] = rl1l2 * w_j[j1] * w_j[j2]
                self.nt_qg[q] = rl1l2 * wt_j[j1] * wt_j[j2]
                q += 1
        self.rgd = rgd

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):


        # This method is called before initialization of motherxc in pass_stuff
        if self.motherxc.slater_part == None:
            #print "GLLB: Not applying the PAW-corrections!"
            H_sp[:] = 0.0
            return 0 #Grr....

        if not self.motherxc.initialization_ready:
            #print "GLLB: Initialization not ready."
            H_sp[:] = 0.0
            return 0 #Grr...

        deg = len(D_sp)
        Exc = 0
        # The calculation of exchange potential is spin-independent
        for s, (D_p, H_p) in enumerate(zip(D_sp, H_sp)):
            Exc += self.calculate_gllb_exchange(D_p, H_p, s, deg, a)

        #print "D_sp", D_sp
        return Exc

    def calculate_gllb_exchange(self, D_p, H_p, s, deg, a):

        # THIS DOES NOT WORK IN PARALLEL nucleus = self.motherxc.nuclei[a] # Get the nucleus with index
        nucleus = find_nucleus(self.motherxc.nuclei, a)

        ni = nucleus.get_number_of_partial_waves() # Get the number of partial waves from nucleus
        np = ni * (ni + 1) // 2 # Number of items in packed density matrix

        Dn_ii = npy.zeros((ni, ni)) # Allocate space for unpacked atomic density matrix
        Dn_p = npy.zeros((np, np)) # Allocate space for packed atomic density matrix

        r_g = self.rgd.r_g
        #xcfunc = self.slater_part

        # The total exchange integral
        E = 0.0

        D_Lq = npy.dot(self.B_Lqp, D_p)
        n_Lg = npy.dot(D_Lq, self.n_qg)
        n_Lg[0] += self.nc_g * npy.sqrt(4 * npy.pi) / deg

        nt_Lg = npy.dot(D_Lq, self.nt_qg)
        nt_Lg[0] += self.nct_g * npy.sqrt(4 * npy.pi) / deg
        dndr_Lg = npy.zeros((self.Lmax, self.ng))
        dntdr_Lg = npy.zeros((self.Lmax, self.ng))

        # Array for exchange potential
        v_g = npy.zeros(len(r_g))
        # Array for smooth exchange potential
        vt_g = npy.zeros(len(r_g))

        for L in range(self.Lmax):
            self.rgd.derivative(n_Lg[L], dndr_Lg[L])
            self.rgd.derivative(nt_Lg[L], dntdr_Lg[L])

        dEdD_p = H_p
        dEdD_p[:] = 0.0
        self.deg = deg

        y = 0
        for slice, (w, Y_L) in enumerate(zip(self.weights, self.Y_yL)):
            A_Li = A_Liy[:self.Lmax, :, y]

            self.Y_L = Y_L

            # Calculate the true density
            self.n_g = npy.dot(Y_L, n_Lg)

            # Calculate gradients for ae-density
            self.a1x_g = npy.dot(A_Li[:, 0], n_Lg)
            self.a1y_g = npy.dot(A_Li[:, 1], n_Lg)
            self.a1z_g = npy.dot(A_Li[:, 2], n_Lg)
            self.a2_g = self.a1x_g**2 + self.a1y_g**2 + self.a1z_g**2
            self.a2_g[1:] /= r_g[1:]**2
            self.a2_g[0] = self.a2_g[1]
            self.a1_g = npy.dot(Y_L, dndr_Lg)
            self.a2_g += self.a1_g**2

            # Calculate the pseudo density
            self.nt_g = npy.dot(Y_L, nt_Lg)
            # Calculate the gradients for pseudo density
            self.a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
            self.a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
            self.a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
            self.at2_g = self.a1x_g**2 + self.a1y_g**2 + self.a1z_g**2
            self.at2_g[1:] /= r_g[1:]**2
            self.at2_g[0] = self.at2_g[1]
            self.a1_g = npy.dot(Y_L, dntdr_Lg)
            self.at2_g += self.a1_g**2

            # Adjust the densities and gradients for spin-count
            # Note, since there is no exchange interaction between different spins, if must hold that
            # E_x[\rho_up, \rho_down] = 1/2 E_x[2\rho_up] + 1/2 E_x[2\rho_down]
            self.n_g *= deg
            self.nt_g *= deg
            self.a2_g*= deg**2
            self.at2_g*= deg**2

            self.D_p = D_p
            E += w / deg * self.motherxc.calculate_non_local_paw_correction(a, s, self, slice, v_g, vt_g)

            # Integrate the slice with respect to orbitals
            dEdD_p += w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                  npy.dot(self.n_qg, v_g * self.dv_g))

            dEdD_p -= w * npy.dot(npy.dot(self.B_pqL, Y_L),
                                  npy.dot(self.nt_qg, vt_g * self.dv_g))
            y += 1

        return (E) - self.Exc0 / deg

