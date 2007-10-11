import Numeric as num
import LinearAlgebra as linalg
from gpaw.Function1D import Function1D
from math import sqrt, pi
from gpaw.utilities import hartree, packed_index, unpack, unpack2, pack, pack2, fac
from LinearAlgebra import inverse

# For XCCorrections
from multiarray import matrixproduct as dot3
from multiarray import innerproduct as inner # avoid the dotblas version!
from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import YL
from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities.complex import cc, real


# load points and weights for the angular integration
from gpaw.sphere import Y_nL, points, weights

#import pylab

# GLLB-Functional
# Gritsenko, Leeuwen, Lenthe, Baerends: Self-consistent approximation to the Kohn-Shan exchange potential
# Physical Review A, vol. 51, p. 1944, March 1995.
# GLLB-Functional is of the same form than KLI-Functional, but it 
# 1) approximates the numerator part of Slater-potential from 2*GGA-energy density
# 2) approximates the response part coefficients from eigenvalues.

SLATER_FUNCTIONAL = "X_B88-None"
SMALL_NUMBER = 1e-8
K_G = 0.382106112167171

class GLLBFunctional:
    def __init__(self):
        self.slater_part = None
        self.initialization_ready = False
        self.fermi_level = -1000
        
    # Called from xc_functional::set_non_local_things method
    # All the necessary classes and methods are passed through this method
    # Not used in 1D-calculations
    def pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation):
        
        self.kpt_u = kpt_u
        self.finegd = finegd
        self.interpolate = interpolate
        self.nspins = nspins
        self.nuclei = nuclei
        self.occupation = occupation
        
        self.tempvxc_g = finegd.zeros()
        self.tempe_g = finegd.zeros()
        
        self.vt_G = gd.zeros()
        self.vt_g = finegd.zeros()
        self.nt_G = gd.zeros()
        self.vt_g = finegd.zeros()
        
    def get_gllb_weight(self, epsilon, fermi_level):
        # Without this systems with degenerate homo-orbitals have convergence problems
        if (epsilon+1e-3) > fermi_level:
            return 0
        return K_G * sqrt(fermi_level - epsilon)

    # input:  ae : AllElectron object.
    # output: extra_xc_data : dictionary. A Dictionary with pair ('name', radial grid)
    def calculate_extra_setup_data(self, extra_xc_data, ae):
        print "Calculating response part for core-electrons..."
        N = len(ae.rgd.r_g)
        v_xc = num.zeros(N, num.Float)
        # Calculate the response part using wavefunctions, eigenvalues etc. from AllElectron calculator

        self.get_non_local_energy_and_potential1D(ae.rgd,
                                                  ae.u_j,
                                                  ae.f_j,
                                                  ae.e_j,
                                                  ae.l_j,
                                                  v_xc,
                                                  njcore = ae.njcore)

        extra_xc_data['core_response'] = v_xc
        
    # Input  gd  : RadialGridDescriptor
    #        u_j : array of wavefunctions
    #        f_j : array of occupation numbers
    #        e_j : array of eigenvalues
    #        l_j : array of angular momenta
    #     njcore : integer. If specified, outputs only the response part for 'njcore' lowest orbitals.
    # Output v_xc: The GLLB-exchange potential in radial grid
    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, v_xc, njcore=None, density=None):

        if self.slater_part == None:
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.slater_part = XCRadialGrid(XCFunctional(SLATER_FUNCTIONAL, 1), gd)
        
        N = len(gd.r_g)
        
	if density == None:
             # Construct the density from supplied orbitals
             n_g = num.dot(f_j, num.where(abs(u_j) < 1e-160, 0, u_j)**2) / (4 * pi)
             n_g[1:] /= gd.r_g[1:]**2
             n_g[0] = n_g[1]
        else:
             # The density is already supplied
             n_g = density.copy()

        # Create arrays for energy-density and potential
        e_g = num.zeros(N, num.Float)
        v_g = num.zeros(N, num.Float)

        # Calculate gga-energy density
        self.slater_part.get_energy_and_potential_spinpaired(n_g, v_g, e_g=e_g)

        Exc = num.dot(e_g, gd.dv_g)
    
        if njcore == None:
            # Approximate the numerator of slater potential with 2*e_g
            v_xc[:] += 2 * e_g / (n_g + SMALL_NUMBER)

        # Find fermi-level
        # Grr!!! sum(num.where(f_j>1e-3, 1, 0))-1 doesn't work
        self.fermi_level = -1000
        for i, (f,e) in enumerate(zip(f_j, e_j)):
            if f > 1e-7:
                if self.fermi_level < e:
                    self.fermi_level = e

        if njcore == None:
            imax = len(f_j)
        else:
            # Add the potential only from core orbitals
            imax = njcore
            
        for i in range(0, imax):
            # Construct the orbital-density of ith orbital
            nn_g = f_j[i] * num.where(abs(u_j[i]) < 1e-160, 0, u_j[i])**2 / (4 * pi) #grr.. Hack for numeric
            nn_g[1:] /= gd.r_g[1:]**2
            nn_g[0] = n_g[1]
            
            v_xc[:] += self.get_gllb_weight(e_j[i], self.fermi_level) *  (nn_g / (n_g + SMALL_NUMBER))

        # There is a serious error in v_xc[0], replace it with v_xc[1]
        v_xc[0] = v_xc[1]

        return Exc

    def add_response_part(self, kpt, vt_G, nt_G, fermi_level):
        """Add contribution of response part to pseudo electron-density."""
        if kpt.typecode is num.Float:
            for psit_G, f, e in zip(kpt.psit_nG, kpt.f_n, kpt.eps_n):
                axpy(f*self.get_gllb_weight(e, fermi_level), psit_G**2, vt_G)  # nt_G += f * psit_G**2
                axpy(f, psit_G**2, nt_G)
        else:
            print "Adding response part COMPLEX. NOT TESTED!"
            for psit_G, f,e  in zip(kpt.psit_nG, kpt.f_n, kpt.eps_n):
                vt_G += f * self.get_gllb_weight(e, fermi_level) * (psit_G * num.conjugate(psit_G)).real
                nt_G += f * (psit_G * num.conjugate(psit_G)).real
                                                                
            
    def calculate_spinpaired(self, e_g, n_g, v_g):
        # Create the revPBEx functional for Slater part (only once per calculation)
        if self.slater_part == None:
            from gpaw.xc_functional import XCFunctional, XC3DGrid
            self.slater_part = XC3DGrid(XCFunctional(SLATER_FUNCTIONAL, self.nspins), self.finegd, self.nspins)

        # Calculate the approximative Slater-potential
        self.slater_part.get_energy_and_potential_spinpaired(n_g, self.tempvxc_g, e_g=self.tempe_g)
        # Add it to the total potential
        v_g += 2*self.tempe_g / (n_g + SMALL_NUMBER)

        # Return the xc-energy
        e_g[:] = self.tempe_g.flat

        # Get the fermi-level from occupations
        try:
            self.fermi_level_old = self.fermi_level
            self.fermi_level = -1000
            for kpt in self.kpt_u:
                for e, f in zip(kpt.eps_n, kpt.f_n):
                    if f > 1e-2:
                        if self.fermi_level < e:
                            self.fermi_level = e
                            
        except AttributeError:
            self.fermi_level = -1000
            print "FERMILEVEL-1000 FIXME!"

        # Coarse grid for response part
        self.vt_G[:] = 0.0
        self.nt_G[:] = 0.0
        # For each k-point
        for kpt in self.kpt_u:
            # Check if we already have eigenvalues
            self.initialization_ready = True
            try:
                e_n = kpt.eps_n
            except AttributeError:
                print "INITIALIZATION NOT YET READY" #XXXXX
                self.initialization_ready = False #GRRR

            if self.initialization_ready:
                self.add_response_part(kpt, self.vt_G, self.nt_G, self.fermi_level)

        self.vt_g[:] = 0.0
        self.vt_G[:] /= self.nt_G[:] + SMALL_NUMBER

        # It's faster to add wavefunctions in coarse-grid and interpolate afterwards
        self.interpolate(self.vt_G, self.vt_g)
        # Add the fine-grid response part to total potential
        v_g[:] += self.vt_g 
        
    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        print "GLLB calculate_spinpolarized not implemented"
        pass

class DummyXC:
    pass

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


class XCGLLBCorrection:
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
                 core_response): # The response parts of core orbitals
                

        self.xc = DummyXC()
        self.xc.xcfunc = DummyXC()
        self.xc.xcfunc.hybrid = 0.0

        self.core_response = core_response.copy()

        self.nc_g = nc
        self.nct_g = nct

        from xc_functional import XCRadialGrid, XCFunctional
        self.slater_part = XCFunctional(SLATER_FUNCTIONAL, 1)

        self.motherxc = xcfunc

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
       
    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        # This is the code from GGA-method of XCCorrections, but
        # it has lines involving GLLB. All lines which contain
        # comments are NOT from xc_corrections.py file:)

        self.nspins = 1 # XXXX SPINHACK
        
        # This method is called before initialization of motherxc in pass_stuff
        if self.motherxc.slater_part == None:
            print "Grr...!!!!!!!!!"
            return 0 #Grr....

        #print "D_sp", D_sp

        nucleus = self.motherxc.nuclei[a] # Get the nucleus with index
        ni = nucleus.get_number_of_partial_waves() # Get the number of partial waves from nucleus
        np = ni * (ni + 1) // 2 # Number of items in packed density matrix
        
        Dn_ii = num.zeros((ni, ni), num.Float) # Allocate space for unpacked atomic density matrix
        Dn_p = num.zeros((np, np), num.Float) # Allocate space for packed atomic density matrix
 
        r_g = self.rgd.r_g
        xcfunc = self.slater_part #get_functional()

        # The total exchange integral
        E = 0.0
        # The total pseudo-exchange integral
        Et = 0.0

        if not len(D_sp) == 1:
            raise "Spin polarized calculation not implemented yet"
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

            if self.motherxc.initialization_ready:
                # For each k-point
                for kpt in self.motherxc.kpt_u:
                    # Get the projection coefficients
                    P_ni = nucleus.P_uni[kpt.u]
                    # Create the coefficients
                    w_i = num.zeros(kpt.eps_n.shape, num.Float)
                    for i in range(len(w_i)):
                        w_i[i] = self.motherxc.get_gllb_weight(kpt.eps_n[i], self.motherxc.fermi_level)

                    w_i = w_i[:, num.NewAxis] * kpt.f_n[:, num.NewAxis] # Calculate the weights

                    # Calculate the 'density matrix' for numerator part of potential
                    Dn_ii = real(num.dot(cc(num.transpose(P_ni)),
                                         P_ni * w_i))
                
                    Dn_p = pack(Dn_ii) # Pack the unpacked densitymatrix

                    Dnn_Lq = dot3(self.B_Lqp, Dn_p) #Contract one nmln'm'l'
                    nn_Lg = num.dot(Dnn_Lq, self.n_qg) # Contract nln'l'
                    nn = num.dot(Y_L, nn_Lg) ### Contract L
            else:
                nn = 0.0

            # Add the Slater-part
            x_g = (2*e_g + nn) / (n_g + SMALL_NUMBER) * self.dv_g
            # Add the response from core
            x_g += self.core_response * self.dv_g

            # Calculate the slice
            dEdD_p += w * num.dot(dot3(self.B_pqL, Y_L),
                                  num.dot(self.n_qg, x_g))
            
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
            Et += w * num.dot(e_g, self.dv_g)

            if self.motherxc.initialization_ready:
                #Dnn_Lq = dot3(self.B_Lqp, Dn_sp) #Contract one nmln'm'l'
                nn_Lg = num.dot(Dnn_Lq, self.nt_qg) # Contract nln'l'
                nn = num.dot(Y_L, nn_Lg) ### Contract L
            else:
                nn = 0.0
                
            x_g = (2*e_g + nn) / (n_g + SMALL_NUMBER) * self.dv_g
            
            dEdD_p -= w * num.dot(dot3(self.B_pqL, Y_L),
                                  num.dot(self.nt_qg, x_g))
            y += 1

        return (E-Et) - self.Exc0
        

class XCKLICorrection:
    def __init__(self, xcfunc, r, dr, beta, N, nspins, M_pp, X_p, ExxC, phi, phit, jl, lda_xc):
        self.xcfunc = xcfunc
        self.nspins = nspins
        self.M_pp = M_pp
        self.X_p  = X_p
        self.ExxC = ExxC
        self.phi = phi
        self.r = r.copy()
        self.r[0] = self.r[1]
        self.dr = dr
        self.beta = beta
        self.N = N
        self.phit = phit
        self.jl = jl

        self.xc = DummyXC()
        self.xc.xcfunc = DummyXC()
        self.xc.xcfunc.hybrid = 0.0
        self.lda_xc = lda_xc
        jlm = []
        for j, l in jl:
            for m in range(-l, l+1):
                jlm.append((j, l, m))
                
        self.jlm = jlm
      
    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        deg = 2 / self.nspins     # Spin degeneracy

        E = 0.0
        hybrid = 1.
        #print "Density matrix", D_sp
        
        for s in range(self.nspins):
            # Get atomic density and Hamiltonian matrices
            D_p  = D_sp[s]
            D_ii = unpack2(D_p)
            H_p  = H_sp[s]
            ni = len(D_ii)

            # Add atomic corrections to the valence-valence exchange energy
            # --
            # >  D   C     D
            # --  ii  iiii  ii
            C_pp = self.M_pp
            for i1 in range(ni):
                for i2 in range(ni):
                    A = 0.0 # = C * D
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            A += C_pp[p13, p24] * D_ii[i3, i4]
                    p12 = packed_index(i1, i2, ni)
                    # Calculate energy only!
                    #H_p[p12] -= 2 * hybrid / deg * A / ((i1!=i2) + 1)
                    E -= hybrid / deg * D_ii[i1, i2] * A

            # Add valence-core exchange energy
            # --
            # >  X   D
            # --  ii  ii
            #E -= hybrid * num.dot(D_p, self.X_p)
            #H_p -= hybrid * self.X_p

        # Add core-core exchange energy
        #E += hybrid * self.ExxC

        nspins  = self.xcfunc.nspins
        nbands  = self.xcfunc.nbands
        
        print "WONT PARALELLRIZE!"
        nucleus = self.xcfunc.ghat_nuclei[a]

        def create_cross_density(nucleus, partial_waves, n1, n2):
            density = Function1D()

            # What an index mess...
            for i1, (j1, l1, m1) in enumerate(self.jlm):
                for i2, (j2, l2, m2) in enumerate(self.jlm):
                    density += Function1D(l1, m1, nucleus.P_uni[spin, n1, i1] * partial_waves[j1]) * Function1D(l2, m2, nucleus.P_uni[spin, n2, i2] * partial_waves[j2])

            return density

        tempKLI = H_sp.copy()
        tempKLI[:] = 0
        
        for spin in range(0, nspins):
            vkli = Function1D()
            vtkli = Function1D()
            vn = Function1D()
            vnt = Function1D()
            for n1 in range(0, nbands):
                for n2 in range(n1, nbands):
                    n_nn  = create_cross_density(nucleus, self.phi, n1, n2)
                    nt_nn = create_cross_density(nucleus, self.phit, n1, n2)
                    if n1 == n2:
                        vn = vn + n_nn
                        vnt = vnt + nt_nn
                        
                    # Generate density matrix
                    P1_i = nucleus.P_uni[spin, n1]
                    P2_i = nucleus.P_uni[spin, n2]
                    D_ii = num.outerproduct(P1_i, P2_i)
                    D_p  = pack(D_ii, tolerance=1e3)#python func! move to C

                    # Determine compensation charge coefficients:
                    Q_L = num.dot(D_p, nucleus.setup.Delta_pL)

                    d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
                           for l in range(nucleus.setup.lmax + 1)]
                    g = nucleus.setup.alpha2**1.5 * num.exp(-nucleus.setup.alpha2 * self.r**2)
                    g[-1] = 0.0
                    #print "Compensation charges:", Q_L

                    index = 0
                    for l in range(nucleus.setup.lmax + 1):
                        radial = d_l[l] * nucleus.setup.alpha2**l * g * self.r**l
                        for m in range(-l, l+1):
                            #nt_nn = nt_nn + Function1D(l, m, Q_L[index]*radial)
                            index += 1

                    v_nn = n_nn.solve_poisson(self.r, self.dr, self.beta, self.N)
                    vt_nn = nt_nn.solve_poisson(self.r, self.dr, self.beta, self.N)

                    #pylab.plot(self.r, n_nn.integrateY())
                    #pylab.plot(self.r, nt_nn.integrateY())
                    #pylab.plot(self.r, v_nn.integrateY())
                    #pylab.show()
                    vkli = vkli + v_nn * n_nn
                    vtkli = vtkli + vt_nn * nt_nn
                    #print "IN KLICORRECTION: Vx_nnnlm for ",n1,n2, nucleus.Vx_nnnlm[n1,n2]
                    
            for i1, (j1, l1, m1) in enumerate(self.jlm):
                for i2, (j2, l2, m2) in enumerate(self.jlm):
                    if i1 == j2:
                        dc = 1
                    else:
                        dc = 0.5
                        
                    coeff = (vkli * Function1D(l1, m1, self.phi[j1]) * Function1D(l2, m2, self.phi[j2])).integrate_with_denominator(vn, self.r, self.dr)
                    coeff -= (vtkli * Function1D(l1, m1, self.phit[j1]) * Function1D(l2, m2, self.phit[j2])).integrate_with_denominator(vnt, self.r, self.dr)

                    tempKLI[spin, packed_index(i1,i2, nucleus.setup.ni)] += coeff * dc
                    
        tempLDA = H_sp.copy()
        self.lda_xc.calculate_energy_and_derivatives(D_sp, tempLDA)
        print "LDA d H_sp", tempLDA
        print "KLI d H_sp", tempKLI
        print "ratio of LDA/KLI ", tempLDA/(tempKLI +1e-20)

        # Currently just use LDA for atomic centered corrections
        # NOTE! H_sp seems to contain some data which
        # must be overrided by XCXCorrections class
        H_sp[:] = tempLDA
        return E

    
class KLIFunctional:
    def pass_stuff(self,
                   kpt_u, gd, finegd, interpolate,
                   restrict, poisson,
                   my_nuclei, ghat_nuclei,
                   nspins, nmyu, nbands,
                   kpt_comm, comm, nt_sg):
        self.kpt_u      = kpt_u      
        self.gd         = gd         
        self.finegd     = finegd     
        self.interpolate= interpolate
        self.restrict   = restrict   
        self.poisson    = poisson    
        self.my_nuclei  = my_nuclei  
        self.ghat_nuclei= ghat_nuclei
        self.nspins     = nspins
        self.nmyu       = nmyu       
        self.nbands     = nbands    
        self.kpt_comm   = kpt_comm
        self.comm       = comm
        self.nt_sg      = nt_sg

        self.fineintegrate = finegd.integrate

        self.rho_g      = finegd.zeros()
        self.rho_G      = gd.zeros()
        
        self.vsn_g      = finegd.zeros()
        self.vklin_g     = finegd.zeros()

        self.oldkli = finegd.zeros(2)
        self.first_iteration = True
        
        self.nt_G       = gd.zeros()
        self.nt_g       = finegd.zeros()
        self.vt_g       = finegd.zeros()

        print "Initializing KLI! PASS STUFF"

    def calculate_extra_setup_data(self, extra_xc_data, ae):
        print "NOT IMPLEMENTED"
        pass

    def calculate_kli_general(self, grid_allocator, fine_grid_allocator, interpolate, poisson_solver, restrict, integrate, scalar_mul, n_g, u_j, f_j):

        # Calculate total number of occupied states
        occupied = 0
        for f in f_j:
            if (f > 0):
                occupied += 1

        u_ix = grid_allocator(occupied)

        nXC_G = fine_grid_allocator() # The fine exchangedensity
        uXC_G = fine_grid_allocator() # The fine potential

        uXC_g = grid_allocator() # The coarse potential

        V_S = grid_allocator() # The Slater's averaged exchange potential
        vXC_G = grid_allocator() # The final potential

        # Calculate the |\Psi_i| times [13] to u_ix. Because of the numerical difficulties
        # we don't divide with \Psi_i here, since it cancels later. 
        for n1 in range(0, occupied):
            # Loop only over "upper diagonal" of indices i and k
            for n2 in range(n1, occupied):
                
                # Interpolate the exchange density to fine grid
                interpolate(u_j[n1]*u_j[n2], nXC_G)
                
                # Solve the poisson equation
                poisson_solver(uXC_G, -nXC_G)

                # Restrict the solution back to coarse grid                    
                restrict(uXC_G, uXC_g)

                # Use the solutions to calculate u_ix
                u_ix[n1] += scalar_mul(f_j[n2], u_j[n2] * uXC_g)
                    
                # Remember also the n2<n1 elements
                if (n1 != n2):
                    u_ix[n2] += scalar_mul(f_j[n1], u_j[n1] * uXC_g)

                   
        # Calculte u_bar and the Slaters single local excange potential
        u_bar = num.zeros((occupied), num.Float)

        for i in range(0, occupied):
            uXC_g = u_ix[i] * u_j[i]

            # Calculate the expection value of u_{x\sigma} respect to the orbitals [19]
            u_bar[i] = integrate(uXC_g)
            
            # Calculate the single exchange potential [37]. Division with density is done later.
            V_S += scalar_mul(f_j[i], uXC_g)
        
        if (occupied > 1):
            # Calculate the A matrix [65]. This uses the M-matrix in [62].
            # That is 
            A = num.zeros((occupied-1, occupied-1), num.Float)

            for i in range(0,occupied-1):
                for j in range(i,occupied-1):
                    term = f_j[i] * f_j[j] * integrate(u_j[i] * u_j[i] * u_j[j] * u_j[j] / n_g)
                    A[i,j] = -term/f_j[j]
                    A[j,i] = -term/f_j[i]

                    # Add Kroneckers delta
                    if (i == j):
                        A[i,j] += 1

            # Calculate the b vector
            # In the rhf of [65] the (V^S_{x\sigma j - \bar u_{j\sigma})
            b = num.zeros((occupied-1), num.Float)
            for i in range(0, occupied-1):
                b[i] = integrate(u_j[i]*u_j[i] * V_S / n_g) - u_bar[i];

            # Solve the linear equation [64] determinating the KLI-potential
            x = linalg.solve_linear_equations(A,b)

        #print "Ci:s ", x
        # Primed sum of [48]
        for i in range(0, occupied-1):
            vXC_G += scalar_mul(f_j[i]*x[i], u_j[i] * u_j[i])

        # First sum of [48]
        for i in range(0, occupied):
            vXC_G += scalar_mul(f_j[i], u_j[i] * u_ix[i])

        #print "vXC_G", vXC_G
        #print "n_g", n_g
        #print "vXC_G/n_g", vXC_G/n_g
        #Return the exchange energy
        return (num.dot(u_bar[0:occupied],f_j[0:occupied])/2, vXC_G/n_g)

    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, vXC):

        r = gd.r_g
        dr = gd.dr_g
        N = len(r)
        beta = 0.4 # XXX Grr.. Default value
        # Avoid division by zero with r this way. Suggestions to do this better are welcome. 
        r = r.copy()
        r[0] = r[1]

        # Create some helper functios to carry out the 1d-calculation
        # in calculate_1d_kli_general function. Grid interpolation and
        # restriction wont do anything in 1d-calculation. Everything
        # is expanded to spherical harmonics using Function1D.

        def grid_alloc(*args):

            if len(args) == 0:
                return Function1D()
            else:
                # How to allocate an array of Function1D object better in python???
                grid = []
                n, = args
                for i in range(0,n):
                    grid.append(Function1D())
                return grid
        
        def dummy(source, target):
            target.copyfrom(source)
        
        def poisson_solver(target, density):
            #print "Poisson solver", density
            target.copyfrom(density.solve_poisson(r,dr,beta, N))
        
        def integrate(u):
            return u.integrateRY(r, dr)

        def scalar_mul(scalar, function):
            temp = Function1D()
            temp.copyfrom(function)
            return temp.scalar_mul(scalar)

        # Expand the m-degeneracy of the wavefunctions
        u_lm = []
        f_lm = []
        occ = 0

        for k in range(0,u_j.shape[0]):
            for m in range(-l_j[k], l_j[k]+1):
                u_lm.append(Function1D(l_j[k], m, u_j[k]/r))
                # Fractional occupation number is f_j / (2l+1) /2
                f_lm.append(f_j[k]*0.5 / (2*l_j[k]+1))

        # Calculate the density
        n = Function1D()
        for n1, f in enumerate(f_lm):
            n += scalar_mul(f, u_lm[n1] * u_lm[n1])

        # Average the density spherically.
        n = Function1D(0,0, 1/sqrt(4*pi)*n.integrateY())

        Exc, result = self.calculate_kli_general(grid_alloc, grid_alloc,
                                                 dummy, poisson_solver, dummy,
                                                 integrate, scalar_mul,
                                                 n, u_lm, f_lm)

        # The spherically averaged potential is returned to solver
        vXC[:] = result.integrateY() / (4*pi)
        return Exc*2



    def calculate_one_spin(self, v_g, s):
        print "CALCULATING ONE SPIN" 

        small_number = 1e-200
        
        # Initialize method-attributes
        kpt = self.kpt_u[s]
        psit_nG = kpt.psit_nG     # Wave functions
        E = 0.0                   # Energy of eXact eXchange and kinetic energy
        f_n  = kpt.f_n.copy()      # Occupation number

        f_n *= self.nspins / 2.0
        occupied = int(sum(f_n))
        print "Occupied orbitals", f_n
        
        if occupied < 1e-3:
            return 0


        self.ubar_n     = num.zeros( occupied-1, num.Float)
        self.c_n        = num.zeros( occupied-1, num.Float)
        
        u = kpt.u               # Local spin/kpoint index
        hybrid = 1.

        self.vsn_g[:] = 0.0
        self.rho_G[:] = 0.0

        if (occupied > 1):
            A = num.zeros( (occupied-1, occupied-1), num.Float)

        # Calculate the density
        for n1 in range(self.nbands):
            f1 = f_n[n1]
            psit1_G = psit_nG[n1]    
            self.rho_G += f1 * psit1_G*psit1_G

        # Interpolate it to fine grid
        self.interpolate(self.rho_G, self.rho_g)
        
        # Determine pseudo-exchange
        for n1 in range(self.nbands):
            psit1_G = psit_nG[n1]      
            f1 = f_n[n1]
            if f1 > 1e-3:
                for n2 in range(n1, self.nbands):
                    psit2_G = psit_nG[n2]
                    f2 = f_n[n2]
                    if f2 > 1e-3:
                        dc = 1 + (n1 != n2) # double count factor

                        # Determine current exchange density ...
                        self.nt_G[:] = psit1_G * psit2_G

                        # and interpolate to the fine grid:
                        self.interpolate(self.nt_G, self.nt_g)

                        if (n1 < occupied-1):
                            if (n2 < occupied-1):
                                A[n1, n2] = -self.finegd.integrate(self.nt_g **2 / (self.rho_g + small_number))
                                A[n2, n1] = A[n1, n2]
                                if (n1 == n2):
                                    A[n1,n1] += 1

                        # Determine the compensation charges for each nucleus:
                        for nucleus in self.ghat_nuclei:
                            if nucleus.in_this_domain:
                                # Generate density matrix
                                P1_i = nucleus.P_uni[u, n1]
                                P2_i = nucleus.P_uni[u, n2]
                                D_ii = num.outerproduct(P1_i, P2_i)
                                D_p  = pack(D_ii, tolerance=1e3)#python func! move to C

                                # Determine compensation charge coefficients:
                                Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                                print "At kli:", Q_L
                            else:
                                Q_L = None

                            # Add compensation charges to exchange density:
                            nucleus.ghat_L.add(self.nt_g, Q_L, communicate=True)

                        # Determine total charge of exchange density:
                        Z = float(n1 == n2)

                        # Determine exchange potential:
                        print "Statring poisson... this is slooooow"
                        npoisson = self.poisson.solve(self.vt_g, -self.nt_g, eps = 1e-12, charge=-Z) # Removed zero initial
                        print "Poisson iterations", npoisson
                        print "Ending poisson..."

                        # Determine the projection within each nucleus
                        for nucleus in self.ghat_nuclei:
                            if nucleus.in_this_domain:
                                coeff = num.zeros((nucleus.setup.lmax + 1)**2, num.Float)
                                nucleus.ghat_L.integrate(self.vt_g, coeff)
                                #nucleus.Vx_nnnlm[n1,n2] = coeff

                        self.vsn_g += self.vt_g * self.nt_g 

                        # Integrate the potential on fine and coarse grids
                        int_fine = self.fineintegrate(self.vt_g * self.nt_g)

                        if (n1 < occupied-1):
                            self.ubar_n[n1] = - dc * int_fine
                        
                        E += 0.5 * f1 * f2 * dc * hybrid * int_fine

        # Calculate the slater potential
        self.vsn_g /= self.rho_g + small_number

        #print "A-matrix", A
        
        # Calculate the coefficients over slater potential
        for n1 in range(occupied-1):
            psit1_G = psit_nG[n1]      
            f1 = f_n[n1]
    
            # Determine current exchange density ...
            self.nt_G[:] = psit1_G * psit2_G

            # and interpolate to the fine grid:
            self.interpolate(self.nt_G, self.nt_g)

            self.ubar_n[n1] += self.finegd.integrate(self.vsn_g * self.nt_g)


        self.vklin_g[:] = 0.0

        # Solve the linear equation [64] determinating the KLI-potential
        if occupied > 1:
            print A.shape
            print self.ubar_n.shape
            x = linalg.solve_linear_equations(A,self.ubar_n)

            for n1 in range(0, occupied-1):
                psit1_G = psit_nG[n1]      
                f1 = f_n[n1]
    
                # Determine current exchange density ...
                self.nt_G[:] = psit1_G * psit2_G

                # and interpolate to the fine grid:
                self.interpolate(self.nt_G, self.nt_g)
                self.vklin_g += f1 * x[n1] * self.nt_g

            print x
            self.vklin_g[:]  /= self.rho_g + small_number

        self.vklin_g     += self.vsn_g

        #pylab.plot(self.vklin_g[23,

        if self.first_iteration:
            v_g[:] += self.vklin_g
            self.first_iteration = False
        else:
            v_g[:] += self.vklin_g # (0.05 * self.vklin_g) + (0.95 * self.oldkli)


        self.oldkli[s, :] = self.vklin_g[:] 

        
        return E

    def calculate_spinpaired(self, e_g, n_g, v_g):
        #from gpaw.xc_functional import XCFunctional
        #my_xc = XCFunctional('LDA')
        #my_xc.calculate_spinpaired(e_g, n_g, v_g)

        E = 2*self.calculate_one_spin(v_g, 0)
        e_g[:] = E / len(e_g) / self.finegd.dv

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        print "NOT HERE"
        E = 0.0
        E += self.calculate_one_spin(va_g,0)
        E += self.calculate_one_spin(vb_g,1)
        e_g[:] = E / len(e_g) / self.finegd.dv


    
