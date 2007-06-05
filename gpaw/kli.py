import Numeric as num
import LinearAlgebra as linalg
from gpaw.Function1D import Function1D
from math import sqrt, pi
from gpaw.utilities import hartree, packed_index, unpack, unpack2, pack, pack2, fac
from LinearAlgebra import inverse

# GLLB-Functional
# Gritsenko, Leeuwen, Lenthe, Baerends: Self-consistent approximation to the Kohn-Shan exchange potential
# Physical Review A, vol. 51, p. 1944, March 1995.
# GLLB-Functional is of the same form than KLI-Functional, but it 
# 1) approximates the numerator part of Slater-potential from 2*GGA-energy density
# 2) approximates the response part coefficients from eigenvalues.

class GLLBFunctional:
    def __init__(self):
        self.slater_part = None

        
    def pass_stuff(self, new_grid_method):
        pass
        #energy_density = new_grid_method()
        #dummy_potential = new_grid_metod()
        
    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, v_xc):

        if self.slater_part == None:
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.slater_part = XCRadialGrid(XCFunctional('revPBEx'), gd)
        
        N = len(gd.r_g)
        
        # Construct the density
        n_g = num.dot(f_j, num.where(abs(u_j) < 1e-160, 0, u_j)**2) / (4 * pi)
        n_g[1:] /= gd.r_g[1:]**2
        n_g[0] = n_g[1]

        #print n_g
        e_g = num.zeros(N, num.Float)
        v_g = num.zeros(N, num.Float)
        
        self.slater_part.get_energy_density_spinpaired(n_g, v_g, e_g)
        #print "e_g", e_g
        v_xc[:] += 2 * e_g / (n_g + 1e-5)

        # Find fermi-level
        homo = len(f_j)-1
        mu = e_j[homo]

        for i in range(0, homo+1):
            # Construct the orbital-density
            nn_g = f_j[i] * num.where(abs(u_j[i]) < 1e-160, 0, u_j[i])**2 / (4 * pi) #grr.. Hack for numeric
            nn_g[1:] /= gd.r_g[1:]**2
            nn_g[0] = n_g[1]
            #print e_j[i]
            #print mu
            sqrt(mu-e_j[i])
            v_xc[:] += 0.382 * sqrt( mu - e_j[i]) * (nn_g / (n_g + 1e-20))

        return 0 
            
    def calculate_spinpaired(self, e_g, n_g, v_g):
        pass
        #Raise NotImplemented
        #self.slater_part.calculate_spinpaired(energy_density, n_g, dummy_potential)
        #v_g += 2*energy_density / (n_g + 1e-20)
        #e_g[:] = 0.0
        
    
    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        pass
        #Raise NotImplemented
        #self.slater_part.calculate_spinpaired(energy_density, na_g / 2, dummy_potential)
        #va_g += 2*energy_density / (na_g + 1e-20)
        #self.slater_part.calculate_spinpaired(energy_density, nb_g / 2, dummy_potential)
        #vb_g += 2*energy_density / (nb_g + 1e-20)
        #e_g[:] = 0.0

class Dummy:
    pass

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

        self.xc = Dummy()
        self.xc.xcfunc = Dummy()
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
        return (num.vdot(u_bar[0:occupied],f_j[0:occupied])/2, vXC_G/n_g)



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


    
