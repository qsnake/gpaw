import Numeric as num
import LinearAlgebra as linalg
from gpaw.utilities import hartree, unpack, unpack2, pack, pack2

class KLIFunctional:
    """KLI functional
       --------------

       Based on article:
         Phys. Rev. A Vol 45 p.101
         Krieger, Li, Iafrate
         Construction and application of an accurate local spin-polarized
         Kohn-Sham potential with integer discontinuity: Exchange-only theory.

       All the equations are refered with [nn] refering to equation in this
       article.

       NOTE: This is the first beta version that works only with all-electron
       mode. There will be also an assertion that has to be disabled in order to
       get this working.
    """

    # This initializes the KLI-f xuxnctional
    # Called from xc_functional.set_non_local_things
    def pass_paw_object(self, paw):
        # Store the paw object for later use
        self.paw = paw

        # For mixing of the potential, allocate this array
        # TODO: Is this correct number for max value of k-point index u
        self.last_vxc = self.paw.gd.zeros(self.paw.nkpts * self.paw.nspins)

        # Count the iterations
        self.iteration = 0

        # The mixing coefficient of potential.
        # I think that in KLI you cannot mix the density.
        self.mixing = 0.40

        self.store_N2poissons = False        
        self.solutions = None
        if self.store_N2poissons:
            print "Warning: Do you really want to store order N^2 poisson solutions?"

        self.E_x = 0
        
    def get_non_local_energy(self):
        return self.E_x

    def get_extra_kinetic_energy(self):
        #print "Extra kinetic", -2*self.E_x
        return -2*self.E_x

    def calculate_spinpaired(self, e_g, n_g, v_g):
        e_g[:] = 0.0    

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        e_g[:] = 0.0    


    def calculate_energy(self, kpt, Htpsit_nG, H_nn):

        deg = 2.0 / self.paw.nspins
        f_n = kpt.f_n / deg       

        wavefunctions = kpt.psit_nG

        spin = kpt.u

        # NOTE: Calculating hydrogen atom: There is no call with spin-index 0, only with 1... so this won't work.
        if (spin == 0):
            self.iteration += 1
        
        # Calculate the KLI potential again only every 5th iteration
        #print "iteration", self.iteration
        if (not (self.iteration % 5 == 4)):
            # Apply the potential to HPsi
            #print "Applying KLI"
            for n1, psi in enumerate(wavefunctions):
                Htpsit_nG[n1] += self.last_vxc[spin] * psi
            
            return

        print "Calculating KLI"
        # At first spin index, zero the energy left from previous iterations
        # This isn't very stylish, but works...
        if (spin == 0):
            self.E_x = 0
        
        if (self.store_N2poissons):
            if (self.solutions == None):
                self.solutions = self.paw.finegd.zeros(total*(total+1)/2)
                print "Created array of size ", self.solutions.shape, " to store poisson solutions."
        
        # Calculate total number of occupied states
        total = 0
        for f in f_n:
            if (f > 0):
                total += 1

        if (total == 0):
            return
            
        # Create some temporary arrays
        # TODO: Allocate these somewhere in outer loop
        u_bar = num.zeros(total, num.Float);
        fine_density = self.paw.finegd.zeros()
        fine_xc_potential = self.paw.finegd.zeros()
        xc_potential = self.paw.gd.zeros()
        temp = self.paw.gd.zeros()
        V_slater = self.paw.gd.zeros();

        # Create array for poisson solutions
        u_ix = self.paw.gd.zeros(total)
            
        # Calculate the |\Psi_i| times [13] to u_ix. Because of the numerical difficulties
        # we don't divide with \Psi_i here, since it cancels later. 

        solutionindex = 0
            
        for n1 in range(0, total):
            u1 = wavefunctions[n1]
            # Loop only over "upper diagonal" of indices i and k
            for n2 in range(n1, total):
                u2 = wavefunctions[n2]
                
                # Interpolate the exchange density to fine grid
                self.paw.density.interpolate(u1 * u2, fine_density)
                    

                # Determine the compensation charges for each nucleus:
                for nucleus in self.paw.ghat_nuclei:
                    if nucleus.in_this_domain:
                        # Generate density matrix
                        P1_i = nucleus.P_uni[spin, n1]
                        P2_i = nucleus.P_uni[spin, n2]
                        D_ii = num.outerproduct(P1_i, P2_i)
                        D_p  = pack(D_ii, tolerance=1e3)
 
                        # Determine compensation charge coefficients:
                        Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    else:
                        Q_L = None

                    # Add compensation charges to exchange density:
                    nucleus.ghat_L.add(fine_density, Q_L, communicate=True)
                    
                # Todo: Find out is fine grid really needed here!
                # Currently the rest of the work is done only on coarse grid.

                # Since the wavefunctions are orthonormal the total exchange charge is
                # zero unless the orbitals are the same.
                Z = (n1 == n2)

                # Solve the poisson equation
                #-----------------------------
                
                # Get the initial guess
                if (self.store_N2poissons):
                    fine_xc_potential = self.solutions[solutionindex]
                else:
                    # Poisson solver crashes sometimes, if the potential is not zeroed first
                    fine_xc_potential[:] = 0
                    
                self.paw.hamiltonian.poisson.solve(fine_xc_potential, -fine_density, charge = -Z)
                
                if (self.store_N2poissons):
                    self.solutions[solutionindex] = fine_xc_potential

                solutionindex += 1
                #print solutionindex, " / ", (total+1)*total/2, " poissons solved."

                # Restrict the solution back to coarse grid                    
                self.paw.hamiltonian.restrict(fine_xc_potential, xc_potential)

                # Use the solutions to calculate u_ix
                u_ix[n1] += f_n[n2] * u2 * xc_potential
                    
                # Remember also the n2<n1 elements
                if (n1 != n2):
                    u_ix[n2] += f_n[n1] * u1 * xc_potential

                   
        # Calculte u_bar and the Slaters single local excange potential

        for i in range(0, total):
            xc_potential = u_ix[i] * wavefunctions[i]

            # Calculate the expection value of u_{x\sigma} respect to the orbitals [19]
            u_bar[i] = self.paw.gd.integrate(xc_potential)
            # Calculate the single exchange potential [37]. Division with density is done later.
            V_slater += f_n[i] * xc_potential
        
        print "u_bar :", u_bar

        # Is paw.density.get_density_array() the correct density?
        # I suppose not, if the density is mixed, not the potential.
        # Lets calculate the density ourselves instead of using
        # coarse_density = self.paw.density.get_density_array() /2
        coarse_density = self.paw.gd.zeros();
        for n1, f in enumerate(f_n):
            coarse_density += f*wavefunctions[n1]**2

        # In some cases this is needed to avoid division by zero.
        # Here might be room for some improvement.
        coarse_density += 1e-200
            
        if (total > 1):
            # Calculate the A matrix [65]. This uses the M-matrix in [62].
            # That is 
            A = num.zeros((total-1, total-1), num.Float)

            for i in range(0,total-1):
                for j in range(i,total-1):
                    off_diag = self.paw.gd.integrate(f_n[i]*f_n[j]*wavefunctions[i]**2 *wavefunctions[j]**2 /coarse_density)
                    A[i,j] = -off_diag/f_n[j]
                    A[j,i] = -off_diag/f_n[i]
                    if (i == j):
                        A[i,j] = A[i,j] + 1;

            # Calculate the b vector
            # In the rhf of [65] the (V^S_{x\sigma j - \bar u_{j\sigma})
            b = num.zeros((total-1), num.Float)    
            for i in range(0, total-1):
                b[i] = self.paw.gd.integrate(wavefunctions[i]**2 * V_slater / coarse_density) - u_bar[i];

            # Solve the linear equation [64] determinating the KLI-potential
            x = linalg.solve_linear_equations(A,b)

            
        xc_potential = 0
            
        # Primed sum of [48]
        for i in range(0, total-1):
            xc_potential += wavefunctions[i]**2 * x[i] * f_n[i]

        # First sum of [48]
        for i in range(0, total):
            xc_potential += wavefunctions[i]*u_ix[i] * f_n[i];

        xc_potential /= coarse_density    

        # Do the mixing here
        if (self.iteration >1):
            self.last_vxc[spin] = xc_potential * self.mixing + self.last_vxc[spin] * (1 - self.mixing)
        else:
            self.last_vxc[spin] = xc_potential
            
        # Apply the potential to HPsi
        for n1, psi in enumerate(wavefunctions):
            Htpsit_nG[n1] += self.last_vxc[spin] * psi 
            
        # E_x seems to be half of the sum of u_bars weighted with occupations
        self.E_x += sum(u_bar[0:total]*f_n[0:total])/2 




    def calculate_1d_kli_potential(alpha, f_n, wavefunctions):

        u_ix = self.paw.gd.zeros(total)
            
        # Calculate the |\Psi_i| times [13] to u_ix. Because of the numerical difficulties
        # we don't divide with \Psi_i here, since it cancels later. 

        solutionindex = 0
            
        for n1 in range(0, total):
            u1 = wavefunctions[n1]
            # Loop only over "upper diagonal" of indices i and k
            for n2 in range(n1, total):
                u2 = wavefunctions[n2]
                
                # Interpolate the exchange density to fine grid
                self.paw.density.interpolate(u1 * u2, fine_density)
                    

                # Determine the compensation charges for each nucleus:
                for nucleus in self.paw.ghat_nuclei:
                    if nucleus.in_this_domain:
                        # Generate density matrix
                        P1_i = nucleus.P_uni[spin, n1]
                        P2_i = nucleus.P_uni[spin, n2]
                        D_ii = num.outerproduct(P1_i, P2_i)
                        D_p  = pack(D_ii, tolerance=1e3)
 
                        # Determine compensation charge coefficients:
                        Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
                    else:
                        Q_L = None

                    # Add compensation charges to exchange density:
                    nucleus.ghat_L.add(fine_density, Q_L, communicate=True)
                    
                # Todo: Find out is fine grid really needed here!
                # Currently the rest of the work is done only on coarse grid.

                # Since the wavefunctions are orthonormal the total exchange charge is
                # zero unless the orbitals are the same.
                Z = (n1 == n2)

                # Solve the poisson equation
                #-----------------------------
                
                # Get the initial guess
                if (self.store_N2poissons):
                    fine_xc_potential = self.solutions[solutionindex]
                else:
                    # Poisson solver crashes sometimes, if the potential is not zeroed first
                    fine_xc_potential[:] = 0
                    
                self.paw.hamiltonian.poisson.solve(fine_xc_potential, -fine_density, charge = -Z)
                
                if (self.store_N2poissons):
                    self.solutions[solutionindex] = fine_xc_potential

                solutionindex += 1
                #print solutionindex, " / ", (total+1)*total/2, " poissons solved."

                # Restrict the solution back to coarse grid                    
                self.paw.hamiltonian.restrict(fine_xc_potential, xc_potential)

                # Use the solutions to calculate u_ix
                u_ix[n1] += f_n[n2] * u2 * xc_potential
                    
                # Remember also the n2<n1 elements
                if (n1 != n2):
                    u_ix[n2] += f_n[n1] * u1 * xc_potential

                   
        # Calculte u_bar and the Slaters single local excange potential

        for i in range(0, total):
            xc_potential = u_ix[i] * wavefunctions[i]

            # Calculate the expection value of u_{x\sigma} respect to the orbitals [19]
            u_bar[i] = self.paw.gd.integrate(xc_potential)
            # Calculate the single exchange potential [37]. Division with density is done later.
            V_slater += f_n[i] * xc_potential
        
        print "u_bar :", u_bar

        # Is paw.density.get_density_array() the correct density?
        # I suppose not, if the density is mixed, not the potential.
        # Lets calculate the density ourselves instead of using
        # coarse_density = self.paw.density.get_density_array() /2
        coarse_density = self.paw.gd.zeros();
        for n1, f in enumerate(f_n):
            coarse_density += f*wavefunctions[n1]**2

        # In some cases this is needed to avoid division by zero.
        # Here might be room for some improvement.
        coarse_density += 1e-200
            
        if (total > 1):
            # Calculate the A matrix [65]. This uses the M-matrix in [62].
            # That is 
            A = num.zeros((total-1, total-1), num.Float)

            for i in range(0,total-1):
                for j in range(i,total-1):
                    off_diag = self.paw.gd.integrate(f_n[i]*f_n[j]*wavefunctions[i]**2 *wavefunctions[j]**2 /coarse_density)
                    A[i,j] = -off_diag/f_n[j]
                    A[j,i] = -off_diag/f_n[i]
                    if (i == j):
                        A[i,j] = A[i,j] + 1;

            # Calculate the b vector
            # In the rhf of [65] the (V^S_{x\sigma j - \bar u_{j\sigma})
            b = num.zeros((total-1), num.Float)    
            for i in range(0, total-1):
                b[i] = self.paw.gd.integrate(wavefunctions[i]**2 * V_slater / coarse_density) - u_bar[i];

            # Solve the linear equation [64] determinating the KLI-potential
            x = linalg.solve_linear_equations(A,b)

            
        xc_potential = 0
            
        # Primed sum of [48]
        for i in range(0, total-1):
            xc_potential += wavefunctions[i]**2 * x[i] * f_n[i]

        # First sum of [48]
        for i in range(0, total):
            xc_potential += wavefunctions[i]*u_ix[i] * f_n[i];

        xc_potential /= coarse_density    

        # Do the mixing here
        if (self.iteration >1):
            self.last_vxc[spin] = xc_potential * self.mixing + self.last_vxc[spin] * (1 - self.mixing)
        else:
            self.last_vxc[spin] = xc_potential
            
        # Apply the potential to HPsi
        for n1, psi in enumerate(wavefunctions):
            Htpsit_nG[n1] += self.last_vxc[spin] * psi 
            
        # E_x seems to be half of the sum of u_bars weighted with occupations
        self.E_x += sum(u_bar[0:total]*f_n[0:total])/2 

