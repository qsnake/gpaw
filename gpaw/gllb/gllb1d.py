import numpy as npy

from gpaw.gllb import SMALL_NUMBER

class GLLB1D:
    def __init__(self):

        # Stuff needed by setup-generator
        self.slater_xc1D = None
        self.v_xc1D = None
        self.v_g1D = None
        self.e_g1D = None
        print "Initializing...", self.v_g1D
        
    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j,
                                             v_xc, njcore=None, density=None):
        """Used by setup generator to calculate the one dimensional potential

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        gd          Radial grid descriptor
        u_j         The wave functions
        f_j         The occupation numbers
        e_j         The eigenvalues
        l_j         The angular momentum quantum numbers
        v_xc        V_{xc} is added to this array.
        nj_core     If njcore is set, only response part will be returned for
                    wave functions u_j[:nj_core]
        density     If density is supplied, it overrides the density
                    calculated from orbitals.
                    This is used is setup-generation.
        =========== ==========================================================
        """

        # Construct the density if required
        if density == None:
            n_g = self.construct_density1D(gd, u_j, f_j)
        else:
            n_g = density

        # Construct the slater potential if required
        if njcore == None:
            # Get the slater potential multiplied by density
            Exc = self.get_slater1D(gd, n_g, u_j, f_j, l_j, v_xc)
            # Add response from all the orbitals
            imax = len(f_j)
        else:
            # Only response part of core orbitals is desired
            v_xc[:] = 0.0
            # Add the potential only from core orbitals
            imax = njcore
            Exc = 0

        # Get the response weigths
        w_j = self.get_response_weights1D(u_j, f_j, e_j)

        # Add the response multiplied with density to potential
        v_xc[:] += self.construct_density1D(gd, u_j[:imax], [f*w for f,w in zip(f_j[:imax] , w_j[:imax])])

        if njcore == None:
            # Divide with the density, beware division by zero
            v_xc[1:] /= n_g[1:] + SMALL_NUMBER
            
            # Do we have already XCRadialGrid object for v_xc
            if self.v_xc is not None:
                if self.v_xc1D == None:
                    from gpaw.xc_functional import XCFunctional, XCRadialGrid
                    self.v_xc1D = XCRadialGrid(self.v_xc, gd)
                    
                self.v_g[:] = 0.0
                self.e_g[:] = 0.0
        
                # Calculate the local v
                self.v_xc1D.get_energy_and_potential_spinpaired(n_g, self.v_g, e_g=self.e_g)
                v_xc[:] += self.v_g

                # Calculate the exchange energy
                Exc += npy.dot(self.e_g, gd.dv_g)

        # Fix the r=0 value
        v_xc[0] = v_xc[1]

        return Exc

    def construct_density1D(self, gd, u_j, f_j):
        """
        Creates one dimensional density from specified wave functions and occupations.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        gd          Radial grid descriptor
        u_j         The wave functions
        f_j         The occupation numbers
        =========== ==========================================================
        """

        n_g = npy.dot(f_j, u_j**2)
        n_g[1:] /=  4 * npy.pi * gd.r_g[1:]**2
        n_g[0] = n_g[1]
        return n_g

    def get_slater1D(self, gd, n_g, u_j, f_j, l_j, vrho_xc, vbar=False):
        """Return approximate exchange energy.

        Used by get_non_local_energy_and_potential1D to calculate an
        approximation to 1D-Slater potential. Returns the exchange
        energy. 

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        gd          Radial grid descriptor
        n_g         The density
        u_j         The 1D-wavefunctions
        f_j         Occupation numbers
        l_j         The angular momentum numbers
        vrho_xc     The slater part multiplied by density is added to this
                    array.
        v_bar       hmmm
        =========== ==========================================================
        """
        # Create temporary arrays only once
        if self.v_g1D == None:
            self.v_g = n_g.copy()

        if self.e_g1D == None:
            self.e_g = n_g.copy()

        # Do we have already XCRadialGrid object, if not, create one
        if self.slater_xc1D == None:
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.slater_xc1D = XCRadialGrid(self.slater_xc, gd)

        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        
        # Calculate B88-energy density
        self.slater_xc1D.get_energy_and_potential_spinpaired(n_g, self.v_g, e_g=self.e_g)

        # Calculate the exchange energy
        Exc = npy.dot(self.e_g, gd.dv_g)

        # The Slater potential is approximated by 2*epsilon / rho
        vrho_xc[:] += 2 * self.e_g

        return Exc

    def get_response_weights1D(self,  u_j, f_j, e_j):
        """
          Calculates the weights for response part of GLLB functional.

          =========== ==========================================================
          Parameters:
          =========== ==========================================================
          u_j         The 1D-wave functions
          f_j         The occupation numbers
          e_j         Eigenvalues
          =========== ==========================================================
        """
        reference_level = self.find_reference_level1D(f_j, e_j)
        w_j = [ self.gllb_weight(e, reference_level) for e in e_j ]
        return w_j

    def gllb_weight(self, epsilon, reference_level):
        """
        Calculates the weight for GLLB functional.
        The parameter K_G is adjusted such that the correct result is obtained for
        exchange energy of non-interacting electron gas.
        
        All orbitals closer than 1e-5 Ha to fermi level are consider the
        give zero response. This is to improve convergence of systems with
        degenerate orbitals.
        
        =============== ==========================================================
        Parameters:
        =============== ==========================================================
        epsilon         The eigenvalue of current orbital
        reference_level The fermi-level of the system
        =============== ==========================================================
        """

        if (epsilon +1e-5 > reference_level):
            return 0

        return self.K_G * npy.sqrt(reference_level-epsilon)

    def find_reference_level1D(self, f_j, e_j, lumo=False):
        """Finds the reference level from occupations and eigenvalue energies.
    
        Uses tolerance 1e-3 for occupied orbital.

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        f_j         The occupations list
        e_j         The eigenvalues list
        lumo        If lumo==True, find LUMO energy instead of HOMO energy.
        =========== ==========================================================
        """
    
        if lumo:
            lumo_level = 1000
            for f,e in zip(f_j, e_j):
                if f < 1e-3:
                    if lumo_level > e:
                        lumo_level = e
            return lumo_level

        homo_level = -1000
        for f,e in zip(f_j, e_j):
            if f > 1e-3:
                if homo_level < e:
                    homo_level = e
        return homo_level

    # input:  ae : AllElectron object.
    # output: extra_xc_data : dictionary. A Dictionary with pair ('name', radial grid)
    def calculate_extra_setup_data(self, extra_xc_data, ae):
        """
        For GLLB-functional one needs the response part of core orbitals to be stored in setup file,
        which is calculated in this section.

        ============= ==========================================================
        Parameters:
        ============= ==========================================================
        extra_xc_data Input: an empty dictionary
                      Output: dictionary with core_response-keyword containing data
        ae            All electron object containing all important data for calculating the core response.
        ============= ==========================================================

        """

        # Allocate new array for core_response
        N = len(ae.rgd.r_g)
        v_xc = npy.zeros(N)

        # Calculate the response part using wavefunctions, eigenvalues etc. from AllElectron calculator
        self.get_non_local_energy_and_potential1D(ae.rgd, ae.u_j, ae.f_j, ae.e_j, ae.l_j, v_xc,
                                                  njcore = ae.njcore)

        extra_xc_data['core_response'] = v_xc.copy()

        #print "Core response looks like", v_xc / (ae.n + SMALL_NUMBER)

        if self.relaxed_core_response:

            for nc in range(0, ae.njcore):
                # Add the response multiplied with density to potential
                orbital_density = self.construct_density1D(ae.rgd, ae.u_j[nc], ae.f_j[nc])
                extra_xc_data['core_orbital_density_'+str(nc)] = orbital_density
                extra_xc_data['core_eigenvalue_'+str(nc)] = [ ae.e_j[nc] ]

            extra_xc_data['njcore'] = [ ae.njcore ]
        

