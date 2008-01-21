import numpy as npy
from gpaw.Function1D import Function1D
from math import sqrt, pi
from gpaw.utilities import hartree, packed_index, unpack, unpack2, pack, pack2, fac
from LinearAlgebra import inverse
from gpaw.operators import Gradient

# For XCCorrections
from numpy import dot as dot3
from gpaw.gaunt import gaunt
from gpaw.spherical_harmonics import YL
from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities.complex import cc, real

# load points and weights for the angular integration
from gpaw.sphere import Y_nL, points, weights

SLATER_FUNCTIONAL = "X_B88-None"
SMALL_NUMBER = 1e-8
K_G = 0.382106112167171

class NonLocalFunctionalDesc:
    def __init__(self, rho, grad, wfs, eigs):
        self.rho = rho
        self.grad = grad
        self.wfs = wfs
        self.eigs = eigs

    def needs_density(self):
        return self.rho

    def needs_gradient(self):
        return self.grad

    def needs_wavefunctions(self):
        return self.wfs

    def needs_eigenvalues(self):
        return self.eigs

class NonLocalFunctional:
    """
     Non-local functional is a superclass for many different types of functionals. It will give
     it 's subclasses ability to use quanities such as density, gradient, wavefunctions and eigenvalues.

     Subclasses will have to override methods get_functional_desc(), which will return
     a NonLocalFunctionalDesc object corresponding to needs of the functional, and
     method calculate_non_local which will do the actual calculation of KS-potential
     using the optional density, gradient, wavefunctions, eigenvalues supplied in info-dictionary.

     NonLocalFunctional handles only 3D-case, so 1D-setup generation must be handled by subclasses.
     
    """

    def __init__(self):
        self.initialization_ready = False
        self.mixing = 0.3

    def pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation):
        """
        Important quanities is supplied to non-local functional using this method.
    
        Called from xc_functional::set_non_local_things method
        All the necessary classes and methods are passed through this method
        Not used in 1D-calculations.
        """
        
        self.kpt_u = kpt_u
        self.finegd = finegd
        self.interpolate = interpolate
        self.nspins = nspins
        self.nuclei = nuclei
        self.occupation = occupation
        self.gd = gd
        self.finegd = finegd

        # Get the description for this functional
        self.desc = self.get_functional_desc()
        
        # Temporary array for density mixing
        self.old_vt_g = None
        self.new_v_g = finegd.zeros()
        

        if self.desc.needs_gradient():
            # Allocate stuff needed for gradient
            self.ddr = [Gradient(finegd, c).apply for c in range(3)]
            self.dndr_cg = finegd.empty(3)
            self.a2_g = finegd.empty()
    
    def get_functional_desc(self):
        """Returns a NonLocalFunctionalDesc instance specifiying the dependence of this functional.
           on different parameters.

           Currently, there is 4 possibilities:
              - density, gradient, wavefunctions and eigenvalues
              
           To be implemented in subclasses of ResponseFunctional

        """
       
        raise "ResponseFunctional::get_functional_info must be overrided"

    def calculate_non_local(self, info, v_g, e_g):
        """Add the KS-Exchange potential to v_g and supply energy density using data
           supplied in dictionary info.

           To be implemented in subclasses of ResponseFunctional

           =========== ========================
           Parameters:
           =========== ========================
           info        A dictiorany, see below
           v_g         The Kohn-Sham potential.
           e_g         The energy density
           =========== ========================
           
           info is a dictiorary with following content:

           =========== =================================================
           Key:        Value:
           =========== =================================================
           dtype    For example float, if the orbitals are real
           gd          The grid descriptor object for coarse grid
           finegd      The grid descriptor object for fine grid
           n_g         Numeric array for density, supplied if
                       needs_density() true
           psit_nG     A _python_ list containing the wavefunctions,
                       if needs_wavefunctions() true
           f_n         A _python_ list containing the occupations,
                       if needs_wavefunctions() true
           a2_g        Numeric array for gradient, if needs_gradient()
                       true
           eps_n       A _python_ list of eigenvalues, if
                       needs_eigenvalues() true
           =========== =================================================
           """
        raise "ResponseFunctional::calculate must be overrided"


    def calculate_spinpaired(self, e_g, n_g, v_g):
        """Calculates the KS-exchange potential and adds it to v_g. Supplies also
           the energy density. This methods cCollects the required info
           (see comments for calculate_non_local) and calls calculate_non_local with this info.

           =========== =========================
           Parameters:
           =========== =========================
           e_g         The energy density
           n_g         The electron density
           v_g         The Kohn-Sham potential.
           =========== =========================

        """

        info = {}
        # Supply parameters for calculate non-local
        info['gd'] = self.gd
        info['finegd'] = self.finegd

        # Supply density if it is required
        if self.desc.needs_density():
            info['n_g'] = n_g

        if self.desc.needs_gradient():
            # Calculate the gradient
            for c in range(3):
                self.ddr[c](n_g, self.dndr_cg[c])
            self.a2_g[:] = npy.sum(self.dndr_cg**2)

            info['a2_g'] = self.a2_g

        # Supply the eigenvalues if required
        if self.desc.needs_eigenvalues():
            info['eps_n'] = []
            for kpt in self.kpt_u:
                for e in kpt.eps_n:
                    info['eps_n'].append(e)

        # Supply the wavefunctions if required
        if self.desc.needs_wavefunctions():
            psit_nG = []
            f_n = []
            for kpt in self.kpt_u:
                for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                    psit_nG.append(psit_G)
                    f_n.append(f)
                    
            info['psit_nG'] = psit_nG
            info['f_n'] = f_n
            info['dtype'] = self.kpt_u[0].dtype

        # Calculate the exchange potential to temporary array
        self.calculate_non_local(info, v_g, e_g)

        #Perform the potential mixing and add the resulting potential to v_g
        if self.old_vt_g == None:
            self.old_vt_g = v_g.copy()
        v_g[:] = self.mixing * v_g[:] + (1.0 - self.mixing) * self.old_vt_g
        self.old_vt_g[:] = v_g[:]

class GLLBFunctional(NonLocalFunctional):
    """
        This class calculates the energy and potential determined by GLLB-Functional [1]. This functional:
           1) approximates the numerator part of Slater-potential from 2*GGA-energy density. This implementation
           follows the orginal authors and uses the Becke88-functional.
           2) approximates the response part coefficients from eigenvalues, given correct result for non-interacting
           electron gas.

        [1] Gritsenko, Leeuwen, Lenthe, Baerends: Self-consistent approximation to the Kohn-Shan exchange potential
        Physical Review A, vol. 51, p. 1944, March 1995.
        GLLB-Functional is of the same form than KLI-Functional, but it 
        
    """

    def __init__(self):
        """
        Initialize the GLLBFunctional. Some variables are initialized with none. All of these
        variables are not needed, and therefore they are only initilized when they are needed.
        """
        
        NonLocalFunctional.__init__(self)
    
        self.slater_part = None
        self.initialization_ready = False
        self.fermi_level = -1000
        self.v_g1D = None
        self.e_g1D = None
        self.slater_part1D = None
        self.slater_part = None

    def pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation):
        NonLocalFunctional.pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation)

        # Temporary arrays needed for GLLB
        self.tempvxc_g = finegd.zeros()
        self.tempe_g = finegd.zeros()
        self.vt_G = gd.zeros()         # Temporary array for coarse potential
        self.vt_g = finegd.zeros()     # Temporary array for fine potential
        self.nt_G = gd.zeros()         # Temporary array for coarse density


    def get_functional_desc(self):
        """
        Retruns info for GLLBFunctional. The GLLB-functional needs density, gradient, wavefunctions
        and eigenvalues.
        
        """
        return NonLocalFunctionalDesc(True, True, True, True)

#################################################################################
#                                                                               #
# Implementation of 1D-GLLB begins                                              #
#                                                                               #
################################################################################# 
        
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

        n_g = npy.dot(f_j, npy.where(abs(u_j) < 1e-160, 0, u_j)**2)
        n_g[1:] /=  4 * pi * gd.r_g[1:]**2
        n_g[0] = n_g[1]
        return n_g

    def get_slater1D(self, gd, n_g, vrho_xc):
        """Headline ...
        
        Used by get_non_local_energy_and_potential1D to calculate an
        approximation to 1D-Slater potential. Returns the exchange
        energy.
          
        =========== =======================================================
        Parameters:
        =========== =======================================================
        gd          Radial grid descriptor
        n_g         The density
        vrho_xc     The slater part multiplied by density is stored here.
        =========== =======================================================
        """
        # Create temporary arrays only once
        if self.v_g1D == None:
            v_g = n_g.copy()
            v_g[:] = 0.0

        if self.e_g1D == None:
            e_g = n_g.copy()
            e_g[:] = 0.0

        # Do we have already XCRadialGrid object, if not, create one
        if self.slater_part1D == None:
            from gpaw.xc_functional import XCFunctional, XCRadialGrid
            self.slater_part1D = XCRadialGrid(XCFunctional(SLATER_FUNCTIONAL, 1), gd)
       
        # Calculate B88-energy density
        self.slater_part1D.get_energy_and_potential_spinpaired(n_g, v_g, e_g=e_g)

        Exc = npy.dot(e_g, gd.dv_g)

        # The Slater potential is approximated by 2*epsilon / rho
        vrho_xc[:] += 2 * e_g

        return Exc

    def gllb_weight(self, epsilon, fermi_level):
        """
        Calculates the weight for GLLB functional.
        The parameter K_G is adjusted such that the correct result is obtained for
        non-interacting electron gas.

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        epsilon     The eigenvalue of current orbital
        fermi_level The fermi-level of the system
        =========== ==========================================================
        """
        
        if (epsilon +1e-3 > fermi_level):
            return 0
        return K_G * sqrt(fermi_level-epsilon)


    def find_fermi_level(self, f_j, e_j):
        """
            Finds the fermilevel from occupations and eigenvalue energies.
            Uses tolerance 1e-5 for occupied orbital.
            =========== ==========================================================
            Parameters:
            =========== ==========================================================
            f_j         The occupations list
            e_j         The eigenvalues list
            =========== ==========================================================

        """
        
        fermi_level = -1000
        for f,e in zip(f_j, e_j):
            if f > 1e-5:
                if fermi_level < e:
                    fermi_level = e
        return fermi_level
    
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
        fermi_level = self.find_fermi_level(f_j, e_j)
        w_j = [ self.gllb_weight(e, fermi_level) for e in e_j ]
        return w_j

    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, v_xc, njcore=None, density=None):
        """
          Used by setup generator to calculate the one dimensional potential
        
          =========== ==========================================================
          Parameters:
          =========== ==========================================================
          gd          Radial grid descriptor
          u_j         The wave functions
          f_j         The occupation numbers
          e_j         The eigenvalues
          l_j         The angular momentum quantum numbers
          v_xc        V_{xc} is stored here.
          nj_core     If njcore is set, only response part will be returned for wave functions u_j[:nj_core]
          density     If density is supplied, it overrides the density calculated from orbitals.
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
            Exc = self.get_slater1D(gd, n_g, v_xc)
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

        # Divide with the density, beware division by zero
        v_xc[1:] /= n_g[1:] + SMALL_NUMBER

        # Fix the r=0 value
        v_xc[0] = v_xc[1]

        return Exc

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

        extra_xc_data['core_response'] = v_xc
        
#################################################################################
#                                                                               #
# Implementation of 3D-GLLB begins                                              #
#                                                                               #
################################################################################# 

    def calculate_non_local(self, info, v_g, e_g):
        """
        Calculate the GLLB-energy and potentail. The GLLB-energy is taken from B88-exchange energy,
        and the potential is combined from approximative Slater part calculated B88-exchange energy
        density, and the response part involving the wave functions and the orbital eigenvalues.
        
        
        ============= ==========================================================
        Parameters:
        ============= ==========================================================
        v_g           The GLLB-potential is added to this supplied potential
        e_g           The GLLB-energy density (is the same as e_g)
        ============= ==========================================================
        
        """

        # Create the B88 functional for Slater part (only once per calculation)

        if self.slater_part == None:
            from gpaw.xc_functional import XCFunctional, XC3DGrid
            self.slater_part = XCFunctional(SLATER_FUNCTIONAL, self.nspins)
            self.initialization_ready = True


        # Calculate the slater potential. self.tempvxc_g and self.vt_g are used just for dummy
        # arrays and they are not used after calculation. Fix?
        self.slater_part.calculate_spinpaired(self.tempe_g,  info['n_g'], self.tempvxc_g,
                                              a2_g = info['a2_g'], deda2_g = self.vt_g)

        # Add it to the total potential
        v_g += 2*self.tempe_g / (info['n_g'] + SMALL_NUMBER)

        # Return the xc-energy correctly
        e_g[:] = self.tempe_g.ravel()

        # Find the fermi-level
        self.fermi_level = self.find_fermi_level(info['f_n'], info['eps_n'])
        # Use the coarse grid for response part
        # Calculate the coarse response multiplied with density and the coarse density
        # and to the division at the end of the loop.
        self.vt_G[:] = 0.0
        self.nt_G[:] = 0.0
        
        # For each orbital, add the response part
        for f, e, psit_G in zip(info['f_n'], info['eps_n'], info['psit_nG']):
            w = self.gllb_weight(e, self.fermi_level)
            if info['dtype'] is float:
                psit_G2 = psit_G**2
                axpy(f*w, psit_G2, self.vt_G) 
                axpy(f, psit_G2, self.nt_G)
            else:
                self.vt_G += f * w * (psit_G * npy.conjugate(psit_G)).real
                self.nt_G += f * (psit_G * npy.conjugate(psit_G)).real
            

        # After the coarse density (w/o compensation charges) and the numerator is calculated, do the division
        self.vt_G[:] /= self.nt_G[:] + SMALL_NUMBER

        self.vt_g[:] = 0.0 # Is this needed for interpolate?
        self.interpolate(self.vt_G, self.vt_g)
        
        # Add the fine-grid response part to total potential
        v_g[:] += self.vt_g 

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        print "GLLB calculate_spinpolarized not implemented"
        pass


        
#################################################################################
#                                                                               #
# Implementation of XCCorrections begins                                        #
# -This part is needs more attention                                            #
################################################################################# 


class DummyXC:
    def set_functional(self, xc):
        print "GLLB: DummyXC::set_functional(xc) with ", xc.xcname
        
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
            self.Y_yL = npy.array([[1.0 / sqrt(4.0 * pi)]])
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
        # This is the code from GGA-method of XCCorrections, but
        # it has lines involving GLLB. All lines which contain
        # comments are NOT from xc_corrections.py file:)

        self.nspins = 1 # XXXX SPINHACK
        
        # This method is called before initialization of motherxc in pass_stuff
        if self.motherxc.slater_part == None:
            print "GLLB: Not applying the PAW-corrections!"
            return 0 #Grr....

        #print "D_sp", D_sp

        nucleus = self.motherxc.nuclei[a] # Get the nucleus with index
        ni = nucleus.get_number_of_partial_waves() # Get the number of partial waves from nucleus
        np = ni * (ni + 1) // 2 # Number of items in packed density matrix
        
        Dn_ii = npy.zeros((ni, ni)) # Allocate space for unpacked atomic density matrix
        Dn_p = npy.zeros((np, np)) # Allocate space for packed atomic density matrix
 
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

            if self.motherxc.initialization_ready:
                # For each k-point
                for kpt in self.motherxc.kpt_u:
                    # Get the projection coefficients
                    P_ni = nucleus.P_uni[kpt.u]
                    # Create the coefficients
                    w_i = npy.zeros(kpt.eps_n.shape)
                    for i in range(len(w_i)):
                        w_i[i] = self.motherxc.gllb_weight(kpt.eps_n[i], self.motherxc.fermi_level)

                    w_i = w_i[:, npy.NewAxis] * kpt.f_n[:, npy.NewAxis] # Calculate the weights
                    # Calculate the 'density matrix' for numerator part of potential
                    Dn_ii = real(npy.dot(cc(npy.transpose(P_ni)),
                                         P_ni * w_i))
                
                    Dn_p = pack(Dn_ii) # Pack the unpacked densitymatrix

                    Dnn_Lq = dot3(self.B_Lqp, Dn_p) #Contract one nmln'm'l'
                    nn_Lg = npy.dot(Dnn_Lq, self.n_qg) # Contract nln'l'
                    nn = npy.dot(Y_L, nn_Lg) ### Contract L
            else:
                nn = 0.0

            # Add the Slater-part
            x_g = (2*e_g + nn) / (n_g + SMALL_NUMBER) * self.dv_g
            # Add the response from core
            x_g += self.core_response * self.dv_g

            # Calculate the slice
            dEdD_p += w * npy.dot(dot3(self.B_pqL, Y_L),
                                  npy.dot(self.n_qg, x_g))
            
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
            Et += w * npy.dot(e_g, self.dv_g)

            if self.motherxc.initialization_ready:
                #Dnn_Lq = dot3(self.B_Lqp, Dn_sp) #Contract one nmln'm'l'
                nn_Lg = npy.dot(Dnn_Lq, self.nt_qg) # Contract nln'l'
                nn = npy.dot(Y_L, nn_Lg) ### Contract L
            else:
                nn = 0.0
                
            x_g = (2*e_g + nn) / (n_g + SMALL_NUMBER) * self.dv_g
            
            dEdD_p -= w * npy.dot(dot3(self.B_pqL, Y_L),
                                  npy.dot(self.nt_qg, x_g))
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
            #E -= hybrid * npy.dot(D_p, self.X_p)
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
                    D_ii = npy.outer(P1_i, P2_i)
                    D_p  = pack(D_ii, tolerance=1e3)#python func! move to C

                    # Determine compensation charge coefficients:
                    Q_L = npy.dot(D_p, nucleus.setup.Delta_pL)

                    d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
                           for l in range(nucleus.setup.lmax + 1)]
                    g = nucleus.setup.alpha2**1.5 * npy.exp(-nucleus.setup.alpha2 * self.r**2)
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
        u_bar = npy.zeros((occupied))

        for i in range(0, occupied):
            uXC_g = u_ix[i] * u_j[i]

            # Calculate the expection value of u_{x\sigma} respect to the orbitals [19]
            u_bar[i] = integrate(uXC_g)
            
            # Calculate the single exchange potential [37]. Division with density is done later.
            V_S += scalar_mul(f_j[i], uXC_g)
        
        if (occupied > 1):
            # Calculate the A matrix [65]. This uses the M-matrix in [62].
            # That is 
            A = npy.zeros((occupied-1, occupied-1))

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
            b = npy.zeros((occupied-1))
            for i in range(0, occupied-1):
                b[i] = integrate(u_j[i]*u_j[i] * V_S / n_g) - u_bar[i];

            # Solve the linear equation [64] determinating the KLI-potential
            x = npy.linalg.solve_linear_equations(A,b)

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
        return (npy.dot(u_bar[0:occupied],f_j[0:occupied])/2, vXC_G/n_g)

    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, vXC, density=None):

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


        self.ubar_n     = npy.zeros( occupied-1)
        self.c_n        = npy.zeros( occupied-1)
        
        u = kpt.u               # Local spin/kpoint index
        hybrid = 1.

        self.vsn_g[:] = 0.0
        self.rho_G[:] = 0.0

        if (occupied > 1):
            A = npy.zeros( (occupied-1, occupied-1))

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
                                D_ii = npy.outer(P1_i, P2_i)
                                D_p  = pack(D_ii, tolerance=1e3)#python func! move to C

                                # Determine compensation charge coefficients:
                                Q_L = npy.dot(D_p, nucleus.setup.Delta_pL)
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
                                coeff = npy.zeros((nucleus.setup.lmax + 1)**2)
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
            x = npy.linalg.solve_linear_equations(A,self.ubar_n)

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


    
