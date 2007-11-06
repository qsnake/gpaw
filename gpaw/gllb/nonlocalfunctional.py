from gpaw.operators import Gradient
import Numeric as num
	
#################################################################################
#                                                                               #
# Implementation of NonLocalFunctionalDesc begins                               #
#                                                                               #
#################################################################################

class NonLocalFunctionalDesc:
    """Contains description of items, which a non-local density functional needs.
       When an exchange and correclation potential is requested from NonLocalFunctional,
       it will use the NonLocalFunctionalDesc object to determine which objects 
       calculate_non_local method needs for this particular functional.

       For example, If NonLocalFunctionalDesc.needs_gradient returns False,
       NonLocalFunctional will NOT calculate gradient and supply it to calculate_non_local method.
 
       NonLocalFunctional.generate_info is currently able to generate following items:
         -The spin-density
         -The gradient
         -A list of wave functions
         -A list of eigenvalues

    """

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

#################################################################################
#                                                                               #
# Implementation of NonLocalFunctional begins                                   #
#                                                                               #
#################################################################################

class NonLocalFunctional:
    """
     Non-local functional is a superclass for many different types of functionals. It will give
     it's subclasses ability to use quanities such as density, gradient, wavefunctions and eigenvalues.

     Subclasses will have to override methods get_functional_desc(), which will return
     a NonLocalFunctionalDesc object corresponding to needs of the functional, and
     method calculate_non_local which will do the actual calculation of KS-potential
     using the optional density, gradient, wavefunctions, eigenvalues supplied in info-dictionary.

     Subclasses will have to override the method calculate_non_local which will be supplied
     with items requested by NonLocalFunctionalDesc. NonLocalFunctional will "help" the it's subclasses
     by calculating these items beforehand, so adding different types of functionals will be more
     straightforward. NonLocalFunctional will also perform the potential mixing needed for potentials
     constructed from orbitals, since the mixing is otherwise performed to density only in GPAW.

     NonLocalFunctional handles only 3D-case, so 1D-setup generation must be handled by subclasses.
    """

    def __init__(self):
        """
        The constructor on NonLocalFunctional. The mixing is setup manually in this function.
       
        """
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
        self.old_v_sg = []

        if self.desc.needs_gradient():
            # Allocate stuff needed for gradient
            self.ddr = [Gradient(finegd, c).apply for c in range(3)]
            self.dndr_cg = finegd.empty(3)
            self.a2_g = finegd.empty(nspins)

    def get_functional_desc(self):
        """Returns a NonLocalFunctionalDesc instance specifiying the dependence of this functional.
           on different parameters.

           Currently, there is 4 possibilities:
              - density, gradient, wavefunctions and eigenvalues

           To be implemented in subclasses of NonLocalFunctional.
           =========== ==========================================================
           Parameters:
           =========== ==========================================================
           none
           =========== ==========================================================

        """

        raise "NonLocalFunctional::get_functional_desc must be overrided"

    def calculate_non_local_paw_correction(self, a, s, xc_corr, v_g, vt_g):
        """Called from XCNonLocalCorrections class to calculate the non-local paw-corrections.
           To be implemented in subclasses of NonLocalFunctional

           Note: This methods parameters are likely to change a lot in future, 
           as I optimize the code) - Mikael
        """

        raise "NonLocalFunctional::calculate_non_local_paw_correction not implemented"

    def calculate_non_local(self, info, v_g, e_g):
        """Add the KS-Exchange potential to v_g and supply energy density using data
           supplied in dictionary info. This dictionary info will contain the necessary
           information to calculate any type of (exchange) potential. 

           To be implemented in subclasses of NonLocalFunctional.

           =========== ==========================================================
           Parameters:
           =========== ==========================================================
           info        A dictiorany, see below
           v_g         The Kohn-Sham potential.
           e_g         The energy density
           =========== ==========================================================

           info is a dictiorary with following content:
           =========== ==========================================================
           Key:        Value:
           =========== ==========================================================
           typecode    For example num.Float, if the orbitals are real
           gd          The grid descriptor object for coarse grid
           finegd      The grid descriptor object for fine grid
           n_g         Numeric array for density, supplied if needs_density() true
           psit_nG     A _python_ list containing the wavefunctions,
                       if needs_wavefunctions() true
           f_n         A _python_ list containing the occupations,
                       if needs_wavefunctions() true
           a2_g        Numeric array for gradient, if needs_gradient() true
           eps_n       A _python_ list of eigenvalues, if needs_eigenvalues() true
           =========== ==========================================================
           """

        raise "NonLocalFunctional::calculate_non_local must be overrided"

    def potential_mixing(self, v_sg):
        """
        Perform the potential mixing.

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        v_sg        A python list containing the potentials to be mixed.
                    v_sg contains one potential for spin-paired calculation
                    and two potentials for spin polarized calculation.
        =========== ==========================================================
        """

        # If old_vt_gs has not been allocated yet, or the potential shape has changed
        if len(self.old_v_sg) != len(v_sg):
            # Create a copy from the orginal potential to be old_vt_g
            old_vt_g = [ v_g.copy() for v_g in v_sg ]

        # Mix the potentials
        for v_g, old_v_g in zip(v_sg, self.old_v_sg):
            v_g[:] = self.mixing * v_g[:] + (1.0 - self.mixing) * old_v_g[:]
            old_v_g[:] = v_g[:]

    def generate_info(self, n_sg):
        """Generate all quanities, the xc-calculator needs.

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        n_gs        A python list of densities for each spin
        =========== ==========================================================
        RETURNS     info object, a python list containing an info dictionary for each spin.
                    see calculate_non_local for more info about this dictionary.
        =========== ==========================================================

        """
        # Calculate the exchange potential
        info_s = []

        for s, n_g in enumerate(n_sg):
            info = {}

            # Supply the grid and the fine grid
            info['gd'] = self.gd
            info['finegd'] = self.finegd

            # Supply density if it is required
            if self.desc.needs_density():
                info['n_g'] = n_g

            # Supply the gradient if needed
            if self.desc.needs_gradient():
                # Calculate the gradient
                for c in range(3):
                    self.ddr[c](n_g, self.dndr_cg[c])
                self.a2_g[s][:] = num.sum(self.dndr_cg**2)

                info['a2_g'] = self.a2_g[s]

            # Supply the eigenvalues if required
            if self.desc.needs_eigenvalues():
                info['eps_n'] = []
                for kpt in self.kpt_u:
                    # Add only eigenvalues for this spin
                    if kpt.s == s:
                        for e in kpt.eps_n:
                            info['eps_n'].append(e)

            # Supply the wavefunctions if required
            if self.desc.needs_wavefunctions():
                psit_nG = []
                f_n = []
                for kpt in self.kpt_u:
                    # Add only wave functions for this spin
                    if kpt.s == s:
                        for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                            psit_nG.append(psit_G)
                            f_n.append(f)

                info['psit_nG'] = psit_nG
                info['f_n'] = f_n
                info['typecode'] = self.kpt_u[0].typecode

            info_s.append(info);

        return info_s

    def calculate_spinpaired(self, e_g, n_g, v_g):
        """Calculates the KS-exchange potential for spin paired calculation
           and adds it to v_g. Supplies also the energy density.
           This methods collects the required info (see generate_info) for each spin
           and calls calculate_non_local with this info for both spins separately.

           This method is called from xc_functional.py.

           =========== ==========================================================
           Parameters:
           =========== ==========================================================
           e_g         The energy density
           n_g         The electron density
           v_g         The Kohn-Sham potential.
           =========== ==========================================================

        """

        info_s = self.generate_info([n_g])
        v_sg = [ v_g ]

        # Calculate the exchange potential
        self.calculate_non_local(info_s, v_sg, e_g)

        # Mix the potential
        # Whatever is already in v_g gets mixed too. This is ok for now, but needs to be checked later
        self.potential_mixing(v_sg)

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        """Calculates the KS-exchange potential for spin polarized calculation
        and adds it to v_g. Supplies also the energy density.
        This methods collects the required info (see generate_info)
        and calls calculate_non_local with this info. 

        This method is called from xc_functional.py.

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        e_g          The energy density
        na_g         The electron density for spin alpha
        va_g         The Kohn-Sham potential for spin alpha
        na_g         The electron density for spin beta
        va_g         The Kohn-Sham potential for spin beta
        =========== ==========================================================

        """
        info_s = self.generate_info([na_g, nb_g])
        v_sg = [ va_g, vb_g ]

        self.calculate_non_local(info_s, v_sg, e_g)

        # Mix the potentials
        # Whatever is already in va_g or vb_g gets mixed too. This is ok for now, but needs to be checked later
        # how this affects convergence.
        self.potential_mixing(v_sg)

