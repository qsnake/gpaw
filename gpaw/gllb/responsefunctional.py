from gpaw.gllb.nonlocalfunctional import NonLocalFunctional
from gpaw.gllb import find_nucleus, SMALL_NUMBER

import Numeric as num
from gpaw.utilities.blas import axpy
from multiarray import matrixproduct as dot3
from gpaw.utilities.complex import cc, real
from gpaw.utilities import pack

#REMOVE ME
K_G = 0.382106112167171
def gllb_weight(epsilon, reference_level):
    """Calculates the weight for GLLB functional.
    
    The parameter K_G is adjusted such that the correct result is obtained for
    non-interacting electron gas.

    All orbitals closer than 1e-3 to reference level are consider the
    give zero response. This is to improve convergence of systems with
    degenerate orbitals.

    =============== ==========================================================
    Parameters:
    =============== ==========================================================
    epsilon         The eigenvalue of current orbital
    reference_level The reference level of the system. (Usually HOMO-orbital)
    =============== ==========================================================
    """

    if (epsilon +1e-5 > reference_level):
        return 0
    return K_G * num.sqrt(reference_level-epsilon)

class ResponseFunctional(NonLocalFunctional):
    """Response-part, for example for KLI and GLLB functionals.
    
    This class implements a functional which contains a
    "response"-part, for example for KLI and GLLB functionals.

    In general any functional containing following term will be a
    response functional::
    
      V_x = V_s + \sum_i w_i frac{|psi_i(r)|^2}{rho(r)}
    
    The subclasses must override the function
    ``get_slater_part_and_weights(self, info_s, v_sg)``.

    For more information, see description of function
    ResponseFunctional.get_slater_part_and_weights, and for example,
    see GLLBFunctional.get_slater_part_and_weights.
    """

    def __init__(self, relaxed_core_response):
        """
        Initialize the ResponseFunctional. 

        """

        NonLocalFunctional.__init__(self)
        self.relaxed_core_response = relaxed_core_response

        self.initialization_ready = False

    def pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation, kpt_comm):
        """Called from xc_corrections to get the important items required in non local calculation
         of vxc. In this method, also some temporary arrays are allocated which are needed
         for calculation of the response part.
        """

        # Pass the arguments forward to superclass.
        NonLocalFunctional.pass_stuff(self, kpt_u, gd, finegd, interpolate, nspins, nuclei, occupation, kpt_comm)

        # These arrays are needed while calculating response-part
        self.vt_G = gd.zeros()         # Temporary array for coarse potential
        self.vt_g = finegd.zeros()     # Temporary array for fine potential
        self.nt_G = gd.zeros()         # Temporary array for coarse density

    def calculate_non_local(self, info_s, v_sg, e_g):
        """
        Calculate the non-local response part. Also includes the energy calculation and slater potential
        which are obtained by a call to a virtual method get_slater_part_and_weigts.
        This method is to be overridden in subclasses.

        ============= ==========================================================
        Parameters:
        ============= ==========================================================
        info_s        The data needed for calculating v_sg
        v_sg          The GLLB-potential is added to this supplied potential array.
        e_g           The GLLB-energy density
        ============= ==========================================================

        """

        e_g[:] = 0.0
        self.w_sn = self.get_slater_part_and_weights(info_s, v_sg, e_g)

        for s, (v_g, info, w_n) in enumerate(zip(v_sg, info_s, self.w_sn)):

            # Use the coarse grid for response part
            # Calculate the coarse response multiplied with density and the coarse density
            # and to the division at the end of the loop.
            self.vt_G[:] = 0.0

            # For each orbital, add the response part
            for f, e, psit_G, w in zip(info['f_n'], info['eps_n'], info['psit_nG'], w_n):
                if info['typecode'] is num.Float:
                    axpy(f*w, psit_G**2, self.vt_G)
                else:
                    self.vt_G += f * w * (psit_G * num.conjugate(psit_G)).real

            # Communicate the coarse-response part
            self.kpt_comm.sum(self.vt_G)

            self.vt_g[:] = 0.0 # TODO: Is this needed for interpolate?
            self.interpolate(self.vt_G, self.vt_g)

            self.vt_g /= info['n_g'] + SMALL_NUMBER

            # Add the fine-grid response part to total potential
            v_g[:] += self.vt_g


    def get_slater_part_and_weights(self, info_s, v_sg, e_g):
        """The Slater potential (or V_s) is to be calculated using this method. 
           ResponseFunctional also let's the it's subclass to decide how the response coefficient's
           w_i are calculated. This method must returns these coefficients as a python list.        
        """
        # TODO: This method needs more arguments to support arbitary slater part
        raise "ResponseFunctional::get_slater_and_weights must be overrided by sub-class"

    def get_slater_part_paw_correction(self, rgd, n_g, a2_g, v_g, pseudo = True):
        """The paw-corrections due to Slater potential (or V_s) is to be calculated using this method.
           This method is called from calculate_non_local_paw_correction. """
        # TODO: This method needs more arguments to support arbitary slater part
        raise "ResponseFunctional::get_slater_part_paw_correction must be overrided by sub-class"

    def calculate_non_local_paw_correction(self, a, s, xccorr, slice, v_g, vt_g):
        nucleus = find_nucleus(self.nuclei, a)
        N = len(xccorr.n_g)

        # TODO: Allocate these only once
        vtemp_g = num.zeros(N, num.Float)
        e_g = num.zeros(N, num.Float)
        deda2_g = num.zeros(N, num.Float)
        # Calculate the density matrix only at first slice
        # Calculate also the core-response in self.relaxed_core_response is True
        if slice == 0:
            if self.relaxed_core_response:
                self.core_response = num.zeros(N, num.Float)
                njcore = xccorr.extra_xc_data['njcore']
                for nc in range(0, njcore):
                    psi2_g = xccorr.extra_xc_data['core_orbital_density_'+str(nc)]
                    epsilon = xccorr.extra_xc_data['core_eigenvalue_'+str(nc)]

                    self.core_response[:] += psi2_g * gllb_weight(epsilon, self.reference_levels[s])
            Dnn_Lq = 0
            # Calculate nn_Lq
            # For each k-point
            # Variable i is result of poor indexing of my w_sn,
            # I need to figure out better storage format (Mikael)
            i = 0
            for kpt in self.kpt_u:
                # Include only k-points with same spin
                if kpt.s == s:
                    # Get the projection coefficients
                    P_ni = nucleus.P_uni[kpt.u]

                    # Create the coefficients
                    # TODO: Better conversion from python array to num.array"
                    w_i = num.zeros(kpt.eps_n.shape, num.Float)
                    for j in range(len(w_i)):
                        w_i[j] = self.w_sn[s][i]
                        i = i + 1
                    w_i = w_i[:, num.NewAxis] * kpt.f_n[:, num.NewAxis] * xccorr.deg

                    # Calculate the 'density matrix' for numerator part of potential
                    Dn_ii = real(num.dot(cc(num.transpose(P_ni)),
                                         P_ni * w_i))
                    Dn_p = pack(Dn_ii) # Pack the unpacked densitymatrix
                    Dnn_Lq += dot3(xccorr.B_Lqp, Dn_p)

            # Communicate over K-points
            self.kpt_comm.sum(Dnn_Lq)

            # Store the Dnn_Lq matrix for later slices
            self.Dnn_Lq = Dnn_Lq
        else:
            # Get the Dnn_Lq matrix calculated in first slice
            Dnn_Lq = self.Dnn_Lq

        # Calculate the real response part of valence electrons multiplied with density
        nn_Lg = num.dot(Dnn_Lq, xccorr.n_qg)
        nn = num.dot(xccorr.Y_L, nn_Lg)

        # Calculate the Slater-parts from both, true and smooth densities
        Exc = self.get_slater_part_paw_correction(xccorr.rgd, xccorr.n_g, xccorr.a2_g, v_g, pseudo=False)
        Exc-= self.get_slater_part_paw_correction(xccorr.rgd, xccorr.nt_g, xccorr.at2_g, vt_g, pseudo=True)

        # Put the response-part of valence electrons and of core electrons to hard potential
        v_g[:] += nn / (xccorr.n_g + SMALL_NUMBER) 
     
        if self.relaxed_core_response:
            v_g[:] += self.core_response / (xccorr.n_g + SMALL_NUMBER)
        else:
            v_g[:] += xccorr.extra_xc_data['core_response']

        # Calculate the pseudo response part of valence electrons multiplied with density
        nn_Lg = num.dot(Dnn_Lq, xccorr.nt_qg)
        nn = num.dot(xccorr.Y_L, nn_Lg)

        # Put this response-part to smooth potential
        vt_g[:] += nn / (xccorr.nt_g + SMALL_NUMBER)

        return Exc

