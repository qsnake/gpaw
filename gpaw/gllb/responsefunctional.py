from gpaw.gllb.nonlocalfunctional import NonLocalFunctional
from gpaw.gllb import find_nucleus, SMALL_NUMBER

import Numeric as num
from gpaw.utilities.blas import axpy
from multiarray import matrixproduct as dot3
from gpaw.utilities.complex import cc, real
from gpaw.utilities import pack

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

    def __init__(self):
        """
        Initialize the ResponseFunctional. 

        """

        NonLocalFunctional.__init__(self)
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
        Calculate the non-local response part. Also icludes the energy calculation and slater potential
        which are obtained by a call to a virtual method get_slater_part_and_weigts.
        This method is to be overridden in subclasses.

        ============= ==========================================================
        Parameters:
        ============= ==========================================================
        v_g           The GLLB-potential is added to this supplied potential
        e_g           The GLLB-energy density (is the same as e_g)
        ============= ==========================================================

        """

        e_g[:] = 0.0
        self.w_sn = self.get_slater_part_and_weights(info_s, v_sg, e_g)

        for s, (v_g, info, w_n) in enumerate(zip(v_sg, info_s, self.w_sn)):

            # Use the coarse grid for response part
            # Calculate the coarse response multiplied with density and the coarse density
            # and to the division at the end of the loop.
            self.vt_G[:] = 0.0
            ####self.nt_G[:] = 0.0

            # For each orbital, add the response part
            for f, e, psit_G, w in zip(info['f_n'], info['eps_n'], info['psit_nG'], w_n):
                if info['typecode'] is num.Float:
                    psit_G2 = psit_G**2
                    axpy(f*w, psit_G2, self.vt_G)
                else:
                    psit_G2 = (psit_G * num.conjugate(psit_G)).real
                    self.vt_G += f * w * psit_G2

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

    def calculate_non_local_paw_correction(self, a, s, xccorr, v_g, vt_g):
        nucleus = find_nucleus(self.nuclei, a)
        N = len(xccorr.n_g)

        # TODO: Allocate these only once
        vtemp_g = num.zeros(N, num.Float)
        e_g = num.zeros(N, num.Float)
        deda2_g = num.zeros(N, num.Float)

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

                # Very serious optimization case here: The "weight density" matrix 
                # is now calculated here in inner loop when it could be done only once!!

                # Calculate the 'density matrix' for numerator part of potential
                Dn_ii = real(num.dot(cc(num.transpose(P_ni)),
                                     P_ni * w_i))

                Dn_p = pack(Dn_ii) # Pack the unpacked densitymatrix
                Dnn_Lq += dot3(xccorr.B_Lqp, Dn_p)

        # Communicate over K-points
        self.kpt_comm.sum(Dnn_Lq)

        # Calculate the real response part of valence electrons multiplied with density
        nn_Lg = num.dot(Dnn_Lq, xccorr.n_qg)
        nn = num.dot(xccorr.Y_L, nn_Lg)

        # Calculate the Slater-parts from both, true and smooth densities
        Exc = self.get_slater_part_paw_correction(xccorr.rgd, xccorr.n_g, xccorr.a2_g, v_g, pseudo=False)
        Exc-= self.get_slater_part_paw_correction(xccorr.rgd, xccorr.nt_g, xccorr.at2_g, vt_g, pseudo=True)

        # Put the response-part of valence electrons and of core electrons to hard potential
        v_g[:] += nn / (xccorr.n_g + SMALL_NUMBER) + xccorr.extra_xc_data['core_response']

        # Calculate the pseudo response part of valence electrons multiplied with density
        nn_Lg = num.dot(Dnn_Lq, xccorr.nt_qg)
        nn = num.dot(xccorr.Y_L, nn_Lg)

        # Put this response-part to smooth potential
        vt_g[:] += nn / (xccorr.nt_g + SMALL_NUMBER)

        return Exc

