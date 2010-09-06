"""This module provides a class for assembling the dynamical matrix."""

__all__ = ["DynamicalMatrix"]

from math import sqrt

import numpy as np
import numpy.fft as fft

import ase.units as units
from gpaw.utilities import unpack, unpack2

class DynamicalMatrix:
    """Class for assembling the dynamical matrix from first-order responses.

    The second order derivative of the total energy with respect to atomic
    displacements (for periodic systems collective atomic displacemnts
    characterized by a q-vector) can be obtained from an expression involving
    the first-order derivatives of the density and the wave-functions.
    
    Each of the various contributions to the second order derivative of the
    total energy are implemented in separate functions.
    
    """
    
    def __init__(self, atoms, ibzq_qc=None, dtype=float):
        """Inititialize class with a list of atoms."""

        # Store useful objects
        self.atoms = atoms
        self.dtype = dtype
        self.calc = atoms.get_calculator()
        self.masses = atoms.get_masses()
        self.N = atoms.get_number_of_atoms()

        #XXX Index of the gamma point -- for the acoustic sum-rule
        self.gamma_index = None
        
        if ibzq_qc is None:
            self.ibzq_qc = [(0., 0., 0.)]
            self.gamma_index = 0
            assert dtype == float
        else:
            #XXX Maybe not needed as an attribute ??
            self.ibzq_qc = ibzq_qc
            for q, q_c in enumerate(self.ibzq_qc):
                if np.all(q_c == 0.):
                    self.gamma_index = q            

        assert self.gamma_index is not None
        
        # Matrix of force constants -- dict of dicts in atomic indices
        # In case of inversion symmetry this is a real matrix !!
        self.C_qaavv = [dict([(atom.index,
                               dict([(atom_.index, np.zeros((3, 3), dtype=dtype))
                                     for atom_ in atoms])) for atom in atoms])
                        for q in self.ibzq_qc]
        
        # Dynamical matrix -- 3Nx3N ndarray (vs q)
        self.D_q = []
        #XXX Temp attribute
        self.D_q_ = []
        self.D = None

    def assemble(self, acoustic=False):
        """Assemble dynamical matrix from the force constants attribute ``C``.

        The elements of the dynamical matrix are given by::

            D_ij(q) = 1/(M_i + M_j) * C_ij(q) ,
                      
        where i and j are collective atomic and cartesian indices.

        Parameters
        ----------
        acoustic: bool
            When True, the diagonal of the matrix of force constants is
            corrected to ensure that the acoustic sum-rule is fulfilled.
            
        """

        # First assemble matrix of force constants, then apply acoustic
        # sum-rule
        for q, C_aavv in enumerate(self.C_qaavv):

            C_avav = np.zeros((3*self.N, 3*self.N), dtype=self.dtype)
    
            for atom in self.atoms:
    
                a = atom.index
    
                for atom_ in self.atoms:
    
                    a_ = atom_.index
    
                    C_avav[3*a : 3*a + 3, 3*a_ : 3*a_ + 3] += C_aavv[a][a_]

            # C(q) is Hermitian
            C = .5 * C_avav
            C = C + C.T.conj()
            self.D_q.append(C)
            #XXX Temp
            self.D_q_.append(C_avav)
            
        # Mass prefactor for the dynamical matrix
        m_av = np.repeat(np.asarray(self.masses)**(-0.5), 3)
        M_avav = m_av[:, np.newaxis] * m_av

        if acoustic:
            C_gamma = self.D_q[self.gamma_index]
            diag = C_gamma.sum(axis=1)
            
            for C in self.D_q:
                C -= np.diag(diag)
                C *= M_avav
        else:
            for C in self.D_q:
                C *= M_avav
            #XXX Temp
            for C in self.D_q_:
                C *= M_avav
                
    def fourier_interpolate(self):
        """Fourier interpolate dynamical matrix to a finer q-grid."""

        D_q = np.array(D_q)
    
    def update_row(self, perturbation, response_calc):
        """Update row of force constant matrix from first-order derivatives.

        Parameters
        ----------

        """

        self.density_derivative(perturbation, response_calc)
        # self.wfs_derivative(perturbation, response_calc)
        
    def density_ground_state(self, calc):
        """Contributions involving ground-state density.

        These terms contains second-order derivaties of the localized functions
        ghat and vbar. They are therefore diagonal in the atomic indices.

        """

        # Use the GS LFC's to integrate with the ground-state quantities !
        ghat = calc.density.ghat
        vbar = calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = calc.density.Q_aL
        
        # Integral of Hartree potential times the second derivative of ghat
        vH_g = calc.hamiltonian.vHt_g
        d2ghat_aLvv = dict([(atom.index, np.zeros((3, 3)))
                            for atom in self.atoms])
        ghat.second_derivative(vH_g, d2ghat_aLvv)

        # Integral of electron density times the second derivative of vbar
        nt_g = calc.density.nt_g
        d2vbar_avv = dict([(atom.index, np.zeros((3, 3)))
                           for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        for C_aavv in self.C_qaavv:

            for atom in self.atoms:
                
                a = atom.index
                # XXX: HGH has only one ghat pr atoms -> generalize when
                # implementing PAW            
                C_aavv[a][a] += d2ghat_aLvv[a] * Q_aL[a]
                C_aavv[a][a] += d2vbar_avv[a]

    def wfs_ground_state(self, calc, response_calc):
        """Ground state contributions from the non-local potential."""

        # Projector functions
        # pt = response_calc.wfs.pt
        pt = calc.wfs.pt
        # Projector coefficients
        dH_asp = calc.hamiltonian.dH_asp
      
        # K-point
        kpt_u = response_calc.wfs.kpt_u
        nbands = response_calc.nbands
        
        for kpt in kpt_u:

            # Index of k
            k = kpt.k
            P_ani = kpt.P_ani
            dP_aniv = kpt.dP_aniv
            
            # Occupation factors include the weight of the k-points
            f_n = kpt.f_n
            psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # Calculate d2P_anivv coefficients
            # d2P_anivv = self.calculate_d2P_anivv()
            d2P_anivv = dict([(atom.index,
                               np.zeros(
                (nbands, pt.get_function_count(atom.index), 3, 3)
                )) for atom in self.atoms])
            #XXX Temp dict, second_derivative method only takes a_G array
            # -- no extra dims
            d2P_avv = dict([(atom.index, np.zeros((3, 3)))
                            for atom in self.atoms])
         
            for n in range(nbands):
                pt.second_derivative(psit_nG[n], d2P_avv)
                # Insert in other dict
                for atom in self.atoms:
                    a = atom.index
                    d2P_anivv[a][n, 0] = d2P_avv[a]
            
            for atom in self.atoms:
    
                a = atom.index
    
                H_ii = unpack(dH_asp[a][0])
                P_ni = P_ani[a]
                dP_niv = -1 * dP_aniv[a]
                d2P_nivv = d2P_anivv[a]
                
                # Term with second-order derivative of projector
                HP_ni = np.dot(P_ni, H_ii)
                d2PHP_nvv = (d2P_nivv.conj() *
                             HP_ni[:, :, np.newaxis, np.newaxis]).sum(1)
                d2PHP_nvv *= kpt.f_n[:, np.newaxis, np.newaxis]
                A_vv = d2PHP_nvv.sum(0)
    
                # Term with first-order derivative of the projectors
                HdP_inv = np.dot(H_ii, dP_niv.conj())
                HdP_niv = np.swapaxes(HdP_inv, 0, 1)
                HdP_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
    
                B_vv = (dP_niv[:, :, np.newaxis, :] * 
                        HdP_niv[:, :, :, np.newaxis]).sum(0).sum(0)

                for C_aavv in self.C_qaavv:
                    
                    C_aavv[a][a] += (A_vv + B_vv) + (A_vv + B_vv).conj()

    def core_corrections(self):
        """Contribution from the derivative of the core density."""

        raise NotImplementedError
    
    def density_derivative(self, perturbation, response_calc):
        """Contributions involving the first-order density derivative."""

        # Get attributes from the phononperturbation
        a = perturbation.a
        v = perturbation.v
        #XXX: careful here, Gamma calculation has q=-1
        q = perturbation.q
        
        # Matrix of force constants to be updated; q=-1 for Gamma calculation!
        C_aavv = self.C_qaavv[q]
        
        # Localized functions 
        ghat = perturbation.ghat
        vbar = perturbation.vbar
        # Compensation charge coefficients
        Q_aL = perturbation.Q_aL

        # Density derivative
        nt1_g = response_calc.nt1_g
        
        # Hartree potential derivative including compensation charges
        vH1_g = response_calc.vH1_g.copy()
        vH1_g += perturbation.vghat1_g

        # Integral of Hartree potential derivative times ghat derivative
        dghat_aLv = ghat.dict(derivative=True)
        # Integral of density derivative times vbar derivative
        dvbar_av = vbar.dict(derivative=True)
        
        # Evaluate integrals
        ghat.derivative(vH1_g, dghat_aLv, q=q)
        vbar.derivative(nt1_g, dvbar_av, q=q)

        # Add to force constant matrix attribute
        for atom_ in self.atoms:
            a_ = atom_.index
            # Minus sign comes from lfc member function derivative
            C_aavv[a][a_][v] -= np.dot(Q_aL[a_], dghat_aLv[a_])
            C_aavv[a][a_][v] -= dvbar_av[a_][0]

    def wfs_derivative(self, perturbation, response_calc):
        """Contributions from the non-local part of the PAW potential."""

        # Get attributes from the phononperturbation
        a = perturbation.a
        v = perturbation.v
        q = perturbation.q

        # Matrix of force constants to be updated
        C_aavv = self.C_qaavv[q]
           
        # Projector functions
        pt = response_calc.wfs.pt
        # Projector coefficients
        dH_asp = perturbation.dH_asp
        
        # K-point
        kpt_u = response_calc.wfs.kpt_u
        nbands = response_calc.nbands

        # Get k+q indices
        if perturbation.has_q():
            q_c = perturbation.get_q()
            kplusq_k = response_calc.kd.find_k_plus_q(q_c)
        else:
            kplusq_k = range(len(kpt_u))
            
        for kpt in kpt_u:

            # Indices of k and k+q
            k = kpt.k
            kplusq = kplusq_k[k]

            # Projector coefficients
            P_ani = kpt.P_ani
            dP_aniv = kpt.dP_aniv
            
            # Occupation factors include the weight of the k-points
            f_n = kpt.f_n
            psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # Overlap between wave-function derivative and projectors
            Pdpsi_ani = pt.dict(shape=nbands, zero=True)
            pt.integrate(psit1_nG, Pdpsi_ani, q=kplusq)
            # Overlap between wave-function derivative and derivative of projectors
            dPdpsi_aniv = pt.dict(shape=nbands, derivative=True)
            pt.derivative(psit1_nG, dPdpsi_aniv, q=kplusq)

            for atom_ in self.atoms:
    
                a_ = atom_.index

                # Coefficients from atom a
                Pdpsi_ni = Pdpsi_ani[a]
                dPdpsi_niv = -1 * dPdpsi_aniv[a]
                # Coefficients from atom a_
                H_ii = unpack(dH_asp[a_][0])
                P_ni = P_ani[a_]
                dP_niv = -1 * dP_aniv[a_]
                
                # Term with dPdpsi and P coefficients
                HP_ni = np.dot(P_ni, H_ii)
                dPdpsiHP_nv = (dPdpsi_niv.conj() * HP_ni[:, :, np.newaxis]).sum(1)
                dPdpsiHP_nv *= f_n[:, np.newaxis]
                A_v = dPdpsiHP_nv.sum(0)
    
                # Term with dP and Pdpsi coefficients
                HPdpsi_ni = np.dot(Pdpsi_ni.conj(), H_ii)
                dPHPdpsi_nv = (dP_niv * HPdpsi_ni[:, :, np.newaxis]).sum(1)
                dPHPdpsi_nv *= f_n[:, np.newaxis]
                B_v = dPHPdpsi_nv.sum(0)

                # Factor of 2 from time-reversal symmetry
                C_aavv[a][a_][v] += 2 * (A_v + B_v)




