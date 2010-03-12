"""This module provides an interface class for phonon calculations."""

__all__ = ["PhononCalculator"]

from math import sqrt

import pickle
import numpy as np

import ase.units as units

from gpaw.utilities import unpack, unpack2

from gpaw.dfpt.linearresponse import LinearResponse
from gpaw.dfpt.phononperturbation import PhononPerturbation


class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    
    def __init__(self, atoms):
        """Inititialize class with a list of atoms."""
 
        # Store useful objects
        self.atoms = atoms
        self.calc = atoms.get_calculator()

        # Phonon perturbation
        self.perturbation = PhononPerturbation(self.calc)
        # Linear response calculator
        self.response = LinearResponse(self.calc, self.perturbation)

        # Dynamical matrix
        self.D = None

    def run(self, tolerance_sc = 1e-5,
            tolerance_sternheimer = 1e-5, use_dfpt = True,
            file_dn = None, h = None, delta = None):
        """Run ..."""

        # Initialize phonon perturbation
        self.perturbation.initialize()

        # Calculate linear response wrt displacements of all atoms
        for atom in self.atoms:

            a = atom.index

            for v in [0,1,2]:
                
                self.perturbation.set_perturbation(a, v)

                if use_dfpt:
                    nt1_G, psit1_unG = self.response(
                        tolerance_sc = tolerance_sc,
                        tolerance_sternheimer = tolerance_sternheimer)
                else: # load from file

                    basename = "eps_%1.1e_h_%1.1e_a_%.1i_v_%.1i.pckl" % (delta,h,a,v)
                    fname = "_".join([file_dn, basename])
                    dpsi_n, dn = pickle.load(open(fname))
                    dn /= 2 * delta / units.Bohr
                    dpsi_n /= 2 * delta / units.Bohr

                    # Project out components from occupied orbitals
                    self.response.initialize()
                    s = self.response.sternheimer_operator
                    s.set_blochstate(0,0)
                    for n in range(len(dpsi_n)):
                        s.project(dpsi_n[n])
                    nt1_G = dn.copy()
                    psit1_unG = [dpsi_n.copy()]
                    # Call this function to calculate derivatives of Vghat to be
                    # used below (only when dn is loaded from file!!)
                    self.perturbation.calculate_derivative()        
        
    def dynamical_matrix(self, tolerance_sc = 1e-5,
                         tolerance_sternheimer = 1e-5, use_dfpt = True,
                         file_dn = None, h = None, delta = None):
        """Calculate second derivative of total energy wrt atomic displacements

        tolerance_sc: float
            Tolerance for the self-consistent density change
        tolerance_sternheimer: float
            Tolerance for the linear solver in the solution of the Sternheimer
            equation 
        use_dfpt: bool
            Temp hack - use SC calculated density changes instead of DFPT
        file_dn: string
            Temp hack - name of file with the density change
        h: float
            Temp hack - used in filename for SC density change
        delta: float
            Temp hack - size of atomic displacement (in Ang) used to calculate
            the density change in the loaded file
            
        """

        # Useful quantities
        density = self.calc.density
        nbands = self.calc.wfs.nvalence/2
        # Localized functions
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar
        pt = self.calc.wfs.pt

        # Initialize phonon perturbation
        self.perturbation.initialize()

        ########################################################################
        # Contributions involving ground-state properties                      #
        ########################################################################
        
        # Integrate Hartree potential with second derivative of ghat
        vH_g = self.calc.hamiltonian.vHt_g.copy()
        d2ghat_aLvv = dict([ (atom.index, np.zeros((3,3)))
                             for atom in self.atoms ])
        ghat.second_derivative(vH_g, d2ghat_aLvv)
 
        # Integrate electron density with second derivative of vbar
        nt_g = self.calc.density.nt_g.copy()
        d2vbar_avv = dict([ (atom.index, np.zeros((3,3)))
                            for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)
  
        ########################################################################
        # Contributions involving the linear response wrt atomic displacements #
        ########################################################################     

        # 1) Contribution from compensation charges and vbar potential
        # Integrate first-order density variation with Hartree potential from ghat derivative
        dghat_aaLvv = dict( [ (atom.index, dict( [
            (atom_.index, np.zeros((ghat.get_function_count(atom.index), 3, 3)))
            for atom_ in self.atoms ])) for atom in self.atoms] )
        # Integrate first-order density variation with vbar derivative
        dvbar_aavv = dict( [ (atom.index, dict( [ (atom_.index, np.zeros((3,3)))
                           for atom_ in self.atoms ] )) for atom in self.atoms])

        # 2) Contribution from projector functions
        nl_aavv = dict( [ (atom.index, dict([(atom_.index, np.zeros((3,3)))
                       for atom_ in self.atoms])) for atom in self.atoms])

        # Calculate linear response wrt displacements of all atoms
        for atom in self.atoms:

            a = atom.index

            for v in [0,1,2]:
                
                self.perturbation.set_perturbation(a, v)

                if use_dfpt:
                    nt1_G, psit1_unG = self.response(
                        tolerance_sc = tolerance_sc,
                        tolerance_sternheimer = tolerance_sternheimer)
                else: # load from file

                    basename = "eps_%1.1e_h_%1.1e_a_%.1i_v_%.1i.pckl" % (delta,h,a,v)
                    fname = "_".join([file_dn, basename])
                    dpsi_n, dn = pickle.load(open(fname))
                    dn /= 2 * delta / units.Bohr
                    dpsi_n /= 2 * delta / units.Bohr

                    # Project out components from occupied orbitals
                    self.response.initialize()
                    s = self.response.sternheimer_operator
                    s.set_blochstate(0,0)
                    for n in range(len(dpsi_n)):
                        s.project(dpsi_n[n])
                    nt1_G = dn.copy()
                    psit1_unG = [dpsi_n.copy()]
                    # Call this function to calculate derivatives of Vghat to be
                    # used below (only when dn is loaded from file!!)
                    self.perturbation.calculate_derivative()
                    
                # First-order density change
                nt1_g = density.finegd.zeros()
                self.calc.density.interpolator.apply(nt1_G, nt1_g)
                # Corresponding potential
                ps = self.calc.hamiltonian.poisson
                v1_g = density.finegd.zeros()
                ps.solve(v1_g, nt1_g)
                #######################################
                # Add derivative of compensation charge potential
                # v1_g += self.perturbation.vghat1_g
                ########################################
                
                # Integrate the potential with derivative of ghat
                dghat_aLv = ghat.dict(derivative=True)
                ghat.derivative(v1_g, dghat_aLv)
                
                ###################################
                dghat2_aLv = ghat.dict(derivative=True)
                ghat.derivative(self.perturbation.vghat1_g, dghat2_aLv)
                ###################################
                
                # Integrate density derivative with vbar derivative
                dvbar_av = vbar.dict(derivative=True)
                vbar.derivative(nt1_g, dvbar_av)

                # Generalize these two to sum over k-points
                # Overlap between wave-function variations and projectors
                Pdpsi_ani = pt.dict(shape=nbands, zero=True)
                pt.integrate(psit1_unG[0], Pdpsi_ani)
                # Overlap between wave-function variations and derivative of projectors
                dPdpsi_aniv = pt.dict(shape=nbands, derivative=True)
                pt.derivative(psit1_unG[0], dPdpsi_aniv)

                for atom_ in self.atoms:
                    
                    a_ = atom_.index
                    
                    dghat_aaLvv[a][a_][:,v] += dghat_aLv[a_]
                    dghat_aaLvv[a][a_][:,v] += dghat2_aLv[a_]
                    dvbar_aavv[a][a_][v] += dvbar_av[a_][0]

                    # Contribution from projectors - remember sign convention
                    # when integrating with first derivative of lfc's
                    dH_ii = unpack(self.calc.hamiltonian.dH_asp[a_][0])
                    P_ni = self.calc.wfs.kpt_u[0].P_ani[a_][:nbands]
                    dP_niv = -1 * self.perturbation.dP_aniv[a_]
                    d2P_nivv = self.perturbation.d2P_anivv[a_]
                    Pdpsi_ni = Pdpsi_ani[a_]
                    dPdpsi_niv = -1 * dPdpsi_aniv[a_]
                    
                    dHP_ni = np.dot(P_ni, dH_ii)
                    dHdP_niv = np.swapaxes(np.dot(dH_ii, dP_niv), 0, 1)
                    dHPdpsi_ni = np.dot(Pdpsi_ni, dH_ii)
                    
                    if a == a_:
                        # Diagonal contributions
                        nl_aavv[a][a] += 2 * (d2P_nivv *
                            dHP_ni[:, :, np.newaxis, np.newaxis]).sum(0).sum(0)
                        # The newaxis below does the following:
                        # dP_niv1 -> dP_niv1v and dHdP_niv2 -> dHdP_nivv2 where
                        # the existing array is repeated along the new v-dimension
                        nl_aavv[a][a] += 2 * (dP_niv[:,:, np.newaxis, :] * \
                            dHdP_niv[..., np.newaxis]).sum(0).sum(0)

                    nl_aavv[a][a_][v] = 2 * (dPdpsi_niv *
                        dHP_ni[...,np.newaxis]).sum(0).sum(0)
                    nl_aavv[a][a_][v] = 2 * (dP_niv *
                        dHPdpsi_ni[..., np.newaxis]).sum(0).sum(0)

        self.nl_aavv = nl_aavv
        self.d2ghat_aLvv = d2ghat_aLvv
        self.d2vbar_avv = d2vbar_avv
        self.dghat_aaLvv = dghat_aaLvv
        self.dvbar_aavv = dvbar_aavv
        
        self.assemble()
        
    def assemble(self):
        """Assemble the dynamical matrix from the calculated dictionaries."""

        # Useful quantities
        N_atoms = self.atoms.get_number_of_atoms()
        masses = self.atoms.get_masses()
        # Compensation charge coefficients
        Q_aL = self.calc.density.Q_aL
       
        # Dynamical matrix
        D = np.zeros((3 * N_atoms, 3 * N_atoms))

        # Hartree potential with d2ghat
        D_1 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # Electron density with d2vbar
        D_2 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # density derivative potential with derivative of ghat
        D_3 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # density derivative with derivative of vbar
        D_4 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # Projectors
        D_5 = np.zeros((3 * N_atoms, 3 * N_atoms))

        # Get the different contributions
        d2ghat_aLvv = self.d2ghat_aLvv
        d2vbar_avv = self.d2vbar_avv
        dghat_aaLvv = self.dghat_aaLvv
        dvbar_aavv = self.dvbar_aavv
        nl_aavv = self.nl_aavv
        
        # Assemble dynamical matrix        
        for atom in self.atoms:

            a = atom.index
            m_a = masses[a]

            ####################################################################
            #                      Diagonal contributions                      #
            ####################################################################
            # NOTICE: HGH has only one ghat pr atoms -> generalize when
            # implementing PAW
            D[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2ghat_aLvv[a] * Q_aL[a]
            D[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2vbar_avv[a]
            
            D_1[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2ghat_aLvv[a] * Q_aL[a]
            D_2[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2vbar_avv[a]
            
            for atom_ in self.atoms:

                ################################################################
                #                Off-diagonal contributions                    #
                ################################################################
                a_ = atom_.index
                m_a_ = masses[a_]
                
                # Mass prefactor
                c = (m_a * m_a_)**(-.5)
                # The minus sign below is due to the definition of the
                # derivative in the lfc derivative method
                # Generalize this to more than one ghat pr atom
                D[3*a:3*a+3, 3*a_: 3*a_+3] -= c * \
                    (dghat_aaLvv[a][a_][0] * Q_aL[a_][0] + dvbar_aavv[a][a_])
                D[3*a:3*a+3, 3*a_: 3*a_+3] += c * nl_aavv[a][a_]
                
                D_3[3*a:3*a+3, 3*a_: 3*a_+3] -= c * dghat_aaLvv[a][a_][0] * Q_aL[a_][0]
                D_4[3*a:3*a+3, 3*a_: 3*a_+3] -= c * dvbar_aavv[a][a_]
                D_5[3*a:3*a+3, 3*a_: 3*a_+3] += c * nl_aavv[a][a_]

        self.D_list = [D_1,D_2,D_3,D_4,D_5]
        # Symmetrize the dynamical matrix
        self.D_ = D.copy()
        D *= 0.5
        self.D = D + D.T
    
    def frequencies(self):
        """Calculate phonon frequencies from the dynamical matrix."""

        if self.D is None:
            print "Calculating dynamical matrix with default parameters."
            self.dynamical_matrix()

        # In Ha/(Bohr^2 * amu)
        D = self.D
        freq2, modes = np.linalg.eigh(D)
        freq = np.sqrt(freq2)
        # Convert to eV
        freq *= sqrt(units.Hartree) / (units.Bohr * 1e-10) / \
                sqrt(units._amu * units._e) * units._hbar

        print "Calculated frequencies (meV):"
        for f in freq:
            print f*1000

        return freq

    def forces(self):
        """Check for the forces."""

        N_atoms = self.atoms.get_number_of_atoms()
        Q_aL = self.calc.density.Q_aL
        # Forces
        F = np.zeros((N_atoms, 3))
        F_1 = np.zeros((N_atoms, 3))
        F_2 = np.zeros((N_atoms, 3))

        # Check the force
        ghat = self.calc.density.ghat
        vH_g = self.calc.hamiltonian.vHt_g.copy()
        F_ghat_av = ghat.dict(derivative=True)
        ghat.derivative(vH_g,F_ghat_av)
        
        vbar = self.calc.hamiltonian.vbar
        nt_g = self.calc.density.nt_g.copy()
        F_vbar_av = vbar.dict(derivative=True)
        vbar.derivative(nt_g,F_vbar_av)

        for atom in self.atoms:

            a = atom.index
            
            F_1[a] += F_ghat_av[a][0] * Q_aL[a][0]
            F_2[a] += F_vbar_av[a][0]
            F[a] += F_1[a] + F_2[a]
            
        # Convert to eV/Ang            
        F *= units.Hartree/units.Bohr
        F_1 *= units.Hartree/units.Bohr
        F_2 *= units.Hartree/units.Bohr        
        return F, F_1, F_2


##     def dynamical_matrix_atom(self, a, tolerance_sc = 1e-5,
##                               tolerance_sternheimer = 1e-5, use_dfpt = True,
##                               file_dn = None, h = None, delta = None):
##         """Calculate second derivative of total energy wrt atomic displacements

##         tolerance_sc: float
##             Tolerance for the self-consistent density change
##         tolerance_sternheimer: float
##             Tolerance for the linear solver in the solution of the Sternheimer
##             equation 
##         use_dfpt: bool
##             Temp hack - use SC calculated density changes instead of DFPT
##         file_dn: string
##             Temp hack - name of file with the density change
##         h: float
##             Temp hack - used in filename for SC density change
##         delta: float
##             Temp hack - size of atomic displacement (in Ang) used to calculate
##             the density change in the loaded file
            
##         """

##         # Useful quantities
##         N_atoms = self.atoms.get_number_of_atoms()
##         masses = self.atoms.get_masses()
##         density = self.calc.density

##         m_a = masses[a]
        
##         # Dynamical matrix
##         D = np.zeros((3, 3))

##         # Hartree potential with d2ghat
##         D_1 = np.zeros((3, 3))
##         # Electron density with d2vbar
##         D_2 = np.zeros((3, 3))
##         # density derivative potential with derivative of ghat
##         D_3 = np.zeros((3, 3))
##         # ghat derivative potential with derivative of ghat
##         D_4 = np.zeros((3, 3))
##         # density derivative with derivative of vbar
##         D_5 = np.zeros((3, 3))
        
##         # Localized functions
##         ghat = self.calc.density.ghat
##         vbar = self.calc.hamiltonian.vbar
##         # Compensation charge coefficients
##         Q_aL = self.calc.density.Q_aL
        
##         ########################################################################
##         # Contributions involving ground-state properties                      #
##         ########################################################################
        
##         # Integrate Hartree potential with second derivative of ghat
##         vH_g = self.calc.hamiltonian.vHt_g.copy()
##         d2ghat_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
##         ghat.second_derivative(vH_g, d2ghat_avv)

       
##         # NOTICE: HGH has only one ghat pr atoms -> generalize when
##         # implementing PAW
##         D += 1./m_a * d2ghat_avv[a] * Q_aL[a][0]
##         D_1 += 1./m_a * d2ghat_avv[a] * Q_aL[a][0]
            
##         # Integrate electron density with second derivative of vbar
##         nt_g = self.calc.density.nt_g.copy()
##         d2vbar_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
##         vbar.second_derivative(nt_g, d2vbar_avv)

##         D += 1./m_a * d2vbar_avv[a]
##         D_2 += 1./m_a * d2vbar_avv[a]
                       
##         ########################################################################
##         # Contributions involving the linear response wrt atomic displacements #
##         ########################################################################

##         for v in [0,1,2]:

##             self.perturbation.set_perturbation(a, v)            

##             if use_dfpt:
##                 nt1_G = self.response(
##                             tolerance_sc = tolerance_sc,
##                             tolerance_sternheimer = tolerance_sternheimer)
##             else:
##                 fcode = "eps_%1.1e_h_%1.1e_a_%.1i_v_%.1i.pckl" % (delta,h,a,v)
##                 fname = "_".join([file_dn, fcode])
##                 dpsi, dn = pickle.load(open(fname))
##                 dn /= 2 * delta / units.Bohr
##                 nt1_G = dn.copy()
                    
##                 # Call this function to calculate derivatives of Vghat to be
##                 # used below (only when dn is loaded from file!!)
##                 self.perturbation.calculate_derivative()    
                
##             # First-order density change
##             nt1_g = density.finegd.zeros()
##             self.calc.density.interpolator.apply(nt1_G, nt1_g)
##             # Corresponding potential
##             ps = self.calc.hamiltonian.poisson
##             v1_g = density.finegd.zeros()
##             ps.solve(v1_g, nt1_g)
##             #######################################
##             # Add derivative of compensation charge potential
##             # v1_g += self.perturbation.vghat1_g
##             ########################################
            
##             # Integrate the potential with derivative of ghat
##             dghat_aLv = ghat.dict(derivative=True)
##             ghat.derivative(v1_g, dghat_aLv)
            
##             ###################################
##             dghat2_aLv = ghat.dict(derivative=True)
##             ghat.derivative(self.perturbation.vghat1_g, dghat2_aLv)
##             ###################################
                
##             # Integrate density derivative with vbar derivative
##             dvbar_av = vbar.dict(derivative=True)
##             vbar.derivative(nt1_g, dvbar_av)

##             # The minus sign below is due to the definition of the
##             # derivative in the lfc derivative method
##             # Generalize this to more than one ghat pr atom
##             D[v] -= 1./m_a * (dghat_aLv[a][0] * Q_aL[a][0] +
##                          dghat2_aLv[a][0] * Q_aL[a][0] +
##                          dvbar_av[a])
##             D_3[v] -= 1./m_a * dghat_aLv[a][0] * Q_aL[a][0]
##             D_4[v] -= 1./m_a * dghat2_aLv[a][0] * Q_aL[a][0]
##             D_5[v] -= 1./m_a * dvbar_av[a]

##         self.D_list = [D_1,D_2,D_3,D_4,D_5]
##         # Symmetrize the dynamical matrix
##         self.D_ = D.copy()
##         D *= 0.5
##         self.D = D + D.T
        
##         return self.D, self.D_list
