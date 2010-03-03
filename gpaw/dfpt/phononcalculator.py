"""This module provides an interface class for phonon calculations."""

__all__ = ["PhononCalculator"]

from math import sqrt

import pickle
import numpy as np

import ase.units as units

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
        self.perturb = PhononPerturbation(self.calc)
        # Linear response calculator
        self.response = LinearResponse(self.calc, self.perturb)

        
        # Dynamical matrix
        self.D = None
        
    def dynamical_matrix(self, eps,
                         tolerance_sc = 1e-5,
                         tolerance_sternheimer = 1e-5,
                         use_dfpt = True,
                         file_dn = None,
                         h = None,
                         delta = None):
        """Calculate second derivative of total energy wrt atomic displacements

        eps: float
            Displacements (in Ang) of atoms in finite-difference derivatives
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
        N_atoms = self.atoms.get_number_of_atoms()
        masses = self.atoms.get_masses()
        density = self.calc.density
        
        # Dynamical matrix
        D = np.zeros((3 * N_atoms, 3 * N_atoms))

        # Hartree potential with d2ghat
        D_1 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # Electron density with d2vbar
        D_2 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # density derivative potential with derivative of ghat
        D_3 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # ghat derivative potential with derivative of ghat
        D_4 = np.zeros((3 * N_atoms, 3 * N_atoms))
        # density derivative with derivative of vbar
        D_5 = np.zeros((3 * N_atoms, 3 * N_atoms))
        
        # Localized functions
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = self.calc.density.Q_aL
        
        ########################################################################
        # Contributions involving ground-state properties                      #
        ########################################################################
        
        # Integrate Hartree potential with second derivative of ghat
        vH_g = self.calc.hamiltonian.vHt_g.copy()
        d2ghat_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
        ghat.second_derivative(vH_g, d2ghat_avv)

        for a, d2ghat_vv in d2ghat_avv.items():
            m_a = masses[a]
            # NOTICE: HGH has only one ghat pr atoms -> generalize when
            # implementing PAW
            D[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2ghat_vv * Q_aL[a]
            D_1[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2ghat_vv * Q_aL[a]
            
        # Integrate electron density with second derivative of vbar
        nt_g = self.calc.density.nt_g.copy()
        d2vbar_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        for a, d2vbar_vv in d2vbar_avv.items():
            m_a = masses[a]            
            D[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2vbar_vv
            D_2[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2vbar_vv            
               
        ########################################################################
        # Contributions involving the linear response wrt atomic displacements #
        ########################################################################

        # Calculate linear response wrt displacements of all atoms
        for atom in self.atoms:

            a = atom.index
            m_a = masses[a]

            for v in [0,1,2]:
                
                av = a * 3 + v

                self.perturb.set_perturbation(a, v)

                if use_dfpt:
                    nt1_G = self.response(
                        tolerance_sc = tolerance_sc,
                        tolerance_sternheimer = tolerance_sternheimer)
                else:
                    fcode = "eps_%1.1e_h_%1.1e_a_%.1i_v_%.1i.pckl" % (delta,h,a,v)
                    fname = "_".join([file_dn, fcode])
                    dpsi, dn = pickle.load(open(fname))
                    dn /= 2 * delta / units.Bohr
                    nt1_G = dn.copy()
                    
                    # Call this function to calculate derivatives of Vghat to be
                    # used below (only when dn is loaded from file!!)
                    self.perturb.calculate_derivative()
                
                # First-order density change
                nt1_g = density.finegd.zeros()
                self.calc.density.interpolator.apply(nt1_G, nt1_g)
                # Corresponding potential
                ps = self.calc.hamiltonian.poisson
                v1_g = density.finegd.zeros()
                ps.solve(v1_g, nt1_g)
                #######################################
                # Add derivative of compensation charge potential
                # v1_g += self.perturb.vghat1_g
                ########################################
              
                # Integrate the potential with derivative of ghat
                dghat_aLv = ghat.dict(derivative=True)
                ghat.derivative(v1_g, dghat_aLv)
                
                ###################################
                dghat2_aLv = ghat.dict(derivative=True)
                ghat.derivative(self.perturb.vghat1_g, dghat2_aLv)
                ###################################
                
                # Integrate density derivative with vbar derivative
                dvbar_av = vbar.dict(derivative=True)
                vbar.derivative(nt1_g, dvbar_av)

                for atom_ in self.atoms:
                
                    a_ = atom_.index
                    m_a_ = masses[a_]
                    
                    # Mass prefactor
                    c = (m_a * m_a_)**(-.5)
                    # The minus sign below is due to the definition of the
                    # derivative in the lfc derivative method
                    # Generalize this to more than one ghat pr atom
                    D[av, 3*a_: 3*a_+3] -= c * (dghat_aLv[a_][0] * Q_aL[a_][0] +
                                                dghat2_aLv[a_][0] * Q_aL[a_][0] +
                                                dvbar_av[a_])
                    D_3[av, 3*a_: 3*a_+3] -= c * dghat_aLv[a_][0] * Q_aL[a_][0]
                    D_4[av, 3*a_: 3*a_+3] -= c * dghat2_aLv[a_][0] * Q_aL[a_][0]
                    D_5[av, 3*a_: 3*a_+3] -= c * dvbar_av[a_]

        self.D_list = [D_1,D_2,D_3,D_4,D_5]
        # Symmetrize the dynamical matrix
        self.D_ = D.copy()
        D *= 0.5
        self.D = D + D.T
        
        return self.D


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


    def dynamical_matrix_atom(self, a, tolerance_sc = 1e-5,
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
        N_atoms = self.atoms.get_number_of_atoms()
        masses = self.atoms.get_masses()
        density = self.calc.density

        m_a = masses[a]
        
        # Dynamical matrix
        D = np.zeros((3, 3))

        # Hartree potential with d2ghat
        D_1 = np.zeros((3, 3))
        # Electron density with d2vbar
        D_2 = np.zeros((3, 3))
        # density derivative potential with derivative of ghat
        D_3 = np.zeros((3, 3))
        # ghat derivative potential with derivative of ghat
        D_4 = np.zeros((3, 3))
        # density derivative with derivative of vbar
        D_5 = np.zeros((3, 3))
        
        # Localized functions
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = self.calc.density.Q_aL
        
        ########################################################################
        # Contributions involving ground-state properties                      #
        ########################################################################
        
        # Integrate Hartree potential with second derivative of ghat
        vH_g = self.calc.hamiltonian.vHt_g.copy()
        d2ghat_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
        ghat.second_derivative(vH_g, d2ghat_avv)

       
        # NOTICE: HGH has only one ghat pr atoms -> generalize when
        # implementing PAW
        D += 1./m_a * d2ghat_avv[a] * Q_aL[a][0]
        D_1 += 1./m_a * d2ghat_avv[a] * Q_aL[a][0]
            
        # Integrate electron density with second derivative of vbar
        nt_g = self.calc.density.nt_g.copy()
        d2vbar_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        D += 1./m_a * d2vbar_avv[a]
        D_2 += 1./m_a * d2vbar_avv[a]
                       
        ########################################################################
        # Contributions involving the linear response wrt atomic displacements #
        ########################################################################

        for v in [0,1,2]:

            self.perturb.set_perturbation(a, v)            

            if use_dfpt:
                nt1_G = self.response(
                            tolerance_sc = tolerance_sc,
                            tolerance_sternheimer = tolerance_sternheimer)
            else:
                fcode = "eps_%1.1e_h_%1.1e_a_%.1i_v_%.1i.pckl" % (delta,h,a,v)
                fname = "_".join([file_dn, fcode])
                dpsi, dn = pickle.load(open(fname))
                dn /= 2 * delta / units.Bohr
                nt1_G = dn.copy()
                    
                # Call this function to calculate derivatives of Vghat to be
                # used below (only when dn is loaded from file!!)
                self.perturb.calculate_derivative()    
                
            # First-order density change
            nt1_g = density.finegd.zeros()
            self.calc.density.interpolator.apply(nt1_G, nt1_g)
            # Corresponding potential
            ps = self.calc.hamiltonian.poisson
            v1_g = density.finegd.zeros()
            ps.solve(v1_g, nt1_g)
            #######################################
            # Add derivative of compensation charge potential
            # v1_g += self.perturb.vghat1_g
            ########################################
            
            # Integrate the potential with derivative of ghat
            dghat_aLv = ghat.dict(derivative=True)
            ghat.derivative(v1_g, dghat_aLv)
            
            ###################################
            dghat2_aLv = ghat.dict(derivative=True)
            ghat.derivative(self.perturb.vghat1_g, dghat2_aLv)
            ###################################
                
            # Integrate density derivative with vbar derivative
            dvbar_av = vbar.dict(derivative=True)
            vbar.derivative(nt1_g, dvbar_av)

            # The minus sign below is due to the definition of the
            # derivative in the lfc derivative method
            # Generalize this to more than one ghat pr atom
            D[v] -= 1./m_a * (dghat_aLv[a][0] * Q_aL[a][0] +
                         dghat2_aLv[a][0] * Q_aL[a][0] +
                         dvbar_av[a])
            D_3[v] -= 1./m_a * dghat_aLv[a][0] * Q_aL[a][0]
            D_4[v] -= 1./m_a * dghat2_aLv[a][0] * Q_aL[a][0]
            D_5[v] -= 1./m_a * dvbar_av[a]

        self.D_list = [D_1,D_2,D_3,D_4,D_5]
        # Symmetrize the dynamical matrix
        self.D_ = D.copy()
        D *= 0.5
        self.D = D + D.T
        
        return self.D, self.D_list



#int_dict_2 = dict([ (i,np.zeros((3,3))) for i in [0,1] ])
#ghat.second_derivative(vH_g, int_dict_2)
# int_dict_4 = dict([ (i,np.zeros((3,3))) for i in [0,1] ])
# vbar.second_derivative(nt_g, int_dict_4)

# int_dict_1 = ghat.dict(derivative=True)
# ghat.derivative(v1_g,int_dict_1)                
# int_dict_3 = vbar.dict(derivative=True)
# vbar.derivative(nt1_g,int_dict_3)
