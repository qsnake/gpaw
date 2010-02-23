"""This modules implements an interface class to phonon calculations."""

from math import sqrt

import pickle
import numpy as np

import ase.units as units

from linearresponse import LinearResponse

__all__ = ["PhononCalculator"]

class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    
    def __init__(self, atoms):
        """Inititialize class with a list of atoms."""
 
        # Store useful objects
        self.atoms = atoms
        self.calc = atoms.get_calculator()

        # Linear response calculator
        self.response = LinearResponse(self.calc)

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
            Displacements of atoms in finite-difference derivatives
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
                                       
        # Integrate electron density with second derivative of vbar
        nt_g = self.calc.density.nt_g.copy()
        d2vbar_avv = dict([ (atom.index, np.zeros((3,3))) for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        for a, d2vbar_vv in d2vbar_avv.items():
            m_a = masses[a]            
            D[3*a:3*a+3, 3*a:3*a+3] += 1./m_a * d2vbar_vv
            
               
        ########################################################################
        # Contributions involving the linear response wrt atomic displacements #
        ########################################################################

        # Calculate linear response wrt displacements of all atoms
        for atom in self.atoms:

            a = atom.index
            m_a = masses[a]

            for v in [0,1,2]:
                
                av = a * 3 + v

                if use_dfpt:
                    
                    nt1_G = self.response.calculate_response(a, v, 
                        eps = eps/units.Bohr,
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
                    self.response.calculate_derivative(a, v, eps/units.Bohr)
                
                
                # First-order density change
                nt1_g = density.finegd.zeros()
                self.calc.density.interpolator.apply(nt1_G, nt1_g)
                # Corresponding potential
                ps = self.calc.hamiltonian.poisson
                v1_g = density.finegd.zeros()
                ps.solve(v1_g, nt1_g)
                # Add derivative of compensation charge potential
                v1_g += self.response.Vghat1_g
                
                # Integrate the potential with derivative of ghat
                dghat_aLv = ghat.dict(derivative=True)
                ghat.derivative(v1_g, dghat_aLv)
                
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
                    D[av, 3*a_: 3*a_+3] -= c * (dghat_aLv[a_] * Q_aL[a_]
                                                + dvbar_av[a_])[0]
                    
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





#int_dict_2 = dict([ (i,np.zeros((3,3))) for i in [0,1] ])
#ghat.second_derivative(vH_g, int_dict_2)
# int_dict_4 = dict([ (i,np.zeros((3,3))) for i in [0,1] ])
# vbar.second_derivative(nt_g, int_dict_4)

# int_dict_1 = ghat.dict(derivative=True)
# ghat.derivative(v1_g,int_dict_1)                
# int_dict_3 = vbar.dict(derivative=True)
# vbar.derivative(nt1_g,int_dict_3)
