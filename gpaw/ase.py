# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

import os
import weakref

from ASE.Units import units, Convert
import ASE
import Numeric as num

from gpaw.paw import PAW


try:
    # Deal with old ASE version 2.3.5 and earlier:
    if 'PBS_NODEFILE' not in os.environ:
        os.environ['PBS_NODEFILE'] = '/dev/null'
    from ASE.Utilities.Parallel import register_parallel_cleanup_function
except ImportError:
    pass
else:
    register_parallel_cleanup_function()


class Calculator(PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation.
    """

    def __init__(self, filename=None, **kwargs):
        # Set units to ASE units:
        lengthunit = units.GetLengthUnit()
        energyunit = units.GetEnergyUnit()
        self.a0 = Convert(1, 'Bohr', lengthunit)
        self.Ha = Convert(1, 'Hartree', energyunit)

        PAW.__init__(self, filename, **kwargs)

        self.text('ASE: ', os.path.dirname(ASE.__file__))
        self.text('units:', lengthunit, 'and', energyunit)

    def convert_units(self, parameters):
        if parameters.get('h') is not None:
            parameters['h'] /= self.a0
        if parameters.get('width') is not None:
            parameters['width'] /= self.Ha
        if parameters.get('external') is not None:
            parameters['external'] = parameters['external'] / self.Ha
        if ('convergence' in parameters and
            'energy' in  parameters['convergence']):
            parameters['convergence']['energy'] /= self.Ha
        
    def GetPotentialEnergy(self, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        
        self.calculate()

        if force_consistent:
            # Free energy:
            return self.Ha * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return self.Ha * (self.Etot + 0.5 * self.S)

    def GetCartesianForces(self):
        """Return the forces for the current state of the ListOfAtoms."""
        self.calculate()
        self.calculate_forces()
        return self.F_ac * (self.Ha / self.a0)
      
    def GetStress(self):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def _SetListOfAtoms(self, atoms):
        """Make a weak reference to the ListOfAtoms."""
        self.lastcount = -1
        self.atoms = weakref.proxy(atoms)
        self.extra_list_of_atoms_stuff = (atoms.GetTags(),
                                          atoms.GetMagneticMoments())
        self.plot_atoms()

    def GetListOfAtoms(self):
        return self.atoms
    
    def GetNumberOfBands(self):
        """Return the number of bands."""
        return self.nbands 
  
    def GetXCFunctional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def GetBZKPoints(self):
        """Return the k-points."""
        return self.bzk_kc
 
    def GetSpinPolarized(self):
        """Is it a spin-polarized calculation?"""
        return self.nspins == 2
    
    def GetIBZKPoints(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.ibzk_kc

    # Alternative name:
    GetKPoints = GetIBZKPoints
 
    def GetIBZKPointWeights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.weight_k

    def GetDensityArray(self):
        """Return pseudo-density array."""
        return self.density.get_density_array() / self.a0**3

    def GetAllElectronDensity(self, gridrefinement=2):
        """Return reconstructed all-electron density array."""
        return self.density.get_all_electron_density(gridrefinement)\
               / self.a0**3

    def GetWignerSeitzDensities(self, spin=None):
        if spin is None and self.nspins == 1:
            return self.GetWignerSeitzDensities(spin=0)
        elif spin is None and self.nspins == 2:
            return (self.GetWignerSeitzDensities(spin=0) +
                    self.GetWignerSeitzDensities(spin=1))

        if not hasattr(self, 'wignerseitz'):
            from gpaw.analyse.wignerseitz import WignerSeitz
            self.wignerseitz = WignerSeitz(self.gd, self.nuclei)
        
        return self.wignerseitz.expand_density(self.density.nt_sG[spin], spin,
                                               self.nspins)

    def GetWaveFunctionArray(self, band=0, kpt=0, spin=0):
        """Return pseudo-wave-function array."""
        return self.get_wave_function_array(band, kpt, spin) / self.a0**1.5

    def GetEigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        result = self.get_eigenvalues(kpt, spin)
        if result is not None:
            return result * self.Ha

    def GetWannierLocalizationMatrix(self, nbands, dirG, kpoint,
                                     nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        c = dirG.index(1)
        kpts = self.GetBZKPoints()
        G = kpts[nextkpoint, c] - kpts[kpoint, c] + G_I[c]

        return self.get_wannier_integrals(c, spin, kpoint, nextkpoint, G)

    def GetMagneticMoment(self):
        """Return the magnetic moment."""
        return self.occupation.magmom

    def GetFermiLevel(self):
        """Return the Fermi-level."""
        e = self.occupation.get_fermi_level()
        if e is None:
            # Zero temperature calculation - return vacuum level:
            e = 0.0
        return e * self.Ha

    def GetGridSpacings(self):
        return self.a0 * self.gd.h_c

    def GetNumberOfGridPoints(self):
        return self.gd.N_c

    def GetEnsembleCoefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.GetPotentialEnergy()
        E0 = self.get_xc_difference('XC-9-1.0')
        coefs = (E + E0,
                 self.get_xc_difference('XC-0-1.0') - E0,
                 self.get_xc_difference('XC-1-1.0') - E0,
                 self.get_xc_difference('XC-2-1.0') - E0)
        self.text('BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs)
        return num.array(coefs)

    def GetExactExchange(self):
        return self.get_exact_exchange()

    def GetElectronicTemperature(self):
        return self.kT * self.Ha
