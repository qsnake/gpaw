# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

import os
import weakref

from ASE.Units import units, Convert
import ASE

from gpaw.paw import PAW

        
class Calculator(PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation.
    """

    def __init__(self, filename=None, **kwargs):
        # Set units to ASE units:
        lengthunit = units.GetLengthUnit()
        energyunit = units.GetEnergyUnit()
        self.a0 = Convert(1, 'Bohr', lengthunit)
        self.Ha = Convert(1, 'Hartree', energyunit)

        self.convert_units(kwargs)
        PAW.__init__(self, filename, **kwargs)

        self.text('ASE: ', os.path.dirname(ASE.__file__))
        self.text('units:', lengthunit, 'and', energyunit)

    def convert_units(self, parameters):
        if parameters.get('h') is not None:
            parameters['h'] /= self.a0
        if parameters.get('width') is not None:
            parameters['width'] /= self.Ha
        if parameters.get('external') is not None:
            parameters['external'] = parameter['external'] / self.Ha
        
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
        return self.paw.nspins == 2
    
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

    def GetWaveFunctionArray(self, band=0, kpt=0, spin=0):
        """Return pseudo-wave-function array."""
        c =  1.0 / self.a0**1.5
        return self.get_wave_function_array(band, kpt, spin) * c

    def GetEigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        return self.get_eigenvalues(kpt, spin) * self.Ha

    def GetWannierLocalizationMatrix(self, G_I, kpoint, nextkpoint, spin,
                                     dirG, **args):
        """Calculate integrals for maximally localized Wannier functions."""

        c = dirG.index(1)
        return self.get_wannier_integrals(c, spin, kpoint, nextkpoint, G_I)

    def GetMagneticMoment(self):
        """Return the magnetic moment."""
        return self.occupation.magmom

    def GetFermiLevel(self):
        """Return the Fermi-level."""
        return self.occupation.get_fermi_level()

    def GetElectronicStates(self):
        """Return electronic-state object."""
        from ASE.Utilities.ElectronicStates import ElectronicStates
        self.write('tmp27.nc', 'all')
        return ElectronicStates('tmp27.nc')
    
