import numpy as np

from ase import Atoms
from ase.units import Bohr
from gpaw.density import Density
from gpaw.lfc import BasisFunctions
from gpaw.mixer import Mixer
from gpaw.setup import Setups
from gpaw.xc import XC
from gpaw.utilities.tools import coordinates

class HirshfeldDensity(Density):
    """Density as sum of atomic densities."""
    def __init__(self, calculator):
        self.calculator = calculator
        density = calculator.density
        Density.__init__(self, density.gd, density.finegd, 1, 0)

    def get_density(self, atom_indicees=None):
        """Get sum of atomic densities from the given atom list.

        All atoms are taken if the list is not given."""
 
        all_atoms = self.calculator.get_atoms()
        if atom_indicees is None:
            atom_indicees = range(len(all_atoms))

        # select atoms
        par = self.calculator.input_parameters
        atoms = []
        Z_a = []
        spos_ac = []
        all_spos_ac = all_atoms.get_scaled_positions() % 1.0
        all_Z_a = all_atoms.get_atomic_numbers()
        for a in atom_indicees:
            atoms.append(all_atoms[a])
            spos_ac.append(all_spos_ac[a])
            Z_a.append(all_Z_a[a])
        atoms = Atoms(atoms, cell=all_atoms.get_cell())
        setups = Setups(Z_a, par.setups, par.basis, par.lmax, 
                        XC(par.xc), 
                        self.calculator.wfs.world)

        # initialize 
        self.initialize(setups, 
                        par.stencils[1], 
                        self.calculator.timer,
                        [0] * len(atoms), False)
        self.set_mixer(None)
        self.set_positions(spos_ac, self.calculator.wfs.rank_a)
        basis_functions = BasisFunctions(self.gd,
                                         [setup.phit_j
                                          for setup in self.setups],
                                         cut=True)
        basis_functions.set_positions(spos_ac)
        self.initialize_from_atomic_densities(basis_functions)
        
        return Density.get_all_electron_density(self, atoms)

class HirshfeldPartitioning:
    """Partion space according to the Hirshfeld method.

    After: F. L. Hirshfeld Theoret. Chim.Acta 44 (1977) 129-138
    """
    def __init__(self, calculator, density_cutoff=1.e-12):
        self.calculator = calculator
        self.atoms = calculator.get_atoms()
        self.hdensity = HirshfeldDensity(calculator)
        density_g = self.hdensity.get_density()[0][0]
        self.invweight_g = np.where(density_g > density_cutoff, 
                                    1.0 /  density_g, 0.0)
        
    def get_effective_volume_ratio(self, atom_index):
        """Effective volume to free volume ratio.

        After: Tkatchenko and Scheffler PRL 102 (2009) 073005
        """
        atoms = self.atoms
        den_g = self.calculator.density.get_all_electron_density(atoms)[0][0]
        denfree_g = self.hdensity.get_density([atom_index])[0][0]

        # my r^3 grid
        finegd = self.calculator.density.finegd
        position = self.atoms[atom_index].position / Bohr
        r_vg, r2_g = coordinates(finegd, origin=position)
        r3_g = r2_g * np.sqrt(r2_g)

        weight_g = denfree_g * self.invweight_g

        nom = finegd.integrate(r3_g * den_g * weight_g)
        denom = finegd.integrate(r3_g * denfree_g)

        return nom / denom

    def get_effective_volume_ratios(self):
        ratios = []
        for a, atom in enumerate(self.atoms):
            ratios.append(self.get_effective_volume_ratio(a))
        return np.array(ratios)

         
