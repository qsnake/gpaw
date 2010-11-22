import numpy as np

from ase import Atoms
from ase.units import Bohr
from gpaw.density import Density
from gpaw.lfc import BasisFunctions
from gpaw.mixer import Mixer
from gpaw.setup import Setups
from gpaw.xc import XC
from gpaw.utilities.tools import coordinates

from gpaw.mpi import rank

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

        density = self.calculator.density
        density.set_positions(all_atoms.get_scaled_positions() % 1.0,
                              self.calculator.wfs.rank_a)

        # select atoms
        atoms = []
        D_asp = {}
        all_D_asp = self.calculator.density.D_asp
        for a in atom_indicees:
            if a in all_D_asp:
                D_asp[len(atoms)] = all_D_asp.get(a)
            atoms.append(all_atoms[a])
        atoms = Atoms(atoms, cell=all_atoms.get_cell())
        spos_ac = atoms.get_scaled_positions() % 1.0
        Z_a = atoms.get_atomic_numbers()

        par = self.calculator.input_parameters
        setups = Setups(Z_a, par.setups, par.basis, par.lmax, 
                        XC(par.xc), 
                        self.calculator.wfs.world)
        self.D_asp = D_asp

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

        aed_sg, gd = Density.get_all_electron_density(self, atoms, 
                                                      gridrefinement=2)

        return aed_sg[0], gd

class HirshfeldPartitioning:
    """Partion space according to the Hirshfeld method.

    After: F. L. Hirshfeld Theoret. Chim.Acta 44 (1977) 129-138
    """
    def __init__(self, calculator, density_cutoff=1.e-12):
        self.calculator = calculator
        self.atoms = calculator.get_atoms()
        self.hdensity = HirshfeldDensity(calculator)
        density_g, gd = self.hdensity.get_density()
        self.invweight_g = np.where(density_g > density_cutoff, 
                                    1.0 /  density_g, 0.0)
    
    def get_calculator(self):
        return self.calculator
    
    def get_effective_volume_ratio(self, atom_index):
        """Effective volume to free volume ratio.

        After: Tkatchenko and Scheffler PRL 102 (2009) 073005
        """
        atoms = self.atoms
        finegd = self.calculator.density.finegd

        den_g, gd = self.calculator.density.get_all_electron_density(atoms)
        assert(gd == finegd)
        denfree_g, gd = self.hdensity.get_density([atom_index])
        assert(gd == finegd)

        # the atoms r^3 grid
        position = self.atoms[atom_index].position / Bohr
        r_vg, r2_g = coordinates(finegd, origin=position)
        r3_g = r2_g * np.sqrt(r2_g)

        weight_g = denfree_g * self.invweight_g

        nom = finegd.integrate(r3_g * den_g[0] * weight_g)
        denom = finegd.integrate(r3_g * denfree_g)

        return nom / denom

    def get_effective_volume_ratios(self):
        """Return the list of effective volume to free volume ratios."""
        ratios = []
        for a, atom in enumerate(self.atoms):
            ratios.append(self.get_effective_volume_ratio(a))
        return np.array(ratios)

         
