"""This module contains helper classes for running simple calculations.

XXX this file should be renamed to something else!!!

The main user of the Runner classes defined in this module is the
``gpaw`` command-line tool.  """

import os
import sys
from math import sqrt

import numpy as np
from ase.atoms import Atoms, string2symbols
from ase.utils.eos import EquationOfState
from ase.calculators.emt import EMT
from ase.io.trajectory import PickleTrajectory
from ase.io import read
import ase.units as units
from ase.data import covalent_radii

from gpaw.aseinterface import GPAW
from gpaw.poisson import PoissonSolver
from gpaw.mpi import world
from gpaw.utilities import devnull, h2gpts
from gpaw.occupations import FermiDirac

# Magnetic moments of isolated atoms:
magmom = {'C': 2, 'N': 3, 'Pt': 2, 'F': 1, 'Mg': 0, 'Na': 1, 'Cl': 1, 'Al': 1,
          'O': 2, 'Li': 1, 'P': 3, 'Si': 2, 'Cu': 1, 'Fe': 4}

class Runner:
    """Base class for running the calculations.

    Subclasses must implement a set_calculator() method."""
    
    def __init__(self, name, atoms, strains=None, tag='', clean=False,
                 out='-'):
        """Construct runner object.

        Results will be written to trajectory files or read from those
        files if the already exist.
        
        name: str
            Name of calculation.
        atoms: Atoms object
            configuration to work on.
        strains: list of floats
            The list of strains to apply to the unit cell or bond length.
            Defaults to [1.0].
        tag: str
            Tag used for filenames like <name>-<tag>.traj.
        clean: bool
            Do *not* read results from files.
        """
        
        if strains is None:
            strains = [1.0]

        if tag:
            self.tag = '-' + tag
        else:
            self.tag = ''

        if world.rank == 0:
            if out is None:
                out = devnull
            elif isinstance(out, str):
                if out == '-':
                    out = sys.stdout
                else:
                    out = open(out, 'w')
        else:
            out = devnull
            
        self.name = name
        self.atoms = atoms
        self.strains = np.array(strains)
        self.clean = clean
        self.out = out
        
        self.volumes = None
        self.energies = None
        self.atomic_energies = {}  # for calculating atomization energy

    def log(self, *args, **kwargs):
        self.out.write(kwargs.get('sep', ' ').join([str(arg)
                                                    for arg in args]) +
                       kwargs.get('end', '\n'))

    def run(self):
        """Start calculation or read results from file."""
        filename = '%s%s.traj' % (self.name, self.tag)
        if self.clean or not os.path.isfile(filename):
            world.barrier()
            if world.rank == 0:
                open(filename, 'w')
            self.calculate(filename)
        else:
            try:
                self.log('Reading', filename, end=' ')
                configs = read(filename, ':')
            except IOError:
                self.log('FAILED')
                self.calculate(filename)
            else:
                self.log()
                if len(configs) == len(self.strains):
                    # Extract volumes and energies:
                    self.volumes = [a.get_volume() for a in configs]
                    self.energies = [a.get_potential_energy() for a in configs]

    def calculate(self, filename):
        """Run calculation and write results to file."""
        self.log('Calculating', self.name, '...')
        config = self.atoms.copy()
        self.set_calculator(config, filename)
        traj = PickleTrajectory(filename, 'w')
        cell = config.get_cell()
        self.volumes = []
        self.energies = []
        for strain in self.strains:
            config.set_cell(strain * cell, scale_atoms=True)
            self.volumes.append(config.get_volume())
            e = config.get_potential_energy()
            self.energies.append(e)
            traj.write(config)
        return config
    
    def summary(self, plot=False, a0=None):
        natoms = len(self.atoms)
        e = v = B = ec = a = None
        if self.energies and len(self.energies) > 1:
            eos = EquationOfState(self.volumes, self.energies)
            v, e, B = eos.fit()
            a = a0 * (v / self.atoms.get_volume())**(1.0 / 3)
            self.log('Fit using %d points:' % len(self.energies))
            self.log('Volume per atom: %.3f Ang^3' % (v / natoms))
            self.log('Lattice constant: %.3f Ang' % a)
            self.log('Bulk modulus: %.1f GPa' % (B * 1e24 / units.kJ))
        elif self.energies and len(self.energies) == 1:
            e = self.energies[0]

        if e is not None:
            self.log('Total energy: %.3f eV (%d atom%s)' %
                     (e, natoms, ' s'[1:natoms]))
            
        if e is not None:
            if plot:
                import pylab as plt
                plt.plot(self.volumes, self.energies, 'o')
                x = np.linspace(self.volumes[0], self.volumes[-1], 50)
                plt.plot(x, eos.fit0(x**-(1.0 / 3)), '-r')
                plt.show()

        return e, v, B, a

class EMTRunner(Runner):
    """EMT implementation"""
    def set_calculator(self, config, filename):
        config.set_calculator(EMT())


class GPAWRunner(Runner):
    """GPAW implementation"""
    def set_parameters(self, vacuum=3.0, **kwargs):
        self.vacuum = vacuum
        self.input_parameters = kwargs
        
    def set_calculator(self, config, filename):
        kwargs = {}
        kwargs.update(self.input_parameters)

        # Use fixed number of gpts:
        h = kwargs.get('h', 0.2 / units.Bohr)
        gpts = h2gpts(h, config.cell)
        kwargs['h'] = None
        kwargs['gpts'] = gpts
        
        if 'txt' not in kwargs:
            kwargs['txt'] = self.name + '.txt'
        
        if not config.pbc.any():
            # Isolated atom or molecule:
            config.center(vacuum=self.vacuum)
            if (len(config) == 1 and
                config.get_initial_magnetic_moments().any()):
                kwargs['hund'] = True
                
        calc = GPAW(**kwargs)
        config.set_calculator(calc)

    def check_occupation_numbers(self, config):
        """Check that occupation numbers are integers."""
        if config.pbc.any():
            return
        calc = config.get_calculator()
        nspins = calc.get_number_of_spins()
        for s in range(nspins):
            f = calc.get_occupation_numbers(spin=s)
            if abs(f % (2 // nspins)).max() > 0.0001:
                raise RuntimeError('Fractional occupation numbers?!')


# XXX This should be moved to ASE!!!
def bulk(name, crystalstructure, a=None, covera=None, orthorhombic=False):
    """Helper function for creating bulk systems.

    name: str
        Chemical symbol or symbols as in 'MgO' or 'NaCl'.
    crystalstructure: str
        Must be one of sc, fcc, bcc, hcp, diamond, zinkblende or
        rocksalt.
    a: float
        Lattice constant.
    covera: float
        c/a raitio used for hcp.  Defaults to ideal ratio.
    orthorhombic: bool
        Construct orthorhombic unit cell instead of primitive cell
        which is the default.
    """

    if covera is None:
        covera = sqrt(8.0 / 3.0)
        
    if a is None:
        a = estimate_lattice_constant(name, crystalstructure, covera)

    x = crystalstructure.lower()

    if orthorhombic and x != 'sc':
        return orthorhombic_bulk(name, x, a, covera)
    
    if x == 'sc':
        atoms = Atoms(name, cell=(a, a, a), pbc=True)
    elif x == 'fcc':
        b = a / 2
        atoms = Atoms(name, cell=[(0, b, b), (b, 0, b), (b, b, 0)], pbc=True)
    elif x == 'bcc':
        b = a / 2
        atoms = Atoms(name, cell=[(-b, b, b), (b, -b, b), (b, b, -b)],
                      pbc=True)
    elif x == 'hcp':
        atoms = Atoms(2 * name,
                      scaled_positions=[(0, 0, 0),
                                        (1.0 / 3.0, 1.0 / 3.0, 0.5)],
                      cell=[(a, 0, 0),
                            (a / 2, a * sqrt(3) / 2, 0),
                            (0, 0, covera * a)],
                      pbc=True)
    elif x == 'diamond':
        atoms = bulk(2 * name, 'zincblende', a)
    elif x == 'zincblende':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a)
        atoms.positions[1] += a / 4
    elif x == 'rocksalt':
        s1, s2 = string2symbols(name)
        atoms = bulk(s1, 'fcc', a) + bulk(s2, 'fcc', a)
        atoms.positions[1, 0] += a / 2
    else:
        raise ValueError('Unknown crystal structure: ' + crystalstructure)
    
    return atoms

def estimate_lattice_constant(name, crystalstructure, covera):
    atoms = bulk(name, crystalstructure, 1.0, covera)
    v0 = atoms.get_volume()
    v = 0.0
    for Z in atoms.get_atomic_numbers():
        r = covalent_radii[Z]
        v += 4 * np.pi / 3 * r**3 * 1.5
    return (v / v0)**(1.0 / 3)

def orthorhombic_bulk(name, x, a, covera=None):
    if x == 'fcc':
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif x == 'bcc':
        atoms = Atoms(2 * name, cell=(a, a, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif x == 'hcp':
        atoms = Atoms(4 * name,
                      cell=(a, a * sqrt(3), covera * a),
                      scaled_positions=[(0, 0, 0),
                                        (0.5, 0.5, 0),
                                        (0.5, 1.0 / 6.0, 0.5),
                                        (0, 2.0 / 3.0, 0.5)],
                      pbc=True)
    elif x == 'diamond':
        atoms = orthorhombic_bulk(2 * name, 'zincblende', a)
    elif x == 'zincblende':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0, 0.25),
                                        (0.5, 0.5, 0.5), (0, 0.5, 0.75)])
    elif x == 'rocksalt':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0),
                                        (0.5, 0.5, 0.5), (0, 0, 0.5)])
    else:
        raise RuntimeError
    
    return atoms
