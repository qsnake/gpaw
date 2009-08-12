"""This module contains helper classes for running simple calculations.

XXX this file should be renamed to something else!!!

The main user of the Runner classes defined in this module is the
``gpaw`` command-line tool.  """

import os

import numpy as np
from ase.atoms import Atoms, string2symbols
from ase.utils.eos import *
from ase.calculators.emt import EMT
from ase.io.trajectory import PickleTrajectory
from ase.io import read
import ase.units as units
from ase.data import covalent_radii

from gpaw.aseinterface import GPAW
from gpaw.mpi import world

# Magnetic moments of isolated atoms:
magmom = {'C': 2, 'N': 3, 'Pt': 2, 'F': 1, 'Mg': 0, 'Na': 1, 'Cl': 1, 'Al': 1,
          'O': 2, 'Li': 1, 'P': 3, 'Si': 2, 'Cu': 1, 'Fe': 4}

class Runner:
    """Base class for running the calculations.

    Subclasses must implement set_calculator() method."""
    
    def __init__(self, name, atoms, strains=None, tag='', clean=False):
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
        
        self.name = name
        self.atoms = atoms

        if strains is None:
            strains = [1.0]
        self.strains = np.array(strains)
        
        if tag:
            self.tag = '-' + tag
        else:
            self.tag = ''

        self.clean = clean
        
        self.volumes = None
        self.energies = None
        self.atomic_energies = {}  # for calculating atomization energy
        
    def run(self):
        """Start calculation or read results from file."""
        filename = '%s%s.traj' % (self.name, self.tag)
        if self.clean or not os.path.isfile(filename):
            world.barrier()
            if world.rank == 0:
                open(filename, 'w')
            print 'Calculating', self.name, '...'
            self.calculate(filename)
        else:
            try:
                print 'Reading', filename,
                configs = read(filename, ':')
            except IOError:
                print 'FAILED'
            else:
                print
                if len(configs) == len(self.strains):
                    # Extract volumes and energies:
                    self.volumes = [a.get_volume() for a in configs]
                    self.energies = [a.get_potential_energy() for a in configs]

    def atomize(self):
        """Start calculations for atoms or read results from file."""
        for symbol in self.atoms.get_chemical_symbols():
            if symbol not in self.atomic_energies:
                filename = '%s%s-atom.traj' % (symbol, self.tag)
                if self.clean or not os.path.isfile(filename):
                    world.barrier()
                    if world.rank == 0:
                        open(filename, 'w')
                    print 'Calculating', symbol, 'atom ...'
                    self.calculate_isolated_atom(symbol, filename)
                else:
                    try:
                        print 'Reading', filename,
                        atom = read(filename)
                    except IOError:
                        print 'FAILED'
                    else:
                        print
                        e = atom.get_potential_energy()
                        self.atomic_energies[symbol] = e
                
    def calculate(self, filename):
        """Run calculation and write results to file."""
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
    
    def calculate_isolated_atom(self, symbol, filename):
        """Run atomic calculations and write results to files."""
        atom = Atoms(symbol, magmoms=[magmom.get(symbol, 0)])
        self.set_calculator(atom, filename)
        traj = PickleTrajectory(filename, 'w')
        e = atom.get_potential_energy()
        self.atomic_energies[symbol] = e
        traj.write(atom)
        return atom
    
    def summary(self, plot=False, a0=None):
        natoms = len(self.atoms)
        e = None
        if self.energies and len(self.energies) > 1:
            eos = EquationOfState(self.volumes, self.energies)
            v, e, B = eos.fit()
            a = a0 * (v / self.atoms.get_volume())**(1.0 / 3)
            print 'Fit using %d points:' % len(self.energies)
            print 'Volume per atom: %.3f Ang^3' % (v / natoms)
            print 'Lattice constant: %.3f Ang' % a
            print 'Bulk modulus: %.1f GPa' % (B * 1e24 / units.kJ)
        elif self.energies and len(self.energies) == 1:
            e = self.energies[0]

        if e is not None:
            print ('Total energy: %.3f eV (%d atom%s)' %
                   (e, natoms, ' s'[1:natoms]))

            
        for symbol, ea in self.atomic_energies.items():
            print 'Energy of %s atom: %.3f eV' % (symbol, ea)
            
        if e is not None:
            ec = -e
            for symbol in self.atoms.get_chemical_symbols():
                if symbol in self.atomic_energies:
                    ec += self.atomic_energies[symbol]
                else:
                    ec = None
                    break
                
            if ec is not None:
                print 'Cohesive energy: %.3f eV' % (ec / natoms)

            if plot:
                import pylab as plt
                plt.plot(self.volumes, self.energies, 'o')
                x = np.linspace(self.volumes[0], self.volumes[-1], 50)
                plt.plot(x, eos.fit0(x**-(1.0 / 3)), '-r')
                plt.show()


class EMTRunner(Runner):
    """EMT implementation"""
    def set_calculator(self, config, filename):
        config.set_calculator(EMT())


class GPAWRunner(Runner):
    """GPAW implementation"""
    def set_parameters(self, mode='fd', basis=None, kpts=None,
                       h=None, xc='LDA', vacuum=4.0):
        self.mode = mode

        if basis is None:
            basis = {}
        self.basis = basis
        
        self.kpts = kpts
        self.h = h
        self.xc = xc
        self.vacuum = vacuum
    
    def set_calculator(self, config, filename):
        if config.pbc.any():
            # Bulk calculation:
            L_c = (config.cell**2).sum(1)**0.5
            d_c = self.h * (L_c.prod() / config.get_volume())**(1.0 / 3)
            gpts = (L_c / d_c / 4).round().astype(int) * 4
            kwargs = dict(kpts=self.kpts, gpts=gpts, txt=filename[:-4] + 'txt')
        else:
            # Isolated atom or molecule:
            hund = (len(config) == 1)
            kwargs = dict(hund=hund, width=0.02, txt=filename[:-4] + 'txt')
            config.center(vacuum=self.vacuum)

        calc = GPAW(mode=self.mode,
                    basis=self.basis,
                    stencils=(3, 3),
                    xc=self.xc,
                    **kwargs)
        config.set_calculator(calc)

    def calculate(self, filename):
        config = Runner.calculate(self, filename)
        self.check_occupation_numbers(config)

    def calculate_isolated_atom(self, symbol, filename):
        atom = Runner.calculate_isolated_atom(self, symbol, filename)
        self.check_occupation_numbers(atom)

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
        covera = np.sqrt(8.0 / 3)
        
    if a is None:
        a = estimate_lattice_constant(name, crystalstructure, covera)

    x = crystalstructure.lower()

    if orthorhombic:
        return orthorhombic_bulk(name, x, a, covera)
    
    if x == 'fcc':
        b = a / 2
        atoms = Atoms(name, cell=[(0, b, b), (b, 0, b), (b, b, 0)], pbc=True)
    elif x == 'bcc':
        b = a / 2
        atoms = Atoms(name, cell=[(-b, b, b), (b, -b, b), (b, b, -b)],
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
    return atoms
