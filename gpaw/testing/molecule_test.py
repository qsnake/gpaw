# -*- coding: utf-8 -*-
import sys
import pickle
import traceback
import os.path as path

from ase.data.molecules import data, atoms, latex, molecule
from ase.atoms import string2symbols
from ase.parallel import paropen
from ase.parallel import rank, barrier
from ase.io.trajectory import PickleTrajectory
from ase.units import kcal, mol
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
except ImportError:
    pass

from gpaw import GPAW, restart, ConvergenceError
from gpaw.testing.atomization_data import atomization_vasp, diatomic

dimers = diatomic.keys()
dimers.remove('FH')
molecules = atomization_vasp.keys()
systems = molecules + atoms


def atomization_energies(E):
    """Write given atomization energies to file atomization_energies.csv."""
    Ea = {}
    fd = open('atomization_energies.csv', 'w')
    for formula in sorted(molecules):
        try:
            ea = -E[formula]
            for a in string2symbols(data[formula]['symbols']):
                ea += E[a]
            eavasp = atomization_vasp[formula][1] * kcal / mol
            Ea[formula] = (ea, eavasp)
            name = latex(data[formula]['name'])
            fd.write('`%s`, %.3f, %.3f, %+.3f\n' %
                     (name[1:-1], ea, eavasp, ea - eavasp))
        except KeyError:
            pass # Happens if required formula or atoms are not in E
    return Ea


def bondlengths(Ea, dE):
    """Calculate bond lengths and write to bondlengths.csv file"""
    B = []
    E0 = []
    csv = open('bondlengths.csv', 'w')
    for formula, energies in dE:
        bref = diatomic[formula][1]
        b = np.linspace(0.96 * bref, 1.04 * bref, 5)
        e = np.polyfit(b, energies, 3)
        if not formula in Ea:
            continue
        ea, eavasp = Ea[formula]
        dedb = np.polyder(e, 1)
        b0 = np.roots(dedb)[1]
        assert abs(b0 - bref) < 0.1
        b = np.linspace(0.96 * bref, 1.04 * bref, 20)
        e = np.polyval(e, b) - ea
        if formula == 'O2':
            plt.plot(b, e, '-', color='0.7', label='GPAW')
        else:
            plt.plot(b, e, '-', color='0.7', label='_nolegend_')
        name = latex(data[formula]['name'])
        plt.text(b[0], e[0] + 0.2, name)
        B.append(bref)
        E0.append(-eavasp)
        csv.write('`%s`, %.3f, %.3f, %+.3f\n' %
                  (name[1:-1], b0, bref, b0 - bref))
        
    plt.plot(B, E0, 'g.', label='reference')
    plt.legend(loc='lower right')
    plt.xlabel(u'Bond length [Å]')
    plt.ylabel('Energy [eV]')
    plt.savefig('bondlengths.png')

    
def read_and_check_results():
    """Read energies from .gpw files."""
    fd = sys.stdout
    E = {}
    fd.write('E = {')
    for formula in systems:
        try:
            atoms, calc = restart(formula, txt=None)
        except (KeyError, IOError):
            #print formula
            continue
    
        nspins = calc.get_number_of_spins()
        fa = calc.get_occupations(spin=0)
        assert ((fa.round() - fa)**2).sum() < 1e-14
        if nspins == 2:
            fb = calc.get_occupations(spin=1)
            assert ((fb.round() - fb)**2).sum() < 1e-9
            if len(atoms) == 1:
                M = data[formula]['magmom']
            else:
                M = sum(data[formula]['magmoms'])
            assert abs((fa-fb).sum() - M) < 1e-9
        e = calc.get_potential_energy()
        fd.write("'%s': %.3f, " % (formula, e))
        fd.flush()
        E[formula] = e

    dE = [] # or maybe {} ?
    fd.write('}\ndE = [')
    
    for formula in dimers:
        try:
            trajectory = PickleTrajectory(formula + '.traj', 'r')
        except IOError:
            continue
        energies = [a.get_potential_energy() for a in trajectory]
        dE.append((formula, (energies)))
        fd.write("('%s', (" % formula)
        fd.write(', '.join(['%.4f' % (energy - E[formula])
                            for energy in energies]))
        fd.write(')),\n      ')
    fd.write(']\n')
    return E, dE


class Test:
    def __init__(self, vacuum=6.0, h=0.16, xc='PBE', setups='paw',
                 eigensolver='rmm-diis', basis=None,
                 calculate_dimer_bond_lengths=True, txt=sys.stdout):
        self.vacuum = vacuum
        self.h = h
        self.xc = xc
        self.setups = setups
        self.eigensolver = eigensolver
        if basis is None:
            basis = {}
        self.basis = basis
        self.calculate_dimer_bond_lengths=calculate_dimer_bond_lengths
        if isinstance(txt, str):
            txt = open(txt + '.log', 'w')
        self.txt = txt

    def do_calculations(self, formulas):
        """Perform calculation on molecules, write results to .gpw files."""
        atoms = {}
        for formula in formulas:
            for symbol in string2symbols(formula.split('_')[0]):
                atoms[symbol] = None
        formulas = formulas + atoms.keys()

        for formula in formulas:
            if path.isfile(formula + '.gpw'):
                continue

            barrier()
            open(formula + '.gpw', 'w')
            s = molecule(formula)
            s.center(vacuum=self.vacuum)
            cell = s.get_cell()
            h = self.h
            s.set_cell((cell / (4 * h)).round() * 4 * h)
            s.center()
            calc = GPAW(h=h,
                        xc=self.xc,
                        eigensolver=self.eigensolver,
                        setups=self.setups,
                        basis=self.basis,
                        fixmom=True,
                        txt=formula + '.txt')

            if len(s) == 1:
                calc.set(hund=True)

            s.set_calculator(calc)

            if formula == 'BeH':
                calc.initialize(s)
                calc.nuclei[0].f_si = [(1, 0, 0.5, 0),
                                       (0.5, 0, 0, 0)]

            if formula in ['NO', 'ClO', 'CH']:
                s.positions[:, 1] += h * 1.5

            try:
                energy = s.get_potential_energy()
            except (RuntimeError, ConvergenceError):
                if rank == 0:
                    print >> sys.stderr, 'Error in', formula
                    traceback.print_exc(file=sys.stderr)
            else:
                print >> self.txt, formula, repr(energy)
                self.txt.flush()
                calc.write(formula)

            if formula in diatomic and self.calculate_dimer_bond_lengths:
                traj = PickleTrajectory(formula + '.traj', 'w')
                d = diatomic[formula][1]
                for x in range(-2, 3):
                    s.set_distance(0, 1, d * (1.0 + x * 0.02))
                    traj.write(s)
