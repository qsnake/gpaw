import time
import sys
import pickle
import traceback

import numpy as npy
from numpy.linalg import inv

from gpaw import Calculator
from gpaw.utilities import devnull
from gpaw.testing import calc, g2, atomization_data


"""
Contains various Test classes ("badness functions") to be used by the gpaw 
setup optimizer
"""

nullpickler = pickle.Pickler(devnull)

class Test:
    """
    Base class of setup badness tests.

    An actual test should override the badness method. Presently a
    test needs not be a subclass of Test, it needs only implement a
    badness method with appropriate signature.
    """

    def __init__(self, name=None):
        # Any relevant data may be dumped into this dictionary when the 
        # badness function is called.  Just overwrite entries each time.
        self.dumplog = {}
        if name is None:
            name = self.__class__.__name__
        self.name = name
            
    def run(self, symbol, setup, out=devnull):
        """Print info and invoke self.badness."""
        msg = 'Test: %s' % str(self)
        print >> out
        print >> out, msg
        print >> out, '=' * len(msg)
        time1 = time.clock()
        rtime1 = time.time()
        badness = self.badness(symbol, setup, out)
        time2 = time.clock()
        rtime2 = time.time()
        print >> out, 'Badness:', badness
        print >> out, 'CPU Time:', time2 - time1
        print >> out, 'Realtime:', rtime2 - rtime1
        return badness

    def __str__(self):
        return self.name

    def badness(self, symbol, setup, out=devnull):
        """
        This method should be overridden by subclasses.  The method will
        be invoked automatically by an Optimizer using this Test and
        should return the badness of the given setup.

        Parameters:
    
          out : a file-like object which may be used to write output.
      
          symbol : the chemical symbol of the element being tested.
      
          setup : the name of the setup which is currently being
          optimized.  This variable can be suppleid to the constructor of
          a Calculator object.

        Returns:

          The badness of the setup in question. The default implementation
          returns 10000.
      
        """
        print >> out, 'Badness function not implemented!'
        return 10000.
        
default_badness_units = (0.05, # atomization energy deviation, eV
                         0.005, # bond length deviation, A
                         0.05, # egg box effect amplitude, eV
                         0.35, # h-convergence energy deviation, eV
                         1.) # (for iteration test - no effect except if None)

class MoleculeTest(Test):
    def __init__(self, a_big=6.5, a_small=4.5, 
                 unit_badness=default_badness_units):
        Test.__init__(self)
        assert len(unit_badness) == 5
        h = .25
        energytest = EnergyTest(a_big, h)
        distancetest = DistanceTest(a_big, h)
        eggboxtest = EggboxTest(a_small, h)
        convergencetest = ConvergenceTest(a_small)
        all_tests = [energytest, distancetest, eggboxtest, convergencetest]
        self.tests = []
        self.unit_badness = []
        for unit, test in zip(unit_badness, all_tests):
            if unit: # if unit is e.g. None, don't perform the test
                self.tests.append(test)
                self.unit_badness.append(unit)
        
        # the iteration count is only relevant for some of the tests
        iterationtests = []
        if unit_badness[0]:
            iterationtests.append(energytest)
        if unit_badness[1]:
            iterationtests.append(distancetest)
        self.iterationtest = IterationTest(iterationtests)

    def badness(self, symbol, setup, out=devnull):
        badnesses = []
        for unit, test in zip(self.unit_badness, self.tests):
            partial_badness = test.run(symbol, setup, out)
            badnesses.append(partial_badness / unit ** 2.)
        bniter = self.iterationtest.run(symbol, setup, out)
        badness = sum(badnesses) * bniter
        print >> out, 'Done'
        return badness


class IterationTest(Test):
    def __init__(self, tests):
        Test.__init__(self)
        self.tests = tests

    def badness(self, symbol, setup, out=devnull):
        if len(self.tests) == 0:
            return 1.
        niter = 0
        count = 0
        for test in self.tests:
            print test,':',test.niter
            count += len(test.niter)
            niter += sum(test.niter)
        print 'Number of calculations:', count
        print 'Total iterations:', niter
        return niter / 24. / count# something which is not too far from 1

def get_reference_energy(formula):
    energy = atomization_data.atomization[formula][2] # 2 -> PBE
    return - energy * 43.364e-3

class DistanceTest(Test):
    def __init__(self, a, h):
        Test.__init__(self)
        self.a = a
        self.h = h
        self.niter = None

    def badness(self, symbol, setup, out=devnull):
        formula = symbol + '2'
        reference_energy = get_reference_energy(formula)
        calculator = Calculator(xc='PBE', txt=None, setups=setup, h=self.h)
        system = g2.get_g2(formula, (self.a,)*3)
        system.set_calculator(calculator)
        original_positions = system.positions.copy()
        displ = original_positions[1] - original_positions[0]
        actual_bondlength = npy.dot(displ, displ) ** .5
        energies = []
        displacements = []
        self.niter = []
        for i in [-1, 0, 1]:
            amount = i * .03 * actual_bondlength
            system.set_positions([original_positions[0] - displ/2. * amount,
                                  original_positions[1] + displ/2. * amount])
            system.center()
            energy = system.get_potential_energy()
            displacements.append(amount)
            energies.append(energy)
            self.niter.append(calculator.niter)
        print >> out, 'Energies  :', ' '.join(['%8.5f' % e for e in energies])
        print >> out, 'Distance  :', ' '.join(['%8.5f' % d for d 
                                              in displacements])
        error, e_molecule, coefs = calc.interpolate(displacements, energies)
        print >> out, 'Error     :', error
        print >> out, 'Relaxed E :', e_molecule
        print >> out, 'Iter      :', self.niter
        return error ** 2
        

class EnergyTest(Test):
    def __init__(self, a, h):
        Test.__init__(self)
        self.a = a
        self.h = h
        self.niter = None

    def badness(self, symbol, setup, out=devnull):
        formula = symbol + '2'
        reference_energy = get_reference_energy(formula)
        a = self.a
        atom = g2.get_g2(symbol, (a,a,a))
        acalc = Calculator(h=self.h, xc='PBE', hund=True, setups=setup,
                           txt=None)
        atom.set_calculator(acalc)
        e_atom = atom.get_potential_energy()
        molecule = g2.get_g2(formula, (a,a,a))
        mcalc = Calculator(h=self.h, xc='PBE', setups=setup, txt=None)
        molecule.set_calculator(mcalc)
        e_molecule = molecule.get_potential_energy()
        e_atomization = e_molecule - 2 * e_atom
        self.niter = [acalc.niter, mcalc.niter]
        deviation = reference_energy - e_atomization
        print >> out, 'Energy     :', e_atomization
        print >> out, '  Molecule :', e_molecule
        print >> out, '  Atomic   :', e_atom
        print >> out, '  Error    :', deviation
        print >> out, 'Iterations :', self.niter
        return deviation ** 2.


class EggboxTest(Test):
    def __init__(self, a, h):
        Test.__init__(self)
        self.a = a
        self.h = h

    def badness(self, symbol, setup, out=devnull):
        formula = symbol + '2'
        system = g2.get_g2(formula, (self.a,)*3)
        system.set_pbc(1)
        calculator = Calculator(xc='PBE', txt=None, setups=setup, h=self.h)
        system.set_calculator(calculator)
        displacement_vector = npy.array([1.,1.,1.])/3.**.5
        original_positions = system.positions.copy()
        energies = []
        displacements = npy.linspace(0., self.h/2, 6)
        for dx in displacements:
            system.set_positions(original_positions + displacement_vector * dx)
            energy = system.get_potential_energy()
            energies.append(energy)
        fluctuation = max(energies) - min(energies)
        print >> out, 'Energies   :', ' '.join(['%8.5f' % e for e in energies])
        print >> out, 'Difference :', fluctuation
        return fluctuation ** 2.


class ConvergenceTest(Test):
    def __init__(self, a):
        Test.__init__(self)
        self.a = a

    def badness(self, symbol, setup, out=devnull):
        formula = symbol + '2'
        hvalues = [.2, .17, .14]
        system = g2.get_g2(formula, (self.a,)*3)
        system.set_pbc(1)
        energies = []
        for h in hvalues:
            calculator = Calculator(h=h, xc='PBE', txt=None, setups=setup)
            system.set_calculator(calculator)
            energy = system.get_potential_energy()
            energies.append(energy)
        fluctuation = max(energies) - min(energies)
        print >> out, 'Energies', ' '.join(['%8.5f' % e for e in energies])
        print >> out, 'Difference', fluctuation
        return fluctuation ** 2
