#!/usr/bin/env python

"""
This file contains an Optimizer class, the purpose of which is to
optimize GPAW setups. The GPAW setup optimizer uses the downhill
simplex algorithm to search a parameter space of any dimension.
"""

import atomization, setupgenerator
import amoeba
import sys, traceback, pickle, random
from LinearAlgebra import inverse
#import numpy as N
import Numeric as N
#from gpaw.utilities.timing import Timer
import time

#from datetime import datetime, timedelta

"""
Returns a list of vertex coordinates forming a regular simplex around
the designated center, where the size argument is the max
vertex-center distance.

This method simply generates a random simplex, and may fail to do so
at a very small probability (if randomly generated vectors are
linearly dependent)
"""
def get_random_simplex(center=[0,0], size=.1, seed=0):
    ndim = len(center)
    mpts = ndim + 1
    r = random.Random(seed)

    points = [None]*mpts
    for i in range(ndim+1):
        points[i] = [(r.random()-.5)*size+center[j] for j in range(ndim)]
        
    return points

def get_simplex(center=[0,0], size=.1):

    points = [None]*(len(center)+1)
    #initialize all points to center
    points = [map(float,list(center)) for p in points]
    
    for i in range(1,len(points)):
        points[i][i-1] += float(size)

    return points

"""
This is used as a separator in the log file
"""
separator = '='*72

class Optimizer:

    def __init__(self, symbol='N', name='test', generator=None,
                 parametercount=None, out=None, quick=False,
                 test=None, simplex=None):
        """
        Creates an optimizer for the specified element which generates
        setups with the name [symbol].opt.[name].PBE

        out is the name of the logfile, which defaults to
        [symbol].opt.[name].log
        
        Generated setup files will be named [symbol].opt.[name].PBE .

        If the optional generator parameter is specified, it should be
        a callable which returns a SetupGenerator given an element
        symbol string and a name (as per the documentation of
        SetupGenerator).
        
        """

        self.element = atomization.elements[symbol]

        if test is None:
            test = MoleculeTest()
        self.test = test

        if generator is None:
            self.generator = setupgenerator.SetupGenerator(self.element.letter,
                                                           'opt.'+name)
        else:
            self.generator = generator(self.element.letter, 'opt.'+name)

        #simplex = None #perhaps we should generate a better simplex
        if simplex is None:
            params = self.generator.standard_parameters()[:parametercount]
            simplex = get_simplex(params)
            simplex[0] = list(params) #We'll want the starting point to be
            #used directly, so we'll just overwrite the first of the
            #random vertices
        self.simplex = simplex
        self.quick = quick

        self.setup_name = 'opt.'+name

        if out is None:
            self.output = open(name+'.opt.'+symbol+'.log', 'w')
            #Note: perhaps also make it possible to not write output
        elif out == '-':
            self.output = sys.stdout#open('/dev/null')
        else:
            self.output = open(output, 'w')

        out = self.output

        print >> out, 'Optimization run commencing'
        print >> out, separator
        print >> out, 'Name:',name
        print >> out, 'Element:',symbol
        #print >> out, 'Dump file:',dumpFile
        print >> out, 'Parameter space dimension:',(len(simplex)-1)
        #print >> out, 'Tolerance:',fTolerance
        print >> out, separator
        #Remember  to write initData header
        if quick:
            print >> out, '>>> This is a quick test! <<<'
        print >> out
        print >> out, 'Simplex points'
        for point in simplex:
            print >> out, '\t',point
        out.flush()
        print >> out
        print >> out, 'Evaluating simplex point badness values'
        out.flush()
            
        values = [self.badness(point) for point in simplex]
        
        print >> out, 'Simplex point evaluation complete.'
        print >> out
        out.flush()
        
        #If resuming from file, remember to make it possible to reuse badness
        #values
        self.amoeba = amoeba.Amoeba(simplex, values, self.badness)
        self.stepcount = 0

    """
    Placeholder until gpaw.utilities.Timer can be used
    """
    """def clock(self):
        return time.clock()

    def time(self):
        return time.time()"""

    def optimize(self, tolerance=0.001):
        self.amoeba.tolerance = tolerance
        out = self.output

        while self.amoeba.relativedeviation > tolerance:
            self.stepcount += 1
            self.amoeba.step()
            print >> out
            print >> out, separator
            print >> out, 'Badness :: Simplex point'
            for i,(p,y) in enumerate(zip(self.simplex, self.amoeba.y)):
                print >> out, '\t',y,'\t',p
            print >> out, 'Relative deviation : ',self.amoeba.relativedeviation
            print >> out, '# Optimization run with tolerance',tolerance
            print >> out, separator
            print >> out
            out.flush()

    """
    Runs a full test of a given GPAW setup
    """
    def badness(self, args):

        #ref_dist = self.element.d
        badness = 10000 #will be overwritten unless bad things happen
        #timer = self #change when gpaw.utilities.Timer works

        out = self.output

        try:
            if not self.quick:
                self.generator.generate_setup(args) #new setup

            print >> out, 'Point: ',args
            badness = self.test.badness(out, self.element.letter,
                                        self.setup_name)


            #energybadness = self.energytest.badness(self)

            #overallbadness = (energybadness + distancebadness +
            #noisebadness + convergencebadness)*iterationbadness

        except KeyboardInterrupt:
            raise KeyboardInterrupt #Don't ignore keyboard interrupts
        except:
            (cl, ex, stacktrace) = sys.exc_info()
            if ex.__dict__.has_key('args'):
                ex_args = ex.__dict__['args']
            else:
                ex_args = '<none>'
            
            print >> out, '=== EXCEPTION ==='
            print >> out, ''.join(traceback.format_tb(stacktrace))
            print >> out, '================='
            #    overallbadness = 10000.
        
        print >> out, 'Badness: ', badness
        print >> out
        out.flush()
        return badness


optimizer = None

"""
Base class of setup badness tests. An actual test should override the
badness method. Presently a test needs not be a subclass of Test, it needs
only implement a badness method with appropriate signature.
"""
class Test:
    def __init__(self):
        pass

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
    def badness(self, out, symbol, setup):
        print >> out, 'Badness function not implemented!'
        return 10000.

class MoleculeTest(Test):
    def __init__(self):
        Test.__init__(self)
        e = EnergyTest()
        d = DistanceTest()
        self.tests = [e, d, NoiseTest(),
                      ConvergenceTest()]
        self.iterationtest = IterationTest([e, d], ['Ea','d '], [1, 3])

    def badness(self, out, symbol, setup):
        starttime = time.clock()
        startwalltime = time.time()
        badnessvalues = [test.badness(out, symbol, setup)
                         for test in self.tests]
        (b_energy, b_dist, b_noise, b_conv) = tuple(badnessvalues)
        b_iter = self.iterationtest.badness(out, symbol, setup,
                                            starttime, startwalltime)
        badness = (b_energy + b_dist + b_noise + b_conv) * b_iter
        return badness

class EnergyTest(Test):
        
    def __init__(self, cellsize=6., unit_badness=.05):
        Test.__init__(self)
        self.cellsize = cellsize
        
        #self.element = atomization.elements[symbol]
        #self.ref_energy = self.element.Ea_PBE
        self.unit_badness = unit_badness
        self.iterationcount = None

    def badness(self, out, symbol, setup):
        #energy_badness_ = 1/.05**2 #badness == 1 for deviation == .05 eV
        
        molecule = atomization.elements[symbol]
        (c1,c2) = atomization.atomizationCalculators(out=None,setup=setup,
                                                     molecule=molecule)
        print >> out, '\tEnergy'
        out.flush()
        subtesttime1 = time.clock()
        #if self.quick:
        #    Ea = 42.
        #else:
        Ea = atomization.calcEnergy(c1,c2,a=self.cellsize,molecule=molecule)

        subtesttime2 = time.clock()

        #if self.quick:
        #    iterationcount_Ea = 42
        #else:
        self.iterationcount = (c1.GetNumberOfIterations()+
                               c2.GetNumberOfIterations())

        ref_energy = atomization.elements[symbol].Ea_PBE
        energybadness = ((Ea - ref_energy)/self.unit_badness)**2

        print >> out, '\t\tEa         : ', Ea
        print >> out, '\t\tTime       : ', (subtesttime2-subtesttime1)
        print >> out, '\t\tBadness    : ', energybadness
        print >> out, '\t\tIterations : ', self.iterationcount
        out.flush()
        return energybadness

    def get_last_iteration_count(self):
        return self.iterationcount
        
class DistanceTest(Test):
    
    def __init__(self, cellsize=5.5, unit_badness=.005):
        Test.__init__(self)
        #self.ref_dist = self.element.d
        self.unit_badness = unit_badness
        self.cellsize = cellsize
    
    def badness(self, out, symbol, setup):
        print >> out, '\tDistance'
        ref_dist = atomization.elements[symbol].d
        out.flush()
        subtesttime1 = time.clock()
        #if self.quick:
        #    (d,iterationcount_d) = (42., 42)
        #else:
        (d, self.iterationcount) = self.bondLength(symbol, setup)

        distancebadness = ((d - ref_dist)/self.unit_badness)**2
        subtesttime2 = time.clock()
        print >> out, '\t\tDistance   : ', d
        print >> out, '\t\tTime       : ', (subtesttime2-subtesttime1)
        print >> out, '\t\tBadness    : ', distancebadness
        print >> out, '\t\tIterations : ', self.iterationcount
        out.flush()

        return distancebadness

    def get_last_iteration_count(self):
        return self.iterationcount

    """
    Returns the bond length. Calculates energy at three locations
    around the reference bond length, interpolates with a 2nd degree
    polynomial and returns the minimum of this polynomial which would
    be roughly equal to the bond length without engaging in a large
    whole relaxation test
    """
    def bondLength(self, symbol, setup):
        element = atomization.elements[symbol]
        d0 = element.d
        #REMEMBER: find out what unit badness should be for other elements!
        dd = ( .2 / 140. )**.5 #around .04 A. Bond properties correspond to
        #an energy of E = .5 k x**2 with k = 140 eV/A**2
        #If we want .1 eV deviation then the above dd should be used
        calc = atomization.MakeCalculator(element.nbands2,
                                          out=None, setup=setup)

        D = [d0-dd, d0, d0+dd]
        #Calculate energies at the three points
        
        E = [atomization.energyAtDistance(d, calc=calc, a=self.cellsize,
                                          molecule=element) for d in D]
        #Now find parabola and determine minimum

        x = N.array(D)
        y = N.array(E)

        A = N.transpose(N.array([x**0, x**1, x**2]))
        c = N.dot(inverse(A), y)
        #c = N.dot(N.linalg.inv(A), y)
        #print 'Coordinates',c

        X = - c[1] / (2.*c[2]) # "-b/(2a)"
        #print 'Bond length',X

        return X, calc.GetNumberOfIterations()


class NoiseTest(Test):

    def __init__(self, unit_badness=.005, points=[(0.,0.,0.),(.35,.35,.35),
                                                  (.5,.5,.5)]):
        Test.__init__(self)
        self.unit_badness=unit_badness
        self.points = tuple(points)

    def badness(self, out, symbol, setup):
        print >> out, '\tFluctuation'
        out.flush()
        subtesttime1 = time.clock()

        #if self.quick:
        #    noise = 42
        #else:
        energies = self.energyFluctuationTest(symbol, setup)
        noise = max(energies) - min(energies)
        
        noisebadness = (noise/self.unit_badness)**2
        subtesttime2 = time.clock()
        print >> out, '\t\tFluctuation:', noise
        print >> out, '\t\tAll points :', energies
        print >> out, '\t\tTime       :', (subtesttime2-subtesttime1)
        print >> out, '\t\tBadness    :', noisebadness

        return noisebadness


    """
    Returns the difference between the energy of a nitrogen molecule
    at the center of the unit cell and the energy of one translated by
    h/2 along the z axis.
    """
    def energyFluctuationTest(self, symbol, setup):
        A = atomization
        element = A.elements[symbol]
        h = .2
        a = 4.
        #if self.quick:
        #    h = .3
        calc = A.MakeCalculator(element.nbands2, out=None, setup=setup,
                                h=h)
        d = element.d
        #E1 = A.energyAtDistance(d, calc=calc, a=a, molecule=element,
        #                        periodic=True)
        energies = [A.energyAtDistance(d, calc=calc, a=a,
                                       dislocation=(dx,dy,dz),
                                       molecule=element, periodic=True)
                    for (dx,dy,dz) in self.points]

        return energies


class ConvergenceTest(Test):

    def __init__(self, unit_badness=.05):
        #Formerly standard value was .2
        #change default unit badness to .005 someday
        Test.__init__(self)
        self.unit_badness = unit_badness

    def badness(self, out, symbol, setup):
        print >> out, '\tConvergence'
        out.flush()
        subtesttime1 = time.clock()
        #if self.quick:
        #    convergencevalue = 42.
        #else:
        convergencevalue = self.convergenceTest(symbol, setup)
        convergencebadness = (convergencevalue/self.unit_badness)**2
        subtesttime2 = time.clock()
        print >> out, '\t\tConvergence: ', convergencevalue
        print >> out, '\t\tTime       : ', (subtesttime2-subtesttime1)
        print >> out, '\t\tBadness    : ', convergencebadness
        return convergencebadness

    """
    Plots the energy of a N2 moleculeas a function of different resolutions
    (h-values) and returns the maximal difference
    """
    def convergenceTest(self, symbol, setup):
        A = atomization
        element = A.elements[symbol]
        h = [.15, .17, .20]
        a = 4.
        #if self.quick:
        #    h = [.20, .22, .25]
        calc = [A.MakeCalculator(element.nbands2, out=None, h=h0,
                                 setup=setup) for h0 in h]
        element = atomization.elements[symbol]
        E = [A.energyAtDistance(element.d, calc=c, a=a,
                                molecule=element) for c in calc]

        return max(E) - min(E)


class IterationTest(Test):

    def __init__(self, tests, names, weights):
        Test.__init__(self)
        self.tests = tests
        self.names = names
        self.weights = weights

    def badness(self, out, symbol, setup, starttime=None, startwalltime=None):
        print >> out, '\tTime/Iterations'
        dt = time.clock() - starttime
        iterationbadness = sum([test.get_last_iteration_count() * weight
                                for (test, weight)
                                in zip(self.tests, self.weights)])/128.
        #In order to make the number more "edible", let's divide by
        #some large number such as 128
        
        #iterationbadness = (iterationcount_Ea + 3*iterationcount_d)/128.
        #if self.quick:
        #    iterationbadness = 42.
        if starttime != None:
            print >> out, '\t\tCpu Time   :', (time.clock() - starttime)
        if startwalltime != None:
            print >> out, '\t\tWall time  :', (time.time() - startwalltime)
        for test,name in zip(self.tests, self.names):
            print >> out, '\t\tIter.',name,':',test.get_last_iteration_count()
        #print >> out, '\t\tIter. (Ea) : ', iterationcount_Ea
        #print >> out, '\t\tIter. (d)  : ', iterationcount_d
        print >> out, '\t\tIter.Badn. : ', iterationbadness
        out.flush()
        return iterationbadness

def main(name='test',symbol='N', argcount=2, tolerance=0.01, quick=False,
         simplex=None):
    global optimizer #make sure optimizer is accessible from
    #interactive interpreter even if something goes wrong

    optimizer = Optimizer(symbol, name, parametercount=argcount,quick=quick,
                          simplex=simplex)
    
    if tolerance != None:
        optimizer.optimize(tolerance)
    return optimizer

def quick(name='quick'):
    return main(name, quick=True, tolerance=-1)

def test_single_setup(test=None, symbol='N', name='paw', out=sys.stdout):
    if test is None:
        test = MoleculeTest()
    return test.badness(out, symbol, name)

def make_and_test_single_setup(setup_parameters, test=None,
                               symbol='N', name='check', out=sys.stdout):
    gen = setupgenerator.SetupGenerator(symbol, name)
    gen.generate_setup(setup_parameters)
    if test is None:
        test = MoleculeTest()
    #fulltest = MoleculeTest()
    out = out #open('checkfile', 'w')
    return test.badness(out, symbol, name)
