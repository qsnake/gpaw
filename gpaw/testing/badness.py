import time
import sys
import pickle
import atomization
import Numeric as num
from gpaw.utilities import devnull
from LinearAlgebra import inverse

"""
Contains various Test classes to be used by the gpaw setup optimizer
"""

nullpickler = pickle.Pickler(devnull)

class Test:
    """
    Base class of setup badness tests. An actual test should override the
    badness method. Presently a test needs not be a subclass of Test, it needs
    only implement a badness method with appropriate signature.
    """

    def __init__(self):
        # Any relevant data may be dumped into this dictionary when the 
        # badness function is called.  Just overwrite entries each time.
        self.dumplog = {}

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

class MoleculeTest(Test):
    """
    Test class for elements that form molecules.
    """
    def __init__(self):
        Test.__init__(self)
        e = EnergyTest()
        d = DistanceTest()
        self.tests = [e, d, NoiseTest(),
                      ConvergenceTest()]
        self.names = ['energy', 'distance', 'noise', 'convergence']
        self.iterationtest = IterationTest([e, d], ['Ea','d '], [5, 1])
        self.dump_to_log(None)        

    def badness(self, symbol, setup, out=devnull):
        """
        Returns a weighted square sum of badnesses from EnergyTest,
        DistanceTest, NoiseTest and FluctuationTest, multiplied by a
        number derived from the iteration count during these badness
        evaluations.
        """
        starttime = time.clock()
        startwalltime = time.time()
        badnessvalues = [test.badness(symbol, setup, out)
                         for test in self.tests]
        (b_energy, b_dist, b_noise, b_conv) = tuple(badnessvalues)
        b_iter = self.iterationtest.badness(symbol, setup, out,
                                            starttime, startwalltime)
        badness = (b_energy + b_dist + b_noise + b_conv) * b_iter
        self.dump_to_log(badness)

        return badness

    def dump_to_log(self, badness):
        log = self.dumplog

        log['badness'] = badness

        for name, test in zip(self.names, self.tests):
            log[name] = dict(test.dumplog)
        log['iterations'] = dict(self.iterationtest.dumplog)        


class EnergyTest(Test):
    """
    This test compares the energy of a diatomic molecule of the
    relevant element to a reference PBE value.
    """
    def __init__(self, cellsize=7.5, unit_badness=.05):
        Test.__init__(self)
        self.cellsize = cellsize
        
        #self.element = atomization.elements[symbol]
        #self.ref_energy = self.element.Ea_PBE
        self.unit_badness = unit_badness
        self.iterationcounts = None
        self.dumplog['unit'] =  unit_badness
        self.dumplog['cell'] = cellsize

    def badness(self, symbol, setup, out=devnull, calctxt=None):
        """
        Returns the badness of the specified setup, which is
        proportional to the square of the deviation of the atomization
        energy from the reference value
        """
        #energy_badness_ = 1/.05**2 #badness == 1 for deviation == .05 eV
        
        #molecule = atomization.elements[symbol]
        (c1,c2) = atomization.atomization_calculators(out=calctxt,setup=setup,
                                                      symbol=symbol)
        print >> out, 'Energy test'
        out.flush()
        subtesttime1 = time.clock()
        energy = atomization.calc_energy(c1,c2,a=self.cellsize,
                                         symbol=symbol)
        subtesttime2 = time.clock()
        self.iterationcounts = (c1.niter, c2.niter)
        ref_energy = atomization.elements[symbol].energy_pbe
        energybadness = ((energy - ref_energy)/self.unit_badness)**2
        testtime = subtesttime2-subtesttime1

        print >> out, '  Ea         : ', energy
        print >> out, '  Time       : ', testtime
        print >> out, '  Badness    : ', energybadness
        print >> out, '  Iterations : ', self.iterationcounts
        out.flush()

        log = self.dumplog
        log['energy'] = energy
        log['time'] = testtime
        log['badness'] = energybadness
        log['niter'] = self.iterationcounts

        return energybadness

    def get_last_iteration_count(self):
        return sum(self.iterationcounts)
        
class DistanceTest(Test):
    """
    This test compares the bond length to some reference value.
    """
    def __init__(self, cellsize=7.5, unit_badness=.005):
        Test.__init__(self)
        self.unit_badness = unit_badness
        self.cellsize = cellsize
        self.dumplog['unit']=unit_badness
        self.dumplog['cell']=cellsize
    
    def badness(self, symbol, setup, out=devnull):
        """
        The badness is proportional to the square of the deviation of
        the bond length from the reference value.

        The bond length is calculated by finding three energies close
        to the bond length, interpolating with a 2nd degree
        polynomial, then finding the minimum value of that polynomial.
        """
        print >> out, 'Distance test'
        ref_dist = atomization.elements[symbol].d
        out.flush()
        subtesttime1 = time.clock()
        d = self.bondlength(symbol, setup)

        distancebadness = ((d - ref_dist)/self.unit_badness)**2
        subtesttime2 = time.clock()
        elapsed = subtesttime2-subtesttime1
        print >> out, '  Distance   : ', d
        print >> out, '  Time       : ', elapsed
        print >> out, '  Badness    : ', distancebadness
        print >> out, '  Iterations : ', self.iterationcounts
        out.flush()

        log = self.dumplog
        log['distance'] = d
        log['time'] = elapsed
        log['badness'] = distancebadness
        log['niter'] = self.iterationcounts

        return distancebadness

    def get_last_iteration_count(self):
        """
        Returns the iteration count from the last calculation.
        """
        return sum(self.iterationcounts)

    def bondlength(self, symbol, setup, out=devnull):
        """
        Returns the bond length. Calculates energy at three locations
        around the reference bond length, interpolates with a 2nd degree
        polynomial and returns the minimum of this polynomial which would
        be roughly equal to the bond length without engaging in a large
        whole relaxation test
        """

        element = atomization.elements[symbol]
        d0 = element.d
        #REMEMBER: find out what unit badness should be for other elements!
        dd = ( .2 / 140. )**.5 #around .04 A. Bond properties correspond to
        #an energy of E = .5 k x**2 with k = 140 eV/A**2
        #If we want .1 eV deviation then the above dd should be used
        calc = atomization.makecalculator(nbands=element.nbands2,
                                          out=None, setup=setup)
        #Calculate energies at the three points
        dists = [d0-dd, d0, d0+dd]
        niter = []
        energies = []
        for d in dists:
            e = atomization.energy_at_distance(d, calc=calc, a=self.cellsize,
                                               symbol=symbol)
            energies.append(e)
            niter.append(calc.niter)

        #Now find parabola and determine minimum
        x = num.array(dists)
        y = num.array(energies)

        coeffmatrix = num.transpose(num.array([x**0, x**1, x**2]))
        c = num.dot(inverse(coeffmatrix), y)
        bond_length = - c[1] / (2.*c[2]) # "-b/(2a)"

        self.iterationcounts = tuple(niter)
        return bond_length#, calc.niter

class NoiseTest(Test):
    """
    This test calculates the energy of several systems differing only
    by a translation smaller than the grid resolution.  The badness is
    proportional to the square of the maximum deviation of these
    energies, which should obviously be identical ideally.
    """
    def __init__(self, unit_badness=.05, points=None):
        Test.__init__(self)
        self.h = .2
        self.a = 4.5
        if points is None:
            d = num.arrayrange(7.)/6.*self.h/2. # [0 .. h/2], where h ~ 0.2
            points = [(x,x,x) for x in d]
        self.unit_badness=unit_badness
        self.points = tuple(points)
        self.dumplog['unit']=unit_badness
        self.dumplog['points'] = points

    def badness(self, symbol, setup, out=devnull):
        print >> out, 'Fluctuation test'
        out.flush()
        subtesttime1 = time.clock()

        energies = self.energy_fluctuation_test(symbol, setup)
        noise = max(energies) - min(energies)
        
        noisebadness = (noise/self.unit_badness)**2
        subtesttime2 = time.clock()
        elapsed = subtesttime2-subtesttime1
        print >> out, '  Fluctuation:', noise
        print >> out, '  All points :', energies
        print >> out, '  Time       :', elapsed
        print >> out, '  Badness    :', noisebadness

        log = self.dumplog
        log['noise'] = noise
        log['energies'] = energies
        log['time'] = elapsed
        log['badness'] = noisebadness

        return noisebadness

    def energy_fluctuation_test(self, symbol, setup, out=devnull):
        """
        Returns the difference between the energy of a nitrogen molecule
        at the center of the unit cell and the energy of one translated by
        h/2 along the z axis.
        """
        a, h = (self.a, self.h)
        element = atomization.elements[symbol]
        calc = atomization.makecalculator(element.nbands2, out=None,
                                          setup=setup,h=h)
        d = element.d

        energies = [atomization.energy_at_distance(d, calc=calc, a=a,
                                                   dislocation=(dx,dy,dz),
                                                   symbol=symbol,
                                                   periodic=True)
                    for (dx,dy,dz) in self.points]

        return energies


class ConvergenceTest(Test):
    """
    Evaluates the energy of a simple system for a couple of different
    grid resolutions.  It is desirable that the energies be equal,
    since this would infer that it is not necessary to use greater
    resolution.  Badness is proportional to the square of the maximal
    deviation of calculated energies.
    """
    def __init__(self, unit_badness=.35):
        #Formerly standard value was .2
        #change default unit badness to .005 someday
        Test.__init__(self)
        self.a = 4.5
        self.unit_badness = unit_badness
        self.dumplog['unit'] = unit_badness

    def badness(self, symbol, setup, out=devnull):
        print >> out, 'Convergence test'
        out.flush()
        subtesttime1 = time.clock()
        convergencevalue = self.convergence_test(symbol, setup)
        convergencebadness = (convergencevalue/self.unit_badness)**2
        subtesttime2 = time.clock()
        elapsed = subtesttime2-subtesttime1
        print >> out, '  Convergence: ', convergencevalue
        print >> out, '  Time       : ', elapsed
        print >> out, '  Badness    : ', convergencebadness

        log = self.dumplog
        log['conv'] = convergencevalue
        log['time'] = elapsed
        log['badness'] = convergencebadness
        
        return convergencebadness

    def convergence_test(self, symbol, setup):
        """ Finds the energy of a nitrogen molecule as a function of
        different resolutions (h-values) and returns the maximal
        difference """

        element = atomization.elements[symbol]
        h = [.15, .17, .20]
        a = self.a
        calc = [atomization.makecalculator(element.nbands2, out=None, h=h0,
                                           setup=setup) for h0 in h]
        element = atomization.elements[symbol]
        energies = [atomization.energy_at_distance(element.d, calc=c, a=a,
                                                   symbol=symbol,
                                                   periodic=True)
                    for c in calc]

        return max(energies) - min(energies)

class IterationTest(Test):
    """
    This test can be instantiated by specifying a couple of other
    tests, each of which must be able to return an iteration count.
    This test will return a badness value proportional to the total
    iteration count of the specified tests, where the iteration counts
    can be weighted.
    """
    def __init__(self, tests, names, weights):
        Test.__init__(self)
        self.tests = tests
        self.names = names
        self.weights = weights

    def badness(self, symbol, setup, out=devnull, starttime=None,
                startwalltime=None):
        print >> out, 'Time/Iterations'
        dt = time.clock() - starttime

        weightedsum = sum([test.get_last_iteration_count() * weight
                           for (test, weight)
                           in zip(self.tests, self.weights)])
        sumweights = float(sum(self.weights))
        
        iterationbadness = weightedsum/sumweights/128.

        cputime = time.clock() - starttime
        walltime = time.time() - startwalltime
        
        if starttime != None:
            print >> out, '  Cpu Time   :', cputime
        if startwalltime != None:
            print >> out, '  Wall time  :', walltime
        for test,name in zip(self.tests, self.names):
            print >> out, '  Iter.',name,':',test.get_last_iteration_count()
        #print >> out, 'Iter. (Ea) : ', iterationcount_Ea
        #print >> out, 'Iter. (d)  : ', iterationcount_d
        print >> out, '  Iter.Badn. : ', iterationbadness
        out.flush()

        log = self.dumplog
        log['badness'] = iterationbadness
        log['walltime'] = walltime
        log['cputime'] = cputime

        return iterationbadness

def test():
    setups = 'paw'
    if len(sys.argv) > 1:
        setups = sys.argv[-1]
    elements = ['H', 'Li', 'Be', 'N', 'O', 'F', 'P', 'Cl']
    test = MoleculeTest()
    results = []
    outputfile = open('test.%s.txt' % setups, 'w')
    for element in elements:
        try:
            result = test.badness(element, setups, sys.stdout)
        except Exception:
            result = 'FAIL'
        results.append(result)
        print >> outputfile, element, result
        outputfile.flush()

if __name__ =='__main__':
    test()
