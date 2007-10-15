#!/usr/bin/env python

"""
This file contains an Optimizer class, the purpose of which is to
optimize GPAW setups. The GPAW setup optimizer uses the downhill
simplex algorithm to search a parameter space of any dimension.
"""


import sys, traceback, pickle, random
import atomization, setupgenerator, amoeba
from badness import MoleculeTest
from gpaw.utilities import devnull

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
    print center
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

    def __init__(self, symbol='N', name='test', generator=None, test=None,
                 simplex=None, out=None, dumplog=None):
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

        generatorname = 'opt.'+name

        if generator is None:
            generator = setupgenerator.SetupGenerator(self.element.symbol,
                                                      generatorname)
        elif isinstance(generator, setupgenerator.SetupGenerator):
            # Override generator defaults
            generator.set_name(generatorname)
            generator.set_symbol(symbol)
        else:
            try: # Test whether generator is specified as a parameter list
                parms = list(generator)
                generator = setupgenerator.SetupGenerator(self.element.symbol,
                                                          generatorname,
                                                          whichparms=parms)
            except Exception:
                raise Exception('Bad generator: '+str(generator))

        self.generator = generator

        if simplex is None:
            params = self.generator.get_standard_parameters()
            simplex = get_simplex(params, 0.03)

        self.simplex = simplex

        self.setup_name = 'opt.'+name

        outname = name+'.opt.'+symbol
        if out is None:
            self.output = open(outname+'.log', 'w')
            #Note: perhaps also make it possible to not write output
        elif out == '-':
            self.output = sys.stdout
        else:
            self.output = open(out, 'w')

        out = self.output

        dumpname = outname + '.dump.pckl'
        if dumplog is None:
            dumplog = pickle.Pickler(open(dumpname, 'w'))
        self.dumplog = pickle.Pickler(dumplog)

        dimension = len(simplex)-1
        self.stepcount = 0

        print >> out, 'Optimization run commencing'
        print >> out, separator
        print >> out, 'Name:',name
        print >> out, 'Element:',symbol
        print >> out, 'Dump file:',dumpname
        print >> out, 'Parameter space dimension:',dimension
        #print >> out, 'Tolerance:',fTolerance
        print >> out, separator
        print >> out
        print >> out, 'Simplex points'
        for point in simplex:
            print >> out, '  ',point
        out.flush()
        print >> out
        print >> out, 'Evaluating simplex point badness values'
        out.flush()

        dumpdict = {'type'      : 'header',
                    'name'      : name,
                    'symbol'    : symbol,
                    'dimension' : dimension,
                    'simplex'   : simplex,
                    'testname'  : str(test.__class__),
                    'test'      : dict(test.dumplog),
                    'generator' : generator.descriptor}

        dumplog.dump(dumpdict)

        values = [self.badness(point) for point in simplex]
        
        print >> out, 'Simplex point evaluation complete.'
        print >> out
        out.flush()
        
        #If resuming from file, remember to make it possible to reuse badness
        #values
        self.amoeba = amoeba.Amoeba(simplex, values, self.badness)

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

            data = {'type'      : 'status',
                    'simplex'   : self.simplex,
                    'deviation' : self.amoeba.relativedeviation,
                    'step'      : self.stepcount}

            self.dumplog.dump(data)

    def badness(self, args):
        """
        Runs a full test of a given GPAW setup
        """
        
        out = self.output

        try:
            self.generator.generate_setup(args) #new setup
        
            print >> out, 'Point: ',args
            badness = self.test.badness(self.element.symbol,
                                        self.setup_name, out)
        except KeyboardInterrupt:
            raise sys.exc_info()[0]
        except Exception:
            badness = 10000
            header = '=== Exception ==='
            print >> out, header
            traceback.print_exc(file=out)
            print >> out, '='*len(header)

        print >> out, 'Badness: ', badness
        print >> out
        out.flush()

        data = {'type'      : 'eval',
                'point'     : args,
                'badness'   : badness,
                'log'       : dict(self.test.dumplog),
                'stepcount' : self.stepcount}
        # NOTE: It is IMPORTANT to create a NEW dict identical to
        # self.test.dumplog, since otherwise pickle will apparently become
        # CONFUSED and dump the SAME values over and over!

        self.dumplog.dump(data)
        
        return badness

optimizer = None

def main(symbol='H', name='test', generator=[3,4], tolerance=0.01,
         simplex=None, test=None):
    global optimizer # Make sure optimizer is accessible from
    # interactive interpreter even if something goes wrong

    if test is None:
        test = MoleculeTest()

    if generator is None:
        generator = setupgenerator.SetupGenerator(symbol, name)
    elif isinstance(generator, setupgenerator.SetupGenerator):
        generator.set_name(name)
        generator.set_symbol(symbol)
    else:
        try:
            whichparms = list(generator)
            generator = setupgenerator.SetupGenerator(symbol, name,
                                                      whichparms=whichparms)
        except Exception:
            raise Exception('Bad generator spec')

    optimizer = Optimizer(symbol, name, generator=generator, test=test,
                          simplex=simplex)

    if tolerance != None:
        optimizer.optimize(tolerance)
    return optimizer

def test_single_setup(test=None, symbol='N', name='paw', out=sys.stdout):
    if test is None:
        test = MoleculeTest()
    return test.badness(symbol, name, out)

def make_and_test_single_setup(generator, setup_parameters, test=None,
                               symbol='N', name='check', out=sys.stdout):
    if not isinstance(generator, setupgenerator.SetupGenerator):
        gen = setupgenerator.SetupGenerator(symbol, name,
                                            whichparms=generator)
    else:
        gen.set_name(name)
        gen.set_symbol(symbol)

    generator.generate_setup(setup_parameters)
    if test is None:
        test = MoleculeTest()
    return test.badness(symbol, name, out)

def parseargs():
    # Args should contain:
    #
    # * symbol
    # * name
    # * parameter list (optional; if specified, this will be
    # appended to the name)
    if 'opt' in sys.argv:
        index = sys.argv.index('opt') + 1
        args = sys.argv[index].split(',')

        name = args[1]
        if len(args) > 2:
            parameters = map(int, args[2])
            name = args[1] + str(parameters)
        else:
            parameters = range(5)
    else:
        return []
    return tuple([args[0], name, parameters, 1e-5, None,
                  None])

if __name__ == '__main__':
    main(*parseargs())
