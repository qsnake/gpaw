#!/usr/bin/env python

import os
import sys
import traceback
import pickle
from optparse import OptionParser

from gpaw.utilities import locked
from gpaw.testing import data
from gpaw.paw import ConvergenceError
from gpaw.testing import calc

tests = {'a' : calc.atomic_energy,
         'c' : calc.grid_convergence_energies,
         'm' : calc.molecular_energies,
         'e' : calc.eggbox_energies}

elements_requiring_lattice_test = ['C']

def loadfromfile(filename):
    unpickler = pickle.Unpickler(open(filename))
    try:
        # return the tuple (parameters, results)
        return unpickler.load()
    except (EOFError, IOError, ValueError):
        return None

def test(function, formula, setups='paw', quick=False, clean=False,
         retrieve=False):
    """Performs one test on the specified molecule or atom using the
    specified setups.  If results are already cached, the cached
    result will be returned without calculation.  Otherwise the result
    will be cached, such that subsequent similar calls will return
    immediately.  Only PBE setups can be used.

    * function - name of the test function. Either 'a', 'c', 'm' or 'e'.
    * formula - a formula or atomic symbol such as 'H2' or 'C2H4', or 'Cl'.
    * setups - the name of the setups to be used (see gpaw.paw documentation).
    * quick - whether the test should use very small unit cells, gpt count etc.
    * clean - delete empty/bad cache files; no calculation performed.
    * retrieve - return cached data; no calculation performed.

    The return value is either None or whatever the specified function returns.
    """
    name = '%s.%s.%s' % (setups, formula, function)
    if quick:
        name = name+'.quick'
    dumpfilename = name+'.pckl'
    logname = name+'.log'
    if locked(dumpfilename):
        results = loadfromfile(dumpfilename)
        if clean and results is None:
            try:
                os.remove(dumpfilename)
                os.remove(logname)
            except OSError: # maybe the log file doesn't exist; ignore
                pass # This will also skip if the file is read-only, etc.
            print 'Cache purged'
            return None
        if results is None:
            print 'Cache empty'
        else:
            print 'Loaded'
        return results
    
    if retrieve or clean:
        print 'No cache found'
        return None

    dumpfile = open(dumpfilename, 'w')
    log = open(logname, 'w')
    print 'Calculating ...',
    sys.stdout.flush()
    try:
        results = tests[function](formula, setups=setups, quick=quick,
                                  txt=log)
        print 'Done'
    except: # It is quite difficult to know which exceptions are relevant
        traceback.print_exc(file=log)
        print 'Error'
        raise sys.exc_info()[0]
    pickle.dump(results, dumpfile)
    log.close()
    return results

def testmultiple(testfunctions='acme', formulae=None, setups='paw',
                 quick=False, clean=False, retrieve=False):
    if testfunctions.strip('acme') != '':
        raise ValueError('Unknown tests designated: %s' % testfunctions)
    if formulae is None:
        formulae = data.moleculedata.keys()

    symbollist = [] # Obtain the atomic symbols of all constituents
    if 'a' in testfunctions:
        for formula in formulae:
            symbollist.extend([atom.GetChemicalSymbol() for atom
                              in data.molecules[formula]])
    # The 'set' builtin is not included in some python versions. Using hack
    symbols = dict(zip(symbollist, symbollist)).keys()
    #symbols = set(symbollist)

    # Make sure that the single-atom test is not performed on molecules
    moleculetests = testfunctions.replace('a','')
    task_args = []
    for formula in formulae:
        for function in moleculetests:
            task_args.append((function, formula, setups, quick, clean,
                              retrieve))

    if 'a' in testfunctions:
        for symbol in symbols:
            task_args.append(('a', symbol, setups, quick, clean, retrieve))

    results = {}
    results['setups'] = setups
    for testname in 'acme':
        results[testname] = {}
    
    for args in task_args:
        try:
            testname, formula = args[0], args[1]
            print 'Running test \'%s\' for %s ...' % (testname,formula),
            sys.stdout.flush()
            result = test(*args)
            results[testname][formula] = result
        except KeyboardInterrupt:
            raise sys.exc_info()[0] # Don't ignore keyboard interrupts
        except: #(RuntimeError, ConvergenceError, AssertionError):
            pass
    return results

def main():
    parser = OptionParser(usage='%prog [options [file]] [formulae ... ]')
    parser.add_option('-t', '--tests', action='store', default='acme',
                       help='Run these tests. [default: \'acme\']')
    parser.add_option('-s', '--setups', action='store', default='paw',
                      help='Use specified setups. [default: \'paw\']')
    parser.add_option('-q', '--quick', action='store_true', default=False,
                      help='Run very quick tests.')
    parser.add_option('-c', '--clean', action='store_true', default=False,
                      help='Remove bad or empty cache files and exit.')
    parser.add_option('-a', '--assemble', action='store',
                      dest='filename',
                      help='Collect all cached results in the specified file.')

    opt, formulae = parser.parse_args()

    if not formulae:
        formulae = data.molecules.keys()

    print '+---------------------------+'
    print '| Molecule tests commencing |'
    print '+---------------------------+'
    print 'Setups:', opt.setups
    print 'Tests:', opt.tests
    print 'Formulae:', formulae
    print
    if opt.quick:
        print 'This is only a trial run!'
    if opt.clean:
        print 'This will remove bad cache files and exit'
    if opt.filename is not None:
        print 'This will dump existing cached results to a single file.'
        print 'No calculations will be performed.'

    results = testmultiple(opt.tests, formulae, setups=opt.setups,
                           quick=opt.quick, clean=opt.clean,
                           retrieve=(opt.filename is not None))
    if opt.filename is not None:
        pickle.dump(results, open(opt.filename, 'w'))

if __name__ == '__main__':
    main()
