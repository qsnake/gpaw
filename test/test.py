"""
A simple test script.

Runs all scripts named ``*.py``.  The tests that execute
the fastest will be run first.
"""

import os
import sys
import unittest
from glob import glob
import time
import gc
from optparse import OptionParser


parser = OptionParser(usage='%prog [options] [tests]',
                      version='%prog 0.1')

parser.add_option('-v', '--verbosity',
                  type='int', default=2,
                  help='Verbocity level.')

parser.add_option('-x', '--exclude',
                  type='string', default=None,
                  help='Exclude tests (comma separated list of tests).',
                  metavar='test1.py,test2.py,...')

parser.add_option('-f', '--run-failed-tests-only',
                  action='store_true',
                  help='Run failed tests only.')

parser.add_option('-d', '--debug',
                  action='store_true', default=False,
                  help='Run tests in debug mode.')

parser.add_option('-p', '--parallel',
                  action='store_true',
                  help='Add parallel tests.')

parser.add_option('--adjust-timings',
                  action='store_true',
                  help='Adjust timing information in scripts.')

opt, tests = parser.parse_args()

if len(tests) == 0:
    tests = glob('*.py')

if opt.run_failed_tests_only:
    tests = [line.strip() for line in open('failed-tests.txt')]

if opt.debug:
    sys.argv.append('--gpaw-debug')

exclude = ['__init__.py', 'test.py', 'C-force.py', 'grr.py']
if opt.exclude is not None:
    exclude += opt.exclude.split(',')

# exclude parallel tests if opt.parallel is not set
if not opt.parallel: 
    exclude.extend(['parallel-restart.py', 'parmigrate.py',
                    'par8.py', 'par6.py', 'exx_parallel.py']) 

for test in exclude:
    if test in tests:
        tests.remove(test)
    
ttests = []
for test in tests:
    line = open(test).readline()
    if line.startswith('# This test takes approximately'):
        t = float(line.split()[-2])
    else:
        t = 10.0
    ttests.append((t, test))

ttests.sort()
tests = [test for t, test in ttests]

#gc.set_debug(gc.DEBUG_SAVEALL)

from ASE.Units import units

class ScriptTestCase(unittest.TestCase):
    garbage = []
    def __init__(self, filename, adjust_timing):
        unittest.TestCase.__init__(self, 'testfile')
        self.filename = filename
        self.adjust_timing = adjust_timing

    def setUp(self):
        self.t = time.time()
        units.length_used = False 
        units.energy_used = False
        units.SetUnits('Ang', 'eV')

    def testfile(self):
        execfile(self.filename, {})

    def tearDown(self):
        t = time.time() - self.t
        gc.collect()
        n = len(gc.garbage)
        ScriptTestCase.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s)' %
                        (n, 's'[:n > 1]))
        if self.adjust_timing:
            lines = open(self.filename).readlines()
            if lines[0].startswith('# This test takes approximately'):
                del lines[0]
            lines[:0] = ['# This test takes approximately %.1f seconds\n' % t]
            os.rename(self.filename, self.filename + '.old')
            open(self.filename, 'w').write(''.join(lines))
        
    def id(self):
        return self.filename

    def __str__(self):
        return '%s' % self.filename

    def __repr__(self):
        return "ScriptTestCase('%s', %r)" % (self.filename, self.adjust_timing)


ts = unittest.TestSuite()
for test in tests:
    ts.addTest(ScriptTestCase(filename=test,
                              adjust_timing=opt.adjust_timings))

from gpaw.utilities import DownTheDrain
sys.stdout = DownTheDrain()
    
ttr = unittest.TextTestRunner(verbosity=opt.verbosity)
result = ttr.run(ts)
failed = [test.filename for test, msg in result.failures + result.errors]

sys.stdout = sys.__stdout__

if len(failed) > 0:
    open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
