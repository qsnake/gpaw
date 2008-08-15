"""
A simple test script.

Runs all scripts named ``*.py``.  The tests that execute
the fastest will be run first.
"""

import os
import sys
import time
import unittest
from glob import glob
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

opt, tests = parser.parse_args()

if len(tests) == 0:
    # Fastest first, slowest last:
    tests = ['pbe-pw91.py', 'xcfunc.py', 'xc.py', 'gp2.py', 'lapack.py',
             'gradient.py', 'lf.py', 'non-periodic.py', 'lxc_xc.py',
             'transformations.py', 'Gauss.py', 'denom_int.py', 'setups.py',
             'poisson.py', 'cluster.py', 'integral4.py', 'cg2.py', 'XC2.py',
             'd2Excdn2.py', 'XC2Spin.py', 'multipoletest.py', 'eed.py',
             'coulomb.py',
             'ase3k.py', 'mixer.py', 'proton.py', 'timing.py', 'restart.py',
             'gauss_func.py', 'xcatom.py', 'wfs_io.py', 'ylexpand.py',
             'nonselfconsistentLDA.py', 'bee1.py', 'gga-atom.py', 'revPBE.py',
             'td_na2.py', 'nonselfconsistent.py', 'external_potential.py',
             'bulk.py', 'spinpol.py', 'refine.py',
             'bulk-lcao.py', 'stdout.py', 'restart2.py', 'hydrogen.py',
             'H-force.py', 'plt.py', 'h2o-xas.py', 'degeneracy.py',
             'davidson.py', 'cg.py', 'ldos.py', 'h2o-xas-recursion.py',
             'atomize.py', 'wannier-ethylene.py', 'lrtddft.py', 'CH4.py',
             'gllb2.py', 'apmb.py', 'relax.py', 'fixmom.py',
             'si-xas.py', 
             'revPBE_Li.py', 'lxc_xcatom.py', 'exx_coarse.py', '2Al.py',
             '8Si.py', 'dscf_test.py', 'lcao-h2o.py', 'IP-oxygen.py',
             'generatesetups.py', 'aedensity.py', 'h2o_dks.py', 'Cu.py',
             'exx.py',
             'H2Al110.py', 'ltt.py', 'ae-calculation.py']

disabled_tests = ['lb.py', 'kli.py', 'C-force.py', 'apply.py',
                  'viewmol_trajectory.py', 'vdw.py', 'fixdensity.py',
                  'average_potential.py', 'lxc_testsetups.py',
                  'restart3.py', 'totype_test.py',
                  'wannier-hwire.py',
                  'lxc_spinpol_Li.py', 'lxc_testsetups.py',
                  'lxc_generatesetups.py', 'simple_stm.py']

tests_parallel = ['parallel/restart.py', 'parallel/parmigrate.py',
                  'parallel/par8.py', 'parallel/par6.py',
                  'parallel/exx.py']

if opt.run_failed_tests_only:
    tests = [line.strip() for line in open('failed-tests.txt')]

if opt.debug:
    sys.argv.append('--debug')

exclude = []
if opt.exclude is not None:
    exclude += opt.exclude.split(',')

# exclude parallel tests if opt.parallel is not set
if not opt.parallel:
    exclude.extend(tests_parallel)

for test in exclude:
    if test in tests:
        tests.remove(test)

#gc.set_debug(gc.DEBUG_SAVEALL)

class ScriptTestCase(unittest.TestCase):
    garbage = []
    def __init__(self, filename):
        unittest.TestCase.__init__(self, 'testfile')
        self.filename = filename

    def setUp(self):
        pass

    def testfile(self):
        try:
            execfile(self.filename, {})
        except KeyboardInterrupt:
            raise RuntimeError('Keyboard interrupt')
        
    def tearDown(self):
        gc.collect()
        n = len(gc.garbage)
        ScriptTestCase.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s)' %
                        (n, 's'[:n > 1]))

    def id(self):
        return self.filename

    def __str__(self):
        return '%s' % self.filename

    def __repr__(self):
        return "ScriptTestCase('%s')" % self.filename

class MyTextTestResult(unittest._TextTestResult):
    def startTest(self, test):
        unittest._TextTestResult.startTest(self, test)
        self.t0 = time.time()
        
    def addSuccess(self, test):
        self.stream.write('(%.3fs) ' % (time.time() - self.t0))    
        unittest._TextTestResult.addSuccess(self, test)

class MyTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return MyTextTestResult(self.stream, self.descriptions, self.verbosity)

ts = unittest.TestSuite()
for test in tests:
    ts.addTest(ScriptTestCase(filename=test))

from gpaw.utilities import devnull
sys.stdout = devnull

ttr = MyTextTestRunner(verbosity=opt.verbosity)
result = ttr.run(ts)
failed = [test.filename for test, msg in result.failures + result.errors]

sys.stdout = sys.__stdout__

if len(failed) > 0:
    open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
