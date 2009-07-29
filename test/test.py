"""
A simple test script.

Runs all scripts named ``*.py``.  The tests that execute
the fastest will be run first.
"""

import os
import sys
import time
from glob import glob
import gc
from optparse import OptionParser

import gpaw

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

parser.add_option('-u', '--new-unittest',
                  action='store_true',
                  help='Use new unittest module with parallel support.')

parser.add_option('--from', metavar='TESTFILE', dest='from_test',
                  help='Run remaining tests, starting from TESTFILE')

parser.add_option('--dry', action='store_true',
                  help='Do not run any tests, but write the names of those '
                  'tests which would be run')

opt, tests = parser.parse_args()

if len(tests) == 0:
    # Fastest first, slowest last:
    tests = [
        'ase3k_version.py',
        'pes.py',
        'lapack.py',
        'eigh.py',
        'setups.py',
        'xc.py',
        'xcfunc.py',
        'gradient.py',
        'pbe-pw91.py',
        'cg2.py',
        'd2Excdn2.py',
        'test_dot.py',
        'blas.py',
        'gp2.py',
        'non-periodic.py',
        'lf.py',
        'lxc_xc.py',
        'Gauss.py',
        'cluster.py',
        'derivatives.py',
        'integral4.py',
        'transformations.py',
        'pbc.py',
        'poisson.py',
        'XC2.py',
        'XC2Spin.py',
        'multipoletest.py',
        'proton.py',
        'parallel/ut_parallel.py',
        'parallel/compare.py',
        'coulomb.py',
        'ase3k.py',
        'eed.py',
        'timing.py',
        'gauss_wave.py',
        'gauss_func.py',
        'xcatom.py',
        'parallel/overlap.py',
        'usesymm.py',
        'mixer.py',
        'mixer_broydn.py',
        'ylexpand.py',
        'wfs_io.py',
        'restart.py',
        'gga-atom.py',
        'nonselfconsistentLDA.py',
        'bee1.py',
        'refine.py',
        'revPBE.py',
        'lcao_largecellforce.py',
        'lrtddft2.py',
        'nonselfconsistent.py',
        'stdout.py',
        'ewald.py',
        'spinpol.py',
        'plt.py',
        'parallel/hamiltonian.py',
        'bulk.py',
        'restart2.py',
        'hydrogen.py',
        'aedensity.py',
        'H-force.py',
        'CL_minus.py',
        'gemm.py',
        'gemv.py',
        'fermilevel.py',
        'degeneracy.py',
        'h2o-xas.py',
        'si_primitive.py',
        'si.py',
        'simple_stm.py',
        'asewannier.py',
        'vdw/quick.py',
        'vdw/potential.py',
        'vdw/quick_spin.py',
        'lxc_xcatom.py',
        'davidson.py',
        'cg.py',
        'h2o-xas-recursion.py',
        'atomize.py',
        'Hubbard_U.py',    
        'lrtddft.py',
        'lcao_force.py',
        'wannier-ethylene.py',
        'CH4.py',
        'neb.py',
        #'hgh_h2o.py',
        'apmb.py',
        'relax.py',
        'generatesetups.py',
        'muffintinpot.py',
        'restart_band_structure.py',
        'ldos.py',
        'lcao_bulk.py',
        'revPBE_Li.py',
        'fixmom.py',
        'xctest.py',
        'td_na2.py',
        'exx_coarse.py',
        '2Al.py',
        'si-xas.py',
        'tpss.py',
        'atomize.py',
        'nsc_MGGA.py',
        '8Si.py',
        'transport.py',
        'fixed_boundary.py',
        'Cu.py',
        'lcao_h2o.py',
        'IP-oxygen.py',
        'exx.py',
        'dscf_CO.py',
        'h2o_dks.py',
        'H2Al110.py',
        'ltt.py',
        'vdw/ar2.py',
        'mgga_restart.py',
        'fd2lcao_restart.py',
        'parallel/ut_hsops.py',
        'parallel/ut_invops.py'
        ]

disabled_tests = [
    'vdw/quick_spin.py',
    'external_potential.py',
    'dscf_H2Al.py',
    'lb.py',
    'kli.py',
    'C-force.py',
    'apply.py',
    'viewmol_trajectory.py',
    'fixdensity.py',
    'average_potential.py',
    'lxc_testsetups.py',
    'restart3.py',
    'totype_test.py',
    'wannier-hwire.py',
    'lxc_spinpol_Li.py',
    'lxc_testsetups.py',
    'lxc_generatesetups.py',
    ]

tests_parallel = [
    'parallel/restart.py',
    'parallel/parmigrate.py',
    'parallel/par8.py',
    'parallel/par6.py',
    'parallel/exx.py',
    'parallel/domain_only.py',
    'parallel/lcao_hamiltonian.py',
    ]

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

from ase.parallel import size
if size > 1:
    exclude += ['pes.py',
                'asewannier.py',
                'wannier-ethylene.py',
                'restart_band_structure.py',
                'muffintinpot.py']

for test in exclude:
    if test in tests:
        tests.remove(test)

if opt.from_test:
    fromindex = tests.index(opt.from_test)
    tests = tests[fromindex:]

#gc.set_debug(gc.DEBUG_SAVEALL)

import gpaw.mpi as mpi

if opt.new_unittest:
    from gpaw.testing.parunittest import ParallelTestCase as TestCase, \
        _ParallelTextTestResult as _TextTestResult, ParallelTextTestRunner as \
        TextTestRunner, ParallelTestSuite as TestSuite
else:
    from unittest import TestCase, _TextTestResult, TextTestRunner, TestSuite

class ScriptTestCase(TestCase):
    garbage = []
    def __init__(self, filename):
        TestCase.__init__(self, 'testfile')
        self.filename = filename

    def setUp(self):
        pass

    def testfile(self):
        try:
            execfile(self.filename, {})
        finally:
            mpi.world.barrier()

    def tearDown(self):
        gc.collect()
        n = len(gc.garbage)
        ScriptTestCase.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s) %s' %
                        (n, 's'[:n > 1], ScriptTestCase.garbage))
    def run(self, result=None):
        if result is None: result = self.defaultTestResult()
        try:
            TestCase.run(self, result)
        except KeyboardInterrupt:
            result.stream.write('SKIPPED\n')
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                result.stop()

    def id(self):
        return self.filename

    def __str__(self):
        return '%s' % self.filename

    def __repr__(self):
        return "ScriptTestCase('%s')" % self.filename

class MyTextTestResult(_TextTestResult):
    def startTest(self, test):
        _TextTestResult.startTest(self, test)
        self.stream.flush()
        self.t0 = time.time()

    def _write_time(self):
        if self.showAll:
            self.stream.write('(%.3fs) ' % (time.time() - self.t0))

    def addSuccess(self, test):
        self._write_time()
        _TextTestResult.addSuccess(self, test)
    def addError(self, test, err):
        self._write_time()
        _TextTestResult.addError(self, test, err)
    def addFailure(self, test, err):
        self._write_time()
        _TextTestResult.addFailure(self, test, err)

class MyTextTestRunner(TextTestRunner):
    parallel = opt.new_unittest
    def _makeResult(self):
        args = (self.stream, self.descriptions, self.verbosity,)
        if self.parallel:
            args = (mpi.world,) + args
        return MyTextTestResult(*args)

if opt.dry:
    for test in tests:
        print test
    raise SystemExit

ts = TestSuite()
for test in tests:
    ts.addTest(ScriptTestCase(filename=test))

from gpaw.utilities import devnull

sys.stdout = devnull

ttr = MyTextTestRunner(verbosity=opt.verbosity, stream=sys.__stdout__)
result = ttr.run(ts)
failed = [test.filename for test, msg in result.failures + result.errors]

sys.stdout = sys.__stdout__

if mpi.rank == 0 and len(failed) > 0:
    open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
