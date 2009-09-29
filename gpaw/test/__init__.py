import gc
import sys
import time

from gpaw.utilities import devnull
import gpaw.mpi as mpi
import gpaw


tests = [
    'ase3k_version.py',
    'lapack.py',
    'eigh.py',
    'setups.py',
    'xc.py',
    'xcfunc.py',
    'gradient.py',
    'pbe_pw91.py',
    'cg2.py',
    'd2Excdn2.py',
    'dot.py',
    'blas.py',
    'gp2.py',
    'non_periodic.py',
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
    'kptpar.py',
    'parallel/overlap.py',
    'symmetry.py',
    'pes.py',
    'usesymm.py',
    'mixer.py',
    'mixer_broydn.py',
    'ylexpand.py',
    'wfs_io.py',
    'restart.py',
    'gga_atom.py',
    'nonselfconsistentLDA.py',
    'bee1.py',
    'refine.py',
    'revPBE.py',
    'jstm.py',
    'lcao_largecellforce.py',
    'lcao_h2o.py',
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
    'H_force.py',
    'CL_minus.py',
    'gemm.py',
    'gemv.py',
    'fermilevel.py',
    'degeneracy.py',
    'h2o_xas.py',
    'si.py',
    'simple_stm.py',
    'asewannier.py',
    'vdw/quick.py',
    'vdw/potential.py',
    'vdw/quick_spin.py',
    'lxc_xcatom.py',
    'davidson.py',
    'cg.py',
    'h2o_xas_recursion.py',
    'atomize.py',
    'Hubbard_U.py',    
    'lrtddft.py',
    'lcao_force.py',
    'parallel/lcao_hamiltonian.py',
    'wannier_ethylene.py',
    'CH4.py',
    'neb.py',
    'hgh_h2o.py',
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
    'lcao_bsse.py',
    '2Al.py',
    'si_primitive.py',
    'si_xas.py',
    'tpss.py',
    'atomize.py',
    'nsc_MGGA.py',
    '8Si.py',
    'coreeig.py',
    'transport.py',
    'Cu.py',
    'IP_oxygen.py',
    'exx.py',
    'dscf_CO.py',
    'h2o_dks.py',
    'H2Al110.py',
    'nscfsic.py',
    'ltt.py',
    'vdw/ar2.py',
    'mgga_restart.py',
    'fd2lcao_restart.py',
    'parallel/ut_hsops.py',
    'parallel/ut_invops.py',
    'parallel/scalapack.py',
    'parallel/lcao_projections.py',
    ]


exclude = []
if mpi.size > 1:
    exclude += ['pes.py',
                'nscfsic.py',
                'coreeig.py',
                'asewannier.py',
                'wannier_ethylene.py',
                'muffintinpot.py']
if mpi.size > 2:
    exclude += ['neb.py', 'transport.py']

if mpi.size != 4:
    exclude += ['parallel/scalapack.py']

for test in exclude:
    if test in tests:
        tests.remove(test)

if mpi.size > 1:
    from gpaw.test.parunittest import ParallelTestCase as TestCase
    from gpaw.test.parunittest import _ParallelTextTestResult as \
         _TextTestResult
    from gpaw.test.parunittest import ParallelTextTestRunner as \
         TextTestRunner
    from gpaw.test.parunittest import ParallelTestSuite as TestSuite
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
    parallel = (mpi.size > 1)
    def _makeResult(self):
        args = (self.stream, self.descriptions, self.verbosity)
        if self.parallel:
            args = (mpi.world,) + args
        return MyTextTestResult(*args)


def run_all(tests, stream=sys.__stdout__, jobs=1):
    ts = TestSuite()
    path = gpaw.__path__[0] + '/test/'
    for test in tests:
        ts.addTest(ScriptTestCase(filename=path + test))

    sys.stdout = devnull
    ttr = MyTextTestRunner(verbosity=2, stream=stream)
    result = ttr.run(ts)
    failed = [test.filename for test, msg in result.failures + result.errors]
    sys.stdout = sys.__stdout__
    return failed

