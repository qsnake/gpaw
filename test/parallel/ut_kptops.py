#!/usr/bin/env python

#partest = True

import gc
import sys
import time
import numpy as np

from ase.utils.memory import MemorySingleton, MemoryStatistics
from gpaw.mpi import world
from gpaw.utilities import gcd
#from gpaw.utilities.tools import tri2full, md5_array, gram_schmidt
from gpaw.kpt_descriptor import KPointDescriptor
#from gpaw.hs_operators import Operator
#from gpaw.parameters import InputParameters
#from gpaw.xc_functional import XCFunctional
#from gpaw.setup import Setup, Setups
#from gpaw.lfc import LFC

from gpaw.testing.ut_common import ase_svnrevision, shapeopt, TestCase, \
    TextTestRunner, CustomTextTestRunner, defaultTestLoader, \
    initialTestLoader
from hs_ops import UTBandParallelSetup

# -------------------------------------------------------------------

class UTKPointParallelSetup(UTBandParallelSetup):
    """
    Setup a simple kpoint parallel calculation."""

    # Number of bands
    nbands = 360//10 #*5

    # Spin-polarized, three kpoints
    nspins = 2
    nibzkpts = 3

    gamma = False # can't be gamma point when nibzkpts > 1 ...
    dtype = complex #XXX virtual so far..

    # =================================

    def setUp(self):
        UTBandParallelSetup.setUp(self)

        # Set up kpoint descriptor:
        self.kd = KPointDescriptor(self.nspins, self.nibzkpts, self.kpt_comm, \
            self.gamma, self.dtype)

    def tearDown(self):
        UTBandParallelSetup.tearDown(self)
        del self.kd

    def get_parsizes(self):
        # Careful, overwriting imported GPAW params may cause amnesia in Python.
        from gpaw import parsize, parsize_bands

        # Choose the largest possible parallelization over kpoint/spins
        test_parsize_ks_pairs = gcd(self.nspins*self.nibzkpts, world.size)
        remsize = world.size//test_parsize_ks_pairs

        # If parsize_bands is not set, choose the largest possible
        test_parsize_bands = parsize_bands or gcd(self.nbands, remsize)

        # If parsize_bands is not set, choose as few domains as possible
        test_parsize = parsize or (remsize//test_parsize_bands)

        return test_parsize, test_parsize_bands

    # =================================

        #mynks = self.nspins*self.nibzkpts // self.kpt_comm.size

        # mynu = len(self.kpt_u)
        # s, k = divmod(ks, self.nibzkpts)
        # kpt_rank, u = divmod(k + self.nibzkpts * s, mynu)

    def verify_ks_pair_combination_consistency(self):
        for u in range(self.kd.nks):
            s, k = self.kd.what_is(u)
            self.assertEqual(self.kd.where_is(s, k), u)

        for s in range(self.kd.nspins):
            for k in range(self.kd.nibzkpts):
                u = self.kd.where_is(s, k)
                self.assertTrue(self.kd.what_is(u) == (s,k,))

    def verify_ks_pair_indexing_consistency(self):
        for u in range(self.kd.nks):
            kpt_rank, myu = self.kd.who_has(u)
            self.assertEqual(self.kd.global_index(myu, kpt_rank), u)

        for kpt_rank in range(self.kd.comm.size):
            for myu in range(self.kd.mynks):
                u = self.kd.global_index(myu, kpt_rank)
                self.assertTrue(self.kd.who_has(u) == (kpt_rank, myu))

    def verify_ks_pair_ranking_consistency(self):
        rank_u = self.kd.get_ks_pair_ranks()

        for kpt_rank in range(self.kd.comm.size):
            my_ks_pair_indices = self.kd.get_ks_pair_indices(kpt_rank)
            matches = np.argwhere(rank_u == kpt_rank).ravel()
            self.assertTrue((matches == my_ks_pair_indices).all())

            for myu in range(self.kd.mynks):
                u = self.kd.global_index(myu, kpt_rank)
                self.assertEqual(my_ks_pair_indices[myu], u)

class UTKPointParallelSetup_Blocked(UTKPointParallelSetup):
    __doc__ = UTKPointParallelSetup.__doc__
    parstride_bands = False

class UTKPointParallelSetup_Strided(UTKPointParallelSetup):
    __doc__ = UTKPointParallelSetup.__doc__
    parstride_bands = True

# -------------------------------------------------------------------

class UTRubbishWavefunctionSetup(UTKPointParallelSetup):
    __doc__ = UTKPointParallelSetup.__doc__ + """
    The pseudo wavefunctions are orthonormalized rubbish."""


# -------------------------------------------------------------------

if __name__ in ['__main__', '__builtin__']:
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_kptops.log', verbosity=2)
    else:
        from gpaw.utilities import devnull
        stream = (world.rank == 0) and sys.stdout or devnull
        testrunner = TextTestRunner(stream=stream, verbosity=2)

    parinfo = []
    for test in [UTKPointParallelSetup_Blocked, UTKPointParallelSetup_Strided]:
        info = ['', test.__name__, test.__doc__.strip('\n'), '']
        testsuite = initialTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        assert testresult.wasSuccessful(), 'Initial verification failed!'
        parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
                        tci in testsuite._tests if hasattr(tci, '_parinfo')])
    parinfo = np.unique(np.sort(parinfo)).tolist()

    testcases = [UTRubbishWavefunctionSetup] #XXX no test cases yet
    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_kptops.log for details.')

