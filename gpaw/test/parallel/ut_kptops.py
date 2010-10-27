#!/usr/bin/env python

#partest = True

import gc
import sys
import time
import numpy as np

from ase.units import Bohr
from ase.utils.memory import MemorySingleton, MemoryStatistics
from gpaw.mpi import world, distribute_cpus
from gpaw.utilities import gcd
#from gpaw.utilities.tools import tri2full, md5_array, gram_schmidt
#from gpaw.kpt_descriptor import KPointDescriptorOld as KPointDescriptor
from gpaw.paw import kpts2ndarray
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor, KPointDescriptorOld #XXX
from gpaw.parameters import InputParameters
from gpaw.setup import SetupData, Setups
from gpaw.xc import XC

#from gpaw.hs_operators import MatrixOperator
#from gpaw.parameters import InputParameters
#from gpaw.xc_functional import XCFunctional
#from gpaw.setup import Setup, Setups
#from gpaw.lfc import LFC

from gpaw.test.ut_common import ase_svnversion, shapeopt, TestCase, \
    TextTestRunner, CustomTextTestRunner, defaultTestLoader, \
    initialTestLoader, create_random_atoms, create_parsize_minbands

# -------------------------------------------------------------------

p = InputParameters(spinpol=False, usesymm=None)
xc = XC(p.xc)
p.setups = dict([(symbol, SetupData(symbol, xc.name)) for symbol in 'HO'])

class UTKPointParallelSetup(TestCase):
    """
    Setup a simple kpoint parallel calculation."""

    # Number of bands
    nbands = 1

    # Spin-polarized
    nspins = 1

    # Mean spacing and number of grid points per axis (G x G x G)
    h = 0.25 / Bohr
    G = 48

    ## Symmetry-reduction of k-points TODO
    #symmetry = p.usesymm #XXX 'None' is an allowed value!!!

    # Whether spin/k-points are equally distributed (determines nibzkpts)
    equipartition = None
    nibzkpts = None

    gamma = False # can't be gamma point when nibzkpts > 1 ...
    dtype = complex #XXX virtual so far..

    # =================================

    def setUp(self):
        for virtvar in ['equipartition']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        kpts = {'even' : (12,1,2), \
                'prime': (23,1,1)}[self.equipartition]

        #primes = [i for i in xrange(50,1,-1) if ~np.any(i%np.arange(2,i)==0)]
        bzk_kc = kpts2ndarray(kpts)
        assert p.usesymm == None
        self.nibzkpts = len(bzk_kc)

        #parsize, parsize_bands = create_parsize_minbands(self.nbands, world.size)
        parsize, parsize_bands = 1, 1 #XXX
        assert self.nbands % np.prod(parsize_bands) == 0
        domain_comm, kpt_comm, band_comm = distribute_cpus(parsize,
            parsize_bands, self.nspins, self.nibzkpts)

        # Set up band descriptor:
        self.bd = BandDescriptor(self.nbands, band_comm, p.parallel['stridebands'])

        # Set up grid descriptor:
        res, ngpts = shapeopt(300, self.G**3, 3, 0.2)
        cell_c = self.h * np.array(ngpts)
        pbc_c = (True, False, True)
        self.gd = GridDescriptor(ngpts, cell_c, pbc_c, domain_comm, parsize)

        # Create randomized gas-like atomic configuration
        self.atoms = create_random_atoms(self.gd)

        # Create setups
        Z_a = self.atoms.get_atomic_numbers()
        self.setups = Setups(Z_a, p.setups, p.basis, p.lmax, xc)
        self.natoms = len(self.setups)

        # Set up kpoint descriptor:
        self.kd = KPointDescriptor(bzk_kc, self.nspins)
        self.kd.set_symmetry(self.atoms, self.setups, p.usesymm)
        self.kd.set_communicator(kpt_comm)

        #self.kd_old = KPointDescriptorOld(self.nspins, self.nibzkpts, \
        #    kpt_comm, self.gamma, self.dtype) #XXX

    def tearDown(self):
        del self.bd, self.gd, self.kd #self.kd_old
        del self.setups, self.atoms

    def get_parsizes(self): #XXX NO LONGER IN UT_HSOPS?!?
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

    def verify_comm_sizes(self): #TODO needs work
        if world.size == 1:
            return
        comm_sizes = tuple([comm.size for comm in [world, self.bd.comm, \
                                                   self.gd.comm, self.kd.comm]])
        self._parinfo =  '%d world, %d band, %d domain, %d kpt' % comm_sizes
        #self.assertEqual((self.nspins*self.nibzkpts) % self.kd.comm.size, 0) #XXX

    def verify_slice_consistency(self):
        for kpt_rank in range(self.kd.comm.size):
            uslice = self.kd.get_slice(kpt_rank)
            myus = np.arange(*uslice.indices(self.kd.nks))
            for myu,u in enumerate(myus):
                self.assertEqual(self.kd.who_has(u), (kpt_rank, myu))

    def verify_combination_consistency(self):
        for u in range(self.kd.nks):
            s, k = self.kd.what_is(u)
            self.assertEqual(self.kd.where_is(s, k), u)

        for s in range(self.kd.nspins):
            for k in range(self.kd.nibzkpts):
                u = self.kd.where_is(s, k)
                self.assertEqual(self.kd.what_is(u), (s,k,))

    def verify_indexing_consistency(self):
        for u in range(self.kd.nks):
            kpt_rank, myu = self.kd.who_has(u)
            self.assertEqual(self.kd.global_index(myu, kpt_rank), u)

        for kpt_rank in range(self.kd.comm.size):
            for myu in range(self.kd.get_count(kpt_rank)):
                u = self.kd.global_index(myu, kpt_rank)
                self.assertEqual(self.kd.who_has(u), (kpt_rank, myu))

    def verify_ranking_consistency(self):
        ranks = self.kd.get_ranks()

        for kpt_rank in range(self.kd.comm.size):
            my_indices = self.kd.get_indices(kpt_rank)
            matches = np.argwhere(ranks == kpt_rank).ravel()
            self.assertTrue((matches == my_indices).all())
            for myu in range(self.kd.get_count(kpt_rank)):
                u = self.kd.global_index(myu, kpt_rank)
                self.assertEqual(my_indices[myu], u)

class UTKPointParallelSetup_Even(UTKPointParallelSetup):
    __doc__ = UTKPointParallelSetup.__doc__
    equipartition = 'even'

class UTKPointParallelSetup_Prime(UTKPointParallelSetup):
    __doc__ = UTKPointParallelSetup.__doc__
    equipartition = 'prime'

# -------------------------------------------------------------------

class UTRubbishWavefunctionSetup(UTKPointParallelSetup):
    __doc__ = UTKPointParallelSetup.__doc__ + """
    The pseudo wavefunctions are orthonormalized rubbish."""

    def test_nothing(self): #TODO
        pass

# -------------------------------------------------------------------

def UTRubbishWavefunctionFactory(equipartition):
    sep = '_'
    classname = 'UTRubbishWavefunctionSetup' \
    + sep + {'even':'Even', 'prime':'Prime'}[equipartition]
    class MetaPrototype(UTRubbishWavefunctionSetup, object):
        __doc__ = UTRubbishWavefunctionSetup.__doc__
        equipartition = equipartition
    MetaPrototype.__name__ = classname
    return MetaPrototype

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
    for test in [UTKPointParallelSetup_Even, UTKPointParallelSetup_Prime]:
        info = ['', test.__name__, test.__doc__.strip('\n'), '']
        testsuite = initialTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        assert testresult.wasSuccessful(), 'Initial verification failed!'
        parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
                        tci in testsuite._tests if hasattr(tci, '_parinfo')])
    parinfo = np.unique(np.sort(parinfo)).tolist()

    testcases = []
    for equipartition in ['even', 'prime']:
        testcases.append(UTRubbishWavefunctionFactory(equipartition))

    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_kptops.log for details.')

