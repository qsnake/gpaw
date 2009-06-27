#!/usr/bin/env python

memstats = False
partest = True

import gc
import sys
import time
import numpy as np
import pylab as pl

from copy import copy
from ase import Atoms, molecule
from ase.units import Bohr
from ase.utils.memory import shapeopt, MemorySingleton, MemoryStatistics
from gpaw import parsize, parsize_bands, debug
from gpaw.mpi import world, distribute_cpus, compare_atoms
from gpaw.utilities import gcd
from gpaw.utilities.tools import tri2full, md5_array
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hs_operators import Operator
from gpaw.parameters import InputParameters
from gpaw.xc_functional import XCFunctional
from gpaw.setup import Setup, Setups
from gpaw.lfc import LFC

if partest:
    from gpaw.testing.parunittest import ParallelTestCase as TestCase, \
        ParallelTextTestRunner as TextTestRunner, ParallelTextTestRunner as \
        CustomTextTestRunner, defaultParallelTestLoader as defaultTestLoader
    def CustomTextTestRunner(logname, verbosity=2):
        return TextTestRunner(stream=logname, verbosity=verbosity)
else:
    # Hack to use a feature from ASE 3.1.0 svn.rev. 929 or later.
    # From test/ase3k_versio.py with different requirement.

    ase_required_svnrevision = 929
    try:
        from ase.svnrevision import svnrevision as ase_svnrevision
    except ImportError:
        print 'The hack in this test may is not working. Disregard import errors.'
    else:
        full_ase_svnrevision = ase_svnrevision
        if ase_svnrevision[-1] == 'M':
            ase_svnrevision = ase_svnrevision[:-1]
        if ase_svnrevision.rfind(':') != -1:
            ase_svnrevision = ase_svnrevision[:ase_svnrevision.rfind(':')]
        assert int(ase_svnrevision) >= int(ase_required_svnrevision)

    from ase.test import CustomTestCase as TestCase, CustomTextTestRunner
    from unittest import TextTestRunner, defaultTestLoader

initialTestLoader = copy(defaultTestLoader)
assert hasattr(initialTestLoader, 'testMethodPrefix')
initialTestLoader.testMethodPrefix = 'verify'

# -------------------------------------------------------------------

class UTBandParallelSetup(TestCase):
    """
    Setup a simple band parallel calculation."""

    # Number of bands
    nbands = 360 #*5

    # Spin-paired, single kpoint
    nspins = 1
    nibzkpts = 1

    # Strided or blocked groups
    parstride_bands = None

    # Display plots (if any) or save to file
    showplots = None

    # Mean spacing and number of grid points per axis (G x G x G)
    h = 0.5 / Bohr
    G = 100//5

    def setUp(self):
        # Careful, overwriting parsize_bands may cause amnesia in Python.
        best_parsize_bands = parsize_bands or gcd(self.nbands, world.size)
        domain_comm, kpt_comm, band_comm = distribute_cpus(parsize,
            best_parsize_bands, self.nspins, self.nibzkpts)

        # Set up band descriptor:
        self.bd = BandDescriptor(self.nbands, band_comm, self.parstride_bands)

        # Set up grid descriptor:
        res, ngpts = shapeopt(100, self.G**3, 3, 0.2)
        cell_c = self.h * np.array(ngpts)
        pbc_c = (True, False, True)
        self.gd = GridDescriptor(ngpts, cell_c, pbc_c, domain_comm, parsize)

        # What to do about kpoints?
        self.kpt_comm = kpt_comm

    def tearDown(self):
        del self.bd, self.gd, self.kpt_comm

    # =================================

    def verify_comm_sizes(self):
        if world.size == 1:
            return
        comm_sizes = tuple(comm.size for comm in [world, self.bd.comm, \
                                                  self.gd.comm, self.kpt_comm])
        self._parinfo =  '%d world, %d band, %d domain, %d kpt' % comm_sizes
        self.assertEqual((self.nspins*self.nibzkpts) % self.kpt_comm.size, 0)

    def verify_stride_related(self):
        # Verify that (q1+q2)%B-q1 falls in ]-B;Q[ where Q=B//2+1
        for B in range(1,256):
            Q = B//2+1
            #dqs = []
            #for q1 in range(B):
            #    for q2 in range(Q):
            #        dq = (q1+q2)%B-q1
            #        dqs.append(dq)
            #dqs = np.array(dqs)
            q1s = np.arange(B)[:,np.newaxis]
            q2s = np.arange(Q)[np.newaxis,:]
            dqs = (q1s+q2s)%B-q1s
            self.assertEqual(dqs.min(), -B+1)
            self.assertEqual(dqs.max(), Q-1)

    def verify_indexing_consistency(self):
        for n in range(self.nbands):
            band_rank, myn = self.bd.who_has(n)
            self.assertEqual(self.bd.global_index(myn, band_rank), n)

        for band_rank in range(self.bd.comm.size):
            for myn in range(self.bd.mynbands):
                n = self.bd.global_index(myn, band_rank)
                self.assertTrue(self.bd.who_has(n) == (band_rank, myn))

    def verify_ranking_consistency(self):
        rank_n = self.bd.get_band_ranks()

        for band_rank in range(self.bd.comm.size):
            my_band_indices = self.bd.get_band_indices(band_rank)
            matches = np.argwhere(rank_n == band_rank).ravel()
            self.assertTrue((matches == my_band_indices).all())

            for myn in range(self.bd.mynbands):
                n = self.bd.global_index(myn, band_rank)
                self.assertEqual(my_band_indices[myn], n)


class UTBandParallelSetup_Blocked(UTBandParallelSetup):
    __doc__ = UTBandParallelSetup.__doc__
    parstride_bands = False

class UTBandParallelSetup_Strided(UTBandParallelSetup):
    __doc__ = UTBandParallelSetup.__doc__
    parstride_bands = True

# -------------------------------------------------------------------

def rigid_motion(atoms, center_c=None, rotation_s=None):
    """Apply rigid motion to atoms, translating the COM and/or rotating
    by a given angle around an axis of rotation through the new COM.
    """
    pos_ac = atoms.get_positions()
    if hasattr(atoms, 'constraints') and atoms.constraints:
        raise NotImplementedError('Constrained rigid motion is not defined.')
    
    # Translate molecule such that the COM becomes center_c
    if center_c is not None:
        center_c = np.asarray(center_c)
        assert center_c.shape == (3,)       
        com_c = atoms.get_center_of_mass()
        pos_ac += (center_c-com_c)[np.newaxis,:]
    else:
        center_c = atoms.get_center_of_mass()
    atoms.set_positions(pos_ac)

    # Rotate molecule around COM according to angles
    if rotation_s is not None:
        # The rotation axis is given by spherical angles phi and theta
        assert len(rotation_s) == 3
        from math import sin, cos
        v,phi,theta = rotation_s[:]
        axis = np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
        atoms.rotate(axis, v, center_c)

    assert np.linalg.norm(atoms.get_center_of_mass()-center_c) < 1e-12


def create_random_atoms(gd, nmolecules=10, name='H2O'):
    # Construct pseudo-random gas of H2O molecules:
    assert not gd.is_non_orthogonal()
    cell_c = gd.cell_cv.diagonal()
    atoms = Atoms(cell=cell_c, pbc=gd.pbc_c)
    np.random.seed(nmolecules)
    for m in range(nmolecules):
        amol = molecule(name)
        dpos_ac = amol.get_positions()-amol.get_center_of_mass()[np.newaxis,:]
        mindist = (np.sum((dpos_ac)**2,axis=1)**0.5).max() / Bohr
        delta_c = 1.1 * (1-np.array(gd.pbc_c)) * mindist
        center_c = [np.random.uniform(delta, cell_c[c]-delta) for c,delta in enumerate(delta_c)]
        rotation_s = np.random.uniform(0.0, 2*np.pi, 3) #last angle [0,pi[ really
        rigid_motion(amol, center_c, rotation_s)
        atoms.extend(amol)
    assert compare_atoms(atoms)
    return atoms

def record_memory(wait=0.1):
    assert gc.collect() == 0, 'Uncollected garbage!'
    world.barrier()
    time.sleep(wait)
    mem = MemoryStatistics()
    time.sleep(wait)
    world.barrier()
    return mem

def create_memory_info(mem1, mem2, vmkey='VmData'):
    dm = np.array([(mem2-mem1)[vmkey]], dtype=float)
    dm_r = np.empty(world.size, dtype=float)
    world.all_gather(dm, dm_r)
    return dm_r, ','.join(map(lambda v: '%8.4f MB' % v, dm_r/1024**2.))

# -------------------------------------------------------------------

class UTConstantWavefunctionSetup(UTBandParallelSetup):
    __doc__ = UTBandParallelSetup.__doc__ + """
    The pseudo wavefunctions are constants normalized to their band index."""

    allocated = False
    dtype = None
    blocking = None
    async = None

    def setUp(self):
        global numfigs
        if 'numfigs' not in globals():
            numfigs = 0

        UTBandParallelSetup.setUp(self)

        # Create randomized atoms
        self.atoms = create_random_atoms(self.gd)

        # Do we agree on the atomic positions?
        pos_ac = self.atoms.get_positions()
        pos_rac = np.empty((world.size,)+pos_ac.shape, pos_ac.dtype)
        world.all_gather(pos_ac, pos_rac)
        if (pos_rac-pos_rac[world.rank,...][np.newaxis,...]).any():
            raise RuntimeError('Discrepancy in atomic positions detected.')

        # Create setups for atoms
        self.Z_a = self.atoms.get_atomic_numbers()
        par = InputParameters()
        xcfunc = XCFunctional('LDA')
        self.setups = Setups(self.Z_a, par.setups, par.basis, self.nspins, \
                             par.lmax, xcfunc)

        # Create atomic projector overlaps
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        self.rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.pt = LFC(self.gd, [setup.pt_j for setup in self.setups], \
                      self.kpt_comm, dtype=self.dtype)
        self.pt.set_positions(spos_ac)

        if memstats:
            # Hack to scramble heap usage into steady-state level
            HEAPSIZE = 25*1024**2
            for i in range(100):
                data = np.empty(np.random.uniform(0,HEAPSIZE//8), float)
                del data
            self.mem_pre = record_memory()
            self.mem_alloc = None
            self.mem_test = None

        # Stuff for pseudo wave functions and projections
        if self.dtype == complex:
            self.gamma = 1j**(1.0/self.nbands)
        else:
            self.gamma = 1.0

        self.psit_nG = None
        self.P_ani = None
        self.Qeff_a = None
        self.Qtotal = None

        self.allocate()

    def tearDown(self):
        UTBandParallelSetup.tearDown(self)
        del self.P_ani, self.psit_nG
        del self.pt, self.setups, self.atoms
        if memstats:
            self.print_memory_summary()
            del self.mem_pre, self.mem_alloc, self.mem_test
        self.allocated = False

    def print_memory_summary(self):
        if not memstats:
            raise RuntimeError('No memory statistics were recorded!')

        if world.rank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

        dm_r, dminfo = create_memory_info(MemorySingleton(), self.mem_pre)
        if world.rank == 0:
            print 'overhead: %s -> %8.4f MB' % (dminfo, dm_r.sum()/1024**2.)

        dm_r, dminfo = create_memory_info(self.mem_pre, self.mem_alloc)
        if world.rank == 0:
            print 'allocate: %s -> %8.4f MB' % (dminfo, dm_r.sum()/1024**2.)

        dm_r, dminfo = create_memory_info(self.mem_alloc, self.mem_test)
        if world.rank == 0:
            print 'test-use: %s -> %8.4f MB' % (dminfo, dm_r.sum()/1024**2.)

    def allocate(self):
        """
        Allocate constant wavefunctions and their projections according to::

                                  _____    i*phase*(n-m)
           <psi |psi > =  Q    * V m*n  * e
               n    n'     total

        """
        if self.allocated:
            raise RuntimeError('Already allocated!')

        self.allocate_wavefunctions()
        self.allocate_projections()

        Qlocal = sum([Qeff for Qeff in self.Qeff_a.values()]) # never np.sum!
        self.Qtotal = self.gd.comm.sum(Qlocal)

        band_indices = np.arange(self.nbands).astype(self.dtype)
        z = self.gamma**band_indices * band_indices**0.5
        self.S0_nn = (1. + self.Qtotal) * np.outer(z.conj(), z)

        self.allocated = True

        if memstats:
            self.mem_alloc = record_memory()

    def allocate_wavefunctions(self):
        """
        Allocate constant pseudo wavefunctions according to::

             ~    ~        _____    i*phase*(n-m)
           <psi |psi > =  V m*n  * e
               n    n'

        """
        if self.allocated:
            raise RuntimeError('Already allocated!')

        # Fill in wave functions
        gpts_c = self.gd.get_size_of_global_array()
        self.psit_nG = self.gd.empty(self.bd.mynbands, self.dtype)
        for myn, psit_G in enumerate(self.psit_nG):
            n = self.bd.global_index(myn)
            # Fill psit_nG: | psit_n > = exp(i*phase*n) * sqrt(n) / sqrt(V)
            psit_G[:] = self.gamma**n * n**0.5 / (self.gd.dv * gpts_c.prod())**0.5

    def allocate_projections(self):
        """
        Construct dummy projection of pseudo wavefunction according to::
           ___
           \     ~   ~a    a   ~a  ~             1     _____    i*phase*(n-m)
            )  <psi |p > dO   <p |psi > =  +/-  --- * V m*n  * e
           /___    n  i    ii'  i'   n'          Z
            ii'                                   a

        """
        if self.allocated:
            raise RuntimeError('Already allocated!')

        # Fill in projector overlaps
        my_band_indices = self.bd.get_band_indices()
        my_atom_indices = np.argwhere(self.gd.comm.rank == self.rank_a).ravel()

        self.Qeff_a = {}
        self.P_ani = self.pt.dict(self.bd.mynbands)
        for a in my_atom_indices:
            ni = self.setups[a].ni
            # Fill P_ni: <p_i | psit_n > = beta_i * exp(i*phase*n) * sqrt(n)
            #
            #  |  ____                  |
            #  |  \        *   a        |      1
            #  |   )   beta   O   beta  |  =  ----
            #  |  /___     i   ij     j |      Z
            #  |    ij                  |       a
            #
            # Substitution by linear transformation: beta_i ->  O_ij alpha_j,
            # where we start out with some initial non-constant vector:
            alpha_i = np.exp(-np.arange(ni).astype(self.dtype)/ni)
            try:
                # Try Cholesky decomposition O_ii = L_ii * L_ii^dag
                L_ii = np.linalg.cholesky(self.setups[a].O_ii)
                alpha_i /= np.vdot(alpha_i, alpha_i)**0.5
                beta_i = np.linalg.solve(L_ii.T.conj(), alpha_i)
            except np.linalg.LinAlgError:
                # Eigenvector decomposition O_ii = V_ii * W_ii * V_ii^dag
                W_i, V_ii = np.linalg.eigh(self.setups[a].O_ii)
                alpha_i /= np.abs(np.vdot(alpha_i, 
                                          np.dot(np.diag(W_i), alpha_i)))**0.5
                beta_i = np.linalg.solve(V_ii.T.conj(), alpha_i)

            # Normalize according to plus/minus charge
            beta_i /= self.Z_a[a]**0.5
            self.Qeff_a[a] = np.vdot(beta_i, np.dot(self.setups[a].O_ii, \
                                                    beta_i)).real
            self.P_ani[a][:] = np.outer(self.gamma**my_band_indices \
                                        * my_band_indices**0.5, beta_i)

    def check_and_plot(self, A_nn, A0_nn, digits, keywords='none'):
        # Construct fingerprint of input matrices for comparison
        fingerprint = np.array([md5_array(A_nn, numeric=True),
                                md5_array(A0_nn, numeric=True)])

        # Compare fingerprints across all processors
        fingerprints = np.empty((world.size, 2), np.int64)
        world.all_gather(fingerprint, fingerprints)
        if fingerprints.ptp(0).any():
            raise RuntimeError('Distributed matrices are not identical!')

        # If assertion fails, catch temporarily while plotting, then re-raise
        try:
            self.assertAlmostEqual(np.abs(A_nn-A0_nn).max(), 0, digits)
        except AssertionError:
            if world.rank == 0:
                global numfigs
                fig = pl.figure(numfigs)
                ax = pl.axes()
                ax.set_title('%s: %s' % (self.__class__.__name__, keywords))
                im = ax.imshow(np.abs(A_nn-A0_nn), cmap=pl.cm.jet)
                pl.colorbar(im)
                numfigs += 1
                if not self.showplots:
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    img = 'ut_hsops_%s_%s.png' % (self.__class__.__name__, \
                        '_'.join(keywords.split(',')))
                    FigureCanvasAgg(fig).print_figure(img.lower(), dpi=90)
            raise

    def get_optimal_number_of_blocks(self, blocking='fast'):
        """Estimate the optimal number of blocks for band parallelization.

        The local number of bands ``mynbands`` must be divisible by the
        number of blocks ``nblocks``. The number of blocks determines how
        many parallel send/receive operations are performed, as well as
        the added memory footprint of the required send/receive buffers.

        ``blocking``  ``nblocks``      Description
        ============  =============    ========================================
        'fast'        ``1``            Heavy on memory, more accurate and fast.
        'light'       ``mynbands``     Light on memory, less accurate and slow.
        'best'        ``...``          Algorithmically balanced middle ground.
        ``int``       [1;mynbands]     Optional
        """

        #if self.bd.comm.size == 1:
        #    return 1

        if blocking == 'fast':
            return 1
        elif blocking == 'light':
            return self.bd.mynbands
        elif blocking == 'best':
            # Estimated number of my bands per block (mynbands/nblocks)
            blocksize_bands = 5

            # Find all divisors of mynbands and pick the closest match
            js = np.array([j for j in range(1,self.bd.mynbands+1) \
                           if self.bd.mynbands%j==0], dtype=int)
            jselect = np.argmin(abs(blocksize_bands-self.bd.mynbands/js))
            return js[jselect]
        else:
            nblocks = blocking
            assert nblocks in range(1,self.bd.mynbands+1)
            assert self.bd.mynbands % nblocks == 0
            return nblocks

    # =================================

    def test_wavefunction_content(self):
        # Integrate diagonal brakets of pseudo wavefunctions
        gpts_c = self.gd.get_size_of_global_array()

        intpsit_myn = self.bd.empty(dtype=self.dtype)
        for myn, psit_G in enumerate(self.psit_nG):
            n = self.bd.global_index(myn)
            intpsit_myn[myn] = np.vdot(psit_G, psit_G) * self.gd.dv
        self.gd.comm.sum(intpsit_myn)

        if memstats:
            self.mem_test = record_memory()

        my_band_indices = self.bd.get_band_indices()
        self.assertAlmostEqual(np.abs(intpsit_myn-my_band_indices).max(), 0, 9)

        intpsit_n = self.bd.collect(intpsit_myn, broadcast=True)
        self.assertAlmostEqual(np.abs(intpsit_n-np.arange(self.nbands)).max(), 0, 9)

    def test_projection_content(self):
        # Distribute inverse effective charges to everybody in domain
        all_Qeff_a = np.empty(len(self.atoms), dtype=float)
        for a,rank in enumerate(self.rank_a):
            if rank == self.gd.comm.rank:
                Qeff = np.array([self.Qeff_a[a]])
            else:
                Qeff = np.empty(1, dtype=float)
            self.gd.comm.broadcast(Qeff, rank)
            all_Qeff_a[a] = Qeff

        # Check absolute values consistency of inverse effective charges
        self.assertAlmostEqual(np.abs(1./self.Z_a-np.abs(all_Qeff_a)).max(), 0, 9)

        # Check sum of inverse effective charges against total
        self.assertAlmostEqual(all_Qeff_a.sum(), self.Qtotal, 9)

        # Make sure that we all agree on inverse effective charges
        fingerprint = np.array([md5_array(all_Qeff_a, numeric=True)])
        all_fingerprints = np.empty(world.size, fingerprint.dtype)
        world.all_gather(fingerprint, all_fingerprints)
        if all_fingerprints.ptp(0).any():
            raise RuntimeError('Distributed eff. charges are not identical!')

    def test_overlaps_hermitian(self):
        # Set up Hermitian overlap operator:
        S = lambda x: x
        dS = lambda a, P_ni: np.dot(P_ni, self.setups[a].O_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = Operator(self.bd, self.gd, nblocks, self.async, True)
        #S_nn = overlap.calculate_matrix_elements(self.psit_nG, \
        #    self.P_ani, S, dS).conj() # conjugate to get <psit_m|A|psit_n>
        #tri2full(S_nn) # lower to upper...
        S_nn = overlap.calculate_matrix_elements(self.psit_nG, \
            self.P_ani, S, dS).T.copy() # transpose to get <psit_m|A|psit_n>
        tri2full(S_nn, 'U') # upper to lower...

        if self.bd.comm.rank == 0:
            self.gd.comm.broadcast(S_nn, 0)
        self.bd.comm.broadcast(S_nn, 0)

        if memstats:
            self.mem_test = record_memory()

        self.check_and_plot(S_nn, self.S0_nn, 9, 'overlaps,hermitian')

    def test_overlaps_nonhermitian(self):
        alpha = np.random.normal(size=1).astype(self.dtype)
        if self.dtype == complex:
            alpha += 1j*np.random.normal(size=1)
        world.sum(alpha)
        alpha /= world.size

        # Set up non-Hermitian overlap operator:
        S = lambda x: alpha*x
        dS = lambda a, P_ni: np.dot(alpha*P_ni, self.setups[a].O_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = Operator(self.bd, self.gd, nblocks, self.async, False)
        S_nn = overlap.calculate_matrix_elements(self.psit_nG, \
            self.P_ani, S, dS).T.copy() # transpose to get <psit_m|A|psit_n>

        if self.bd.comm.rank == 0:
            self.gd.comm.broadcast(S_nn, 0)
        self.bd.comm.broadcast(S_nn, 0)

        if memstats:
            self.mem_test = record_memory()

        self.check_and_plot(S_nn, alpha*self.S0_nn, 9, 'overlaps,nonhermitian')

# -------------------------------------------------------------------

def UTConstantWavefunctionFactory(dtype, parstride_bands, blocking, async):
    sep = '_'
    classname = 'UTConstantWavefunctionSetup' \
    + sep + {float:'Float', complex:'Complex'}[dtype] \
    + sep + {False:'Blocked', True:'Strided'}[parstride_bands] \
    + sep + {'fast':'Fast', 'light':'Light', 'best':'Best'}[blocking] \
    + sep + {False:'Synchronous', True:'Asynchronous'}[async]
    class MetaPrototype(UTConstantWavefunctionSetup, object):
        __doc__ = UTConstantWavefunctionSetup.__doc__
        dtype = dtype
        parstride_bands = parstride_bands
        blocking = blocking
        async = async
        showplots = (__name__ == '__main__')
    MetaPrototype.__name__ = classname
    return MetaPrototype

# -------------------------------------------------------------------

if __name__ in ['__main__', '__builtin__']:
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_hsops.log', verbosity=2)
    else:
        from gpaw.utilities import devnull
        stream = (world.rank == 0) and sys.stdout or devnull
        testrunner = TextTestRunner(stream=stream, verbosity=2)

    parinfo = []
    for test in [UTBandParallelSetup]:
        info = ['', test.__name__, test.__doc__.strip('\n'), '']
        testsuite = initialTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        assert testresult.wasSuccessful(), 'Initial verification failed!'
        parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
                        tci in testsuite._tests if hasattr(tci, '_parinfo')])

    testcases = []
    for dtype in [float, complex]:
        for parstride_bands in [False, True]:
            for blocking in ['fast', 'best']: # 'light'
                for async in [False, True]:
                    testcases.append(UTConstantWavefunctionFactory(dtype, \
                        parstride_bands, blocking, async))

    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_hsops.log for details.')

    global numfigs
    if numfigs > 0:
        pl.show()

