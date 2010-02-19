#!/usr/bin/env python

import sys
import numpy as np

from gpaw import debug
from gpaw.mpi import world
from gpaw.utilities.tools import tri2full
from gpaw.hs_operators import MatrixOperator
from gpaw.matrix_descriptor import BlacsBandMatrixDescriptor

from gpaw.utilities import scalapack
#from gpaw.utilities.blacs import scalapack_set
from gpaw.blacs import parallelprint, BlacsBandDescriptor

if debug:
    np.set_printoptions(linewidth=168) #XXX large xterm width

# -------------------------------------------------------------------

from gpaw.test.ut_common import ase_svnrevision, TextTestRunner, \
    CustomTextTestRunner, defaultTestLoader, initialTestLoader

memstats = False
if memstats:
    # Developer use of this feature requires ASE 3.1.0 svn.rev. 905 or later.
    assert ase_svnrevision >= 905 # wasn't bug-free untill 973!
    from ase.utils.memory import MemorySingleton, MemoryStatistics

from gpaw.test.parallel.ut_hsops import UTBandParallelSetup, \
                                        UTConstantWavefunctionSetup

# -------------------------------------------------------------------

class UTBandParallelBlacsSetup(UTBandParallelSetup):
    """
    Setup a simple band parallel calculation using BLACS."""

    # Number of bands
    nbands = 36 # XXX a bit low

    def setUp(self):
        UTBandParallelSetup.setUp(self)
        # create blacs grid and descriptors here
        cpus = self.bd.comm.size * self.gd.comm.size
        self.mcpus = int(cpus**0.5)
        self.ncpus = cpus//self.mcpus

    def tearDown(self):
        # clean-up blacs grid and descriptors here
        UTBandParallelSetup.tearDown(self)

    # =================================

    def verify_comm_sizes(self):
        if world.size == 1:
            return
        comm_sizes = tuple([comm.size for comm in [world, self.bd.comm, \
                                                   self.gd.comm, self.kpt_comm]])
        comm_sizes += (self.mcpus, self.ncpus)
        self._parinfo =  '%d world, %d band, %d domain, %d kpt, %d x %d BLACS' % comm_sizes
        self.assertEqual((self.nspins*self.nibzkpts) % self.kpt_comm.size, 0)

    def verify_blacs_stuff(self):
        pass

class UTBandParallelBlacsSetup_Blocked(UTBandParallelBlacsSetup):
    __doc__ = UTBandParallelBlacsSetup.__doc__
    parstride_bands = False

class UTBandParallelBlacsSetup_Strided(UTBandParallelSetup):
    __doc__ = UTBandParallelBlacsSetup.__doc__
    parstride_bands = True

# -------------------------------------------------------------------

class UTConstantWavefunctionBlacsSetup(UTConstantWavefunctionSetup,
                                       UTBandParallelBlacsSetup):
    __doc__ = UTBandParallelBlacsSetup.__doc__ + """
    The pseudo wavefunctions are constants normalized to their band index."""

    def setUp(self):
        UTBandParallelBlacsSetup.setUp(self) #XXX diamond inheritance
        UTConstantWavefunctionSetup.setUp(self)

    def tearDown(self):
        UTConstantWavefunctionSetup.tearDown(self)
        #UTBandParallelBlacsSetup.tearDown(self) #XXX diamond inheritance

    # =================================

    def test_overlaps_hermitian(self):
        # Set up Hermitian overlap operator:
        S = lambda x: x
        dS = lambda a, P_ni: np.dot(P_ni, self.setups[a].dO_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = MatrixOperator(self.bd, self.gd, nblocks, self.async, True)
        overlap.bmd = BlacsBandMatrixDescriptor(self.bd, self.gd) # XXX override full-matrix descriptor
        overlap.bmd.redistribute_input = lambda A_nN: A_nN # XXX override full-matrix disassembly
        overlap.bmd.redistribute_output = lambda A_Nn: A_Nn # XXX override full-matrix assembly

        #S_nn = overlap.calculate_matrix_elements(self.psit_nG, \
        #    self.P_ani, S, dS).T.copy() # transpose to get <psit_m|A|psit_n>
        #tri2full(S_nn, 'U') # upper to lower...

        #if self.bd.comm.rank == 0:
        #    self.gd.comm.broadcast(S_nn, 0)
        #self.bd.comm.broadcast(S_nn, 0)

        S_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.gd.comm.broadcast(S_Nn, 0)

        if memstats:
            self.mem_test = record_memory()

        S_NN = self.bd.collect(S_Nn.T.copy(), broadcast=True)
        tri2full(S_NN, 'U') # upper to lower...
        self.check_and_plot(S_NN, self.S0_nn, 9, 'overlaps,hermitian')

    def test_overlaps_nonhermitian(self):
        alpha = np.random.normal(size=1).astype(self.dtype)
        if self.dtype == complex:
            alpha += 1j*np.random.normal(size=1)
        world.broadcast(alpha, 0)

        # Set up non-Hermitian overlap operator:
        S = lambda x: alpha*x
        dS = lambda a, P_ni: np.dot(alpha*P_ni, self.setups[a].dO_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = MatrixOperator(self.bd, self.gd, nblocks, self.async, False)
        overlap.bmd = BlacsBandMatrixDescriptor(self.bd, self.gd) # XXX override full-matrix descriptor
        overlap.bmd.redistribute_input = lambda A_nN: A_nN # XXX override full-matrix disassembly
        overlap.bmd.redistribute_output = lambda A_Nn: A_Nn # XXX override full-matrix assembly

        #S_nn = overlap.calculate_matrix_elements(self.psit_nG, \
        #    self.P_ani, S, dS).T.copy() # transpose to get <psit_m|A|psit_n>

        #if self.bd.comm.rank == 0:
        #    self.gd.comm.broadcast(S_nn, 0)
        #self.bd.comm.broadcast(S_nn, 0)

        S_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.gd.comm.broadcast(S_Nn, 0)

        if memstats:
            self.mem_test = record_memory()

        S_NN = self.bd.collect(S_Nn.T.copy(), broadcast=True)
        self.check_and_plot(S_NN, alpha*self.S0_nn, 9, 'overlaps,nonhermitian')

    def test_trivial_cholesky(self):
        # Set up Hermitian overlap operator:
        S = lambda x: x
        dS = lambda a, P_ni: np.dot(P_ni, self.setups[a].dO_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = MatrixOperator(self.bd, self.gd, nblocks, self.async, True)
        overlap.bmd = BlacsBandMatrixDescriptor(self.bd, self.gd) # XXX override full-matrix descriptor
        overlap.bmd.redistribute_input = lambda A_nN: A_nN # XXX override full-matrix disassembly
        overlap.bmd.redistribute_output = lambda A_Nn: A_Nn # XXX override full-matrix assembly
        S_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.assertEqual(S_Nn.shape, (self.bd.nbands,self.bd.mynbands))

        # Known starting point of SI_nn = <psit_m|S+alpha*I|psit_n>
        #I_Nn = np.eye(self.bd.nbands, self.bd.mynbands, -self.bd.beg) # XXX only valid for blocked groups!
        I_NN = np.eye(self.bd.nbands)
        I_nN = np.empty((self.bd.mynbands, self.bd.nbands), dtype=I_NN.dtype)
        self.bd.distribute(I_NN, I_nN)
        I_Nn = I_nN.T.copy()
        del I_nN

        alpha = 1e-3 # shift eigenvalues away from zero
        SI_Nn = S_Nn + alpha * I_Nn

        if 0:
            SI_NN = self.bd.collect(SI_Nn.T.copy(), broadcast=True)
            tri2full(SI_NN, 'U')
            SI_nN = np.empty((self.bd.mynbands, self.bd.nbands), dtype=SI_NN.dtype)
            self.bd.distribute(SI_NN, SI_nN)
            SI_Nn[:] = SI_nN.T
            del SI_nN

        if debug and self.gd.comm.rank == 0:
            if self.bd.comm.rank == 0: print 'SI_Nn:'
            parallelprint(self.bd.comm, SI_Nn)

        blocksize = 6
        bbd = BlacsBandDescriptor(world, self.gd, self.bd, self.kpt_comm,
                                  self.mcpus, self.ncpus, blocksize)

        # We would create C_nN in the real-space code this way.
        C_nN = np.empty((self.bd.mynbands, self.bd.nbands), dtype=S_Nn.dtype)
        diagonalizer = bbd.get_diagonalizer()
        diagonalizer.inverse_cholesky(SI_Nn, C_nN)
        self.assertEqual(C_nN.shape, (self.bd.mynbands,self.bd.nbands))

        if debug and self.gd.comm.rank == 0:
            if self.bd.comm.rank == 0: print 'C_nN:'
            parallelprint(self.bd.comm, C_nN)

        self.psit_nG = overlap.matrix_multiply(C_nN, self.psit_nG, self.P_ani)
        D_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.gd.comm.broadcast(D_Nn, 0)

        if debug and self.gd.comm.rank == 0:
            if self.bd.comm.rank == 0: print 'D_Nn:'
            parallelprint(self.bd.comm, D_Nn)

        # D_nn = C_nn^dag * S_nn * C_nn = I_nn - alpha * C_nn^dag * C_nn
        C_NN = self.bd.collect(C_nN, broadcast=True).T.copy()
        D_NN = self.bd.collect(D_Nn.T.copy(), broadcast=True)
        tri2full(D_NN, 'U') # upper to lower...
        D0_NN = I_NN - alpha * np.dot(C_NN.T.conj(), C_NN)

        C0_NN = np.linalg.inv(np.linalg.cholesky(self.S0_nn + alpha*I_NN).T.conj())

        if debug and self.gd.comm.rank == 0 and self.bd.comm.rank == 0:
            print 'C_NN:\n', C_NN
            print 'C0_NN:\n', C0_NN
            print 'D_NN:\n', D_NN
            print 'D0_NN:\n', D0_NN

        self.check_and_plot(C_NN, C0_NN, 6, 'trivial,cholesky')
        self.check_and_plot(D_NN, D0_NN, 6, 'trivial,cholesky') #XXX precision

    def test_trivial_diagonalize(self): #XXX XXX XXX
        # Known starting point of S_nn = <psit_m|S|psit_n>
        S_nn = self.S0_nn

        # Eigenvector decomposition S_nn = V_nn * W_nn * V_nn^dag
        # Utilize the fact that they are analytically known (cf. Maple)
        band_indices = np.arange(self.nbands)
        V_nn = np.eye(self.nbands).astype(self.dtype)
        if self.dtype == complex:
            V_nn[1:,1] = np.conj(self.gamma)**band_indices[1:] * band_indices[1:]**0.5
            V_nn[1,2:] = -self.gamma**band_indices[1:-1] * band_indices[2:]**0.5
        else:
            V_nn[2:,1] = band_indices[2:]**0.5
            V_nn[1,2:] = -band_indices[2:]**0.5

        W_n = np.zeros(self.nbands).astype(self.dtype)
        W_n[1] = (1. + self.Qtotal) * self.nbands * (self.nbands - 1) / 2.

        # Find the inverse basis
        Vinv_nn = np.linalg.inv(V_nn)

        # Test analytical eigenvectors for consistency against analytical S_nn
        D_nn = np.dot(Vinv_nn, np.dot(S_nn, V_nn))
        self.assertAlmostEqual(np.abs(D_nn.diagonal()-W_n).max(), 0, 8)
        self.assertAlmostEqual(np.abs(np.tril(D_nn, -1)).max(), 0, 4)
        self.assertAlmostEqual(np.abs(np.triu(D_nn, 1)).max(), 0, 4)
        del Vinv_nn, D_nn

        # Set up Hermitian overlap operator:
        S = lambda x: x
        dS = lambda a, P_ni: np.dot(P_ni, self.setups[a].dO_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = MatrixOperator(self.bd, self.gd, nblocks, self.async, True)
        overlap.bmd = BlacsBandMatrixDescriptor(self.bd, self.gd) # XXX override full-matrix descriptor
        overlap.bmd.redistribute_input = lambda A_nN: A_nN # XXX override full-matrix disassembly
        overlap.bmd.redistribute_output = lambda A_Nn: A_Nn # XXX override full-matrix assembly
        S_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.assertEqual(S_Nn.shape, (self.bd.nbands,self.bd.mynbands))

        blocksize = 6
        bbd = BlacsBandDescriptor(world, self.gd, self.bd, self.kpt_comm,
                                  self.mcpus, self.ncpus, blocksize)

        # We would create C_nN in the real-space code this way.
        C_nN = np.empty((self.bd.mynbands, self.bd.nbands), dtype=S_Nn.dtype)
        diagonalizer = bbd.get_diagonalizer()
        eps_n = np.zeros(self.bd.mynbands) # XXX dtype?
        diagonalizer.diagonalize(S_Nn, C_nN, eps_n)
        self.assertEqual(C_nN.shape, (self.bd.mynbands,self.bd.nbands))
        #if self.gd.comm.rank == 0:
        #    print world.rank, 'eps_n:', self.bd.collect(eps_n, broadcast=True), 'ref_n:', W_n
        eps_N = self.bd.collect(eps_n, broadcast=True)
        self.assertAlmostEqual(np.abs(np.sort(eps_N)-np.sort(W_n)).max(), 0, 9)

        ## Crude redistribute from row to column layout
        #C_NN = self.bd.collect(C_nN, broadcast=True)
        #self.bd.distribute(C_NN.T.copy(), C_nN)
        #C_Nn = C_nN.T.copy()

        # Rotate wavefunctions to diagonalize the overlap
        self.psit_nG = overlap.matrix_multiply(C_nN, self.psit_nG, self.P_ani)

        # Recaulculate the overlap matrix, which should now be diagonal
        D_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.gd.comm.broadcast(D_Nn, 0) #XXX?

        if memstats:
            self.mem_test = record_memory()

        # D_nn = C_nn^dag * S_nn * C_nn = W_n since Q_nn^dag = Q_nn^(-1)
        C_NN = self.bd.collect(C_nN, broadcast=True).T.copy()
        D0_nn = np.dot(C_NN.T.conj(), np.dot(S_nn, C_NN))
        D_NN = self.bd.collect(D_Nn.T.copy(), broadcast=True)
        tri2full(D_NN, 'U') # upper to lower...
        #self.assertAlmostEqual(np.abs(D0_nn-np.diag(W_n)).max(), 0, 9)
        self.check_and_plot(D_NN, D0_nn, 9, 'trivial,diagonalize')

    def test_multiply_randomized(self):
        # Known starting point of S_nn = <psit_m|S|psit_n>
        S_NN = self.S0_nn

        if self.dtype == complex:
            C_NN = np.random.uniform(size=self.nbands**2) * \
                np.exp(1j*np.random.uniform(0,2*np.pi,size=self.nbands**2))
        else:
            C_NN = np.random.normal(size=self.nbands**2)
        C_NN = C_NN.reshape((self.nbands,self.nbands)) / np.linalg.norm(C_NN,2)
        world.broadcast(C_NN, 0)
        C_nN = np.empty((self.bd.mynbands, self.bd.nbands), dtype=C_NN.dtype)
        self.bd.distribute(C_NN.T.copy(), C_nN)

        # Set up Hermitian overlap operator:
        S = lambda x: x
        dS = lambda a, P_ni: np.dot(P_ni, self.setups[a].dO_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = MatrixOperator(self.bd, self.gd, nblocks, self.async, True)
        overlap.bmd = BlacsBandMatrixDescriptor(self.bd, self.gd) # XXX override full-matrix descriptor
        overlap.bmd.redistribute_input = lambda A_nN: A_nN # XXX override full-matrix disassembly
        overlap.bmd.redistribute_output = lambda A_Nn: A_Nn # XXX override full-matrix assembly
        self.psit_nG = overlap.matrix_multiply(C_nN, self.psit_nG, self.P_ani)
        D_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.gd.comm.broadcast(D_Nn, 0)

        if memstats:
            self.mem_test = record_memory()

        # D_nn = C_nn^dag * S_nn * C_nn
        D0_NN = np.dot(C_NN.T.conj(), np.dot(S_NN, C_NN))
        D_NN = self.bd.collect(D_Nn.T.copy(), broadcast=True)
        tri2full(D_NN, 'U') # upper to lower...
        self.check_and_plot(D_NN, D0_NN, 9, 'multiply,randomized')

    def test_multiply_nonhermitian(self):
        alpha = np.random.normal(size=1).astype(self.dtype)
        if self.dtype == complex:
            alpha += 1j*np.random.normal(size=1)
        world.broadcast(alpha, 0)

        # Known starting point of S_nn = <psit_m|S|psit_n>
        S_NN = alpha*self.S0_nn

        if self.dtype == complex:
            C_NN = np.random.uniform(size=self.nbands**2) * \
                np.exp(1j*np.random.uniform(0,2*np.pi,size=self.nbands**2))
        else:
            C_NN = np.random.normal(size=self.nbands**2)
        C_NN = C_NN.reshape((self.nbands,self.nbands)) / np.linalg.norm(C_NN,2)
        world.broadcast(C_NN, 0)
        C_nN = np.empty((self.bd.mynbands, self.bd.nbands), dtype=C_NN.dtype)
        self.bd.distribute(C_NN.T.copy(), C_nN)

        # Set up non-Hermitian overlap operator:
        S = lambda x: alpha*x
        dS = lambda a, P_ni: np.dot(alpha*P_ni, self.setups[a].dO_ii)
        nblocks = self.get_optimal_number_of_blocks(self.blocking)
        overlap = MatrixOperator(self.bd, self.gd, nblocks, self.async, False)
        overlap.bmd = BlacsBandMatrixDescriptor(self.bd, self.gd) # XXX override full-matrix descriptor
        overlap.bmd.redistribute_input = lambda A_nN: A_nN # XXX override full-matrix disassembly
        overlap.bmd.redistribute_output = lambda A_Nn: A_Nn # XXX override full-matrix assembly
        self.psit_nG = overlap.matrix_multiply(C_nN, self.psit_nG, self.P_ani)
        D_Nn = overlap.calculate_matrix_elements(self.psit_nG, self.P_ani, S, dS)
        self.gd.comm.broadcast(D_Nn, 0)

        if memstats:
            self.mem_test = record_memory()

        # D_nn = C_nn^dag * S_nn * C_nn
        D0_NN = np.dot(C_NN.T.conj(), np.dot(S_NN, C_NN))
        D_NN = self.bd.collect(D_Nn.T.copy(), broadcast=True)
        self.check_and_plot(D_NN, D0_NN, 9, 'multiply,nonhermitian')


# -------------------------------------------------------------------

def UTConstantWavefunctionFactory(dtype, parstride_bands, blocking, async):
    sep = '_'
    classname = 'UTConstantWavefunctionBlacsSetup' \
    + sep + {float:'Float', complex:'Complex'}[dtype] \
    + sep + {False:'Blocked', True:'Strided'}[parstride_bands] \
    + sep + {'fast':'Fast', 'light':'Light', 'best':'Best'}[blocking] \
    + sep + {False:'Synchronous', True:'Asynchronous'}[async]
    class MetaPrototype(UTConstantWavefunctionBlacsSetup, object):
        __doc__ = UTConstantWavefunctionBlacsSetup.__doc__
        dtype = dtype
        parstride_bands = parstride_bands
        blocking = blocking
        async = async
    MetaPrototype.__name__ = classname
    return MetaPrototype

# -------------------------------------------------------------------

if __name__ in ['__main__', '__builtin__'] and scalapack(True):
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_hsblacs.log', verbosity=2)
    else:
        from gpaw.utilities import devnull
        stream = (world.rank == 0) and sys.stdout or devnull
        testrunner = TextTestRunner(stream=stream, verbosity=2)

    parinfo = []
    for test in [UTBandParallelBlacsSetup_Blocked]: #, UTBandParallelBlacsSetup_Strided]:
        info = ['', test.__name__, test.__doc__.strip('\n'), '']
        testsuite = initialTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        assert testresult.wasSuccessful(), 'Initial verification failed!'
        parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
                        tci in testsuite._tests if hasattr(tci, '_parinfo')])
    parinfo = np.unique(np.sort(parinfo)).tolist()

    testcases = []
    for dtype in [float, complex]:
        for parstride_bands in [False]: #XXX [False, True]:
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
            raise SystemExit('Test failed. Check ut_hsblacs.log for details.')

