#!/usr/bin/env python

import sys
import numpy as np

from ase.units import Bohr
from gpaw.mpi import world, distribute_cpus
from gpaw.utilities import gcd
from gpaw.utilities.tools import md5_array
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.parameters import InputParameters
from gpaw.xc_functional import XCFunctional
from gpaw.setup import Setups
from gpaw.lfc import LFC
from gpaw.wavefunctions import EmptyWaveFunctions
from gpaw.utilities.timing import NullTimer
#from gpaw.kpoint import KPoint
from gpaw.pair_overlap import PairOverlap

# -------------------------------------------------------------------

from ut_hsops import mpl, ase_svnrevision, TestCase, CustomTextTestRunner, \
                     TextTestRunner, defaultTestLoader, initialTestLoader, \
                     create_random_atoms

# Hack to use a feature from ASE 3.1.0 svn. rev. 1001 or later.
if ase_svnrevision >= 1001: # wasn't bug-free between rev. 893 and 1000
    from ase.utils.memory import shapeopt
else:
    # Bogus function only valid for one set of parameters.
    def shapeopt(maxseed, size, ndims, ecc): 
        assert (maxseed,size,ndims,ecc) == (300, 90**3, 3, 0.2)
        return -np.inf, (75.0, 108.0, 90.0)

# -------------------------------------------------------------------

#XXX DEBUG
if sys.argv[-1] == '--wait':
    import os, time
    for i in range(10):
        print 'PID: %6d, Waited %02d seconds...' % (os.getpid(),i)
        time.sleep(1)
#XXX DEBUG



class UTDomainParallelSetup(TestCase):
    """
    Setup a simple domain parallel calculation."""

    # Number of bands
    nbands = 1

    # Spin-paired, single kpoint
    nspins = 1
    nibzkpts = 1

    # Mean spacing and number of grid points per axis (G x G x G)
    h = 0.1 / Bohr
    G = 90

    # Type of boundary conditions employed
    boundaries = None

    def setUp(self):
        for virtvar in ['boundaries']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        parsize, parsize_bands = self.get_parsizes()
        assert self.nbands % np.prod(parsize_bands) == 0
        domain_comm, kpt_comm, band_comm = distribute_cpus(parsize,
            parsize_bands, self.nspins, self.nibzkpts)

        # Set up band descriptor:
        self.bd = BandDescriptor(self.nbands, band_comm)

        # Set up grid descriptor:
        res, ngpts = shapeopt(300, self.G**3, 3, 0.2)
        cell_c = self.h * np.array(ngpts)
        pbc_c = self.get_periodic_scenario(self.boundaries)
        self.gd = GridDescriptor(ngpts, cell_c, pbc_c, domain_comm, parsize)

        # What to do about kpoints?
        self.kpt_comm = kpt_comm

    def tearDown(self):
        del self.bd, self.gd, self.kpt_comm

    def get_parsizes(self):
        # Careful, overwriting imported GPAW params may cause amnesia in Python.
        from gpaw import parsize, parsize_bands

        # Just pass domain parsize through (None is tolerated)
        test_parsize = parsize

        # If parsize_bands is not set, choose the largest possible
        test_parsize_bands =  parsize_bands or gcd(self.nbands, world.size)

        return test_parsize, test_parsize_bands

    def get_periodic_scenario(self, boundaries='zero'):
        """XXX TODO

        ``boundaries``  ``pbc_c``        Description
        ==============  =============    ====================================
        'zero'          ``False``        Zero boundary conditions on all axes.
        'periodic'      ``True``         All boundary conditions are periodic.
        'mixed'         ``...``          XXX
        """

        if boundaries == 'zero':
            return False
        elif boundaries == 'periodic':
            return True
        elif boundaries == 'mixed':
            return (True, False, True)
        else:
            raise ValueError('Unrecognized boundary mode "%s".' % boundaries)

    # =================================

    def verify_comm_sizes(self):
        if world.size == 1:
            return
        comm_sizes = tuple([comm.size for comm in [world, self.bd.comm, \
                                                   self.gd.comm, self.kpt_comm]])
        self._parinfo =  '%d world, %d band, %d domain, %d kpt' % comm_sizes
        self.assertEqual(self.nbands % self.bd.comm.size, 0)
        self.assertEqual((self.nspins*self.nibzkpts) % self.kpt_comm.size, 0)


class UTDomainParallelSetup_Zero(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'zero'

class UTDomainParallelSetup_Periodic(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'periodic'

class UTDomainParallelSetup_Mixed(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'mixed'

# -------------------------------------------------------------------

# Helper functions here

# -------------------------------------------------------------------


class UTGaussianWavefunctionSetup(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__ + """
    The pseudo wavefunctions are moving gaussians centered around each atom."""

    allocated = False
    dtype = None

    boundaries = 'zero' #XXX TEMPORARY

    # Helper functions for memory efficient reshaping to match any grid
    _all = slice(None)
    _newgd = (np.newaxis, np.newaxis, np.newaxis,)
    def _slice(sclass, c=-1):
        newgd = list(sclass._newgd)
        if c == -1:
            return [sclass._all]+newgd
        elif c in range(3):
            newgd[c] = sclass._all
            return newgd
        else:
            raise ValueError('Invalid slicing index "%s".' % c)
    _slice = classmethod(_slice)

    # Generate density components from atom-centered Gaussians
    # 4*pi*int(exp(-(r-r0)**2/(2*sigma**2)),r=0...infinity)
    # = sigma**3*(2*pi)**(3/2.) = 1/norm
    _sigma0 = 2.0 #0.75
    _gauss = lambda sclass, r_cG, r0_c, dr: 1/(dr*(2*np.pi)**0.5)**3 \
        * np.exp(-np.sum((r_cG-r0_c[sclass._slice(-1)])**2, axis=0)/(2*dr**2))
    _gauss = classmethod(_gauss)

    def setUp(self):
        UTDomainParallelSetup.setUp(self)

        for virtvar in ['dtype']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        # Create randomized atoms
        self.atoms = create_random_atoms(self.gd, 1, 'NH3') #NH3/BDA

        # XXX DEBUG START
        if False:
            from ase import view
            view(self.atoms*(1+self.gd.pbc_c))
        # XXX DEBUG END

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

        # Also create pseudo partial waveves
        self.phit = LFC(self.gd, [setup.phit_j for setup in self.setups], \
                        self.kpt_comm, dtype=self.dtype)
        self.phit.set_positions(spos_ac)

        self.r_cG = None
        self.psit_nG = None

        self.allocate()

    def tearDown(self):
        UTDomainParallelSetup.tearDown(self)
        del self.r_cG
        del self.pt, self.setups, self.atoms
        self.allocated = False

    def allocate(self):
        self.r_cG = self.gd.empty((3,), dtype=self.dtype)
        for c, r_G in enumerate(self.r_cG):
            slice_c2G = list(self._newgd)
            slice_c2G[c] = slice(None) #this means ':'
            assert slice_c2G == self._slice(c)
            r_G[:] = self.gd.h_c[c]*np.arange(self.gd.beg_c[c], \
                                              self.gd.end_c[c])[slice_c2G]

        slice_c2cG = [slice(None)]+list(self._newgd)
        assert slice_c2cG == self._slice(-1)

        cell_cv = self.atoms.get_cell() / Bohr
        assert np.abs(cell_cv-self.gd.cell_cv).max() < 1e-9
        center_c = 0.5*cell_cv.diagonal()
        psit_G = 10*self.get_gaussian(self.r_cG, center_c) \
            * np.sin(2*np.pi*(self.r_cG[0]/5.0 + self.r_cG[1]/3.0))
        for pos_c in self.atoms.get_positions() / Bohr:
            sigma = self._sigma0/(1+np.sum(pos_c**2))**0.5
            psit_G += 5 * self.get_gaussian(self.r_cG, pos_c, sigma) \
                * np.cos(2*np.pi*(self.r_cG[0]/2.0-self.r_cG[1]/7.0))
        assert self.bd.mynbands == 1
        self.psit_nG = psit_G.reshape((1,)+psit_G.shape)

        #work_G = self.gd.empty(dtype=dtype)
        #restored_G = np.empty_like(work_G)

        self.allocated = True

    def get_gaussian(self, r_cG, center_c, sigma=None):
        if sigma is None:
            sigma = self._sigma0
        return self._gauss(r_cG, center_c, sigma).astype(self.dtype)

    def collect_projections(self, P_ani):
        #XXX copy/paste from WaveFunctions.collect_projections

        assert self.kpt_comm.size == 1
        kpt_rank = 0
        natoms = len(self.atoms)
        nproj = sum([setup.ni for setup in self.setups])
        P_In = np.empty((nproj, self.nbands), self.dtype)

        if world.rank == 0:
            mynu = self.nspins * self.nibzkpts // self.kpt_comm.size
            for band_rank in range(self.bd.comm.size):
                nslice = self.bd.get_slice(band_rank)
                i = 0
                for a in range(natoms):
                    ni = self.setups[a].ni
                    if kpt_rank == 0 and band_rank == 0 and a in P_ani:
                        P_ni = P_ani[a]
                    else:
                        P_ni = np.empty((self.bd.mynbands, ni), self.dtype)
                        world_rank = (self.rank_a[a] +
                                      kpt_rank * self.gd.comm.size *
                                      self.bd.comm.size +
                                      band_rank * self.gd.comm.size)
                        world.receive(P_ni, world_rank, 1303 + a)
                    P_In[i:i + ni, nslice] = P_ni.T
                    i += ni
                assert i == nproj
        elif self.kpt_comm.rank == kpt_rank: # plain else works too...
            for a in range(natoms):
                if a in P_ani:
                    world.send(P_ani[a], 0, 1303 + a)

        world.broadcast(P_In, 0)
        return P_In

    def check_and_plot(self, P_ani, P0_ani, digits, keywords='none'):

        if self.gd.comm.size == 1 and self.bd.comm.size == 1:
            # Collapse into viewable matrices
            P_In = np.concatenate([P_ni.T for P_ni in P_ani.values()])
            P0_In = np.concatenate([P0_ni.T for P0_ni in P0_ani.values()])
        else:
            P_In = self.collect_projections(P_ani)
            P0_In = self.collect_projections(P0_ani)


        # Construct fingerprint of input matrices for comparison
        fingerprint = np.array([md5_array(P_In, numeric=True),
                                md5_array(P0_In, numeric=True)])

        # Compare fingerprints across all processors
        fingerprints = np.empty((world.size, 2), np.int64)
        world.all_gather(fingerprint, fingerprints)
        if fingerprints.ptp(0).any():
            raise RuntimeError('Distributed matrices are not identical!')

        # If assertion fails, catch temporarily while plotting, then re-raise
        try:
            self.assertAlmostEqual(np.abs(P_In-P0_In).max(), 0, digits)
        except AssertionError:
            if world.rank == 0 and mpl is not None:
                from matplotlib.figure import Figure
                fig = Figure()
                ax = fig.add_axes([0.0, 0.1, 1.0, 0.83])
                ax.set_title(self.__class__.__name__)
                im = ax.imshow(np.abs(P_In-P0_In), interpolation='nearest')
                fig.colorbar(im)
                fig.legend((im,), (keywords,), 'lower center')

                from matplotlib.backends.backend_agg import FigureCanvasAgg
                img = 'ut_invops_%s_%s.png' % (self.__class__.__name__, \
                    '_'.join(keywords.split(',')))
                FigureCanvasAgg(fig).print_figure(img.lower(), dpi=90)
            raise


    # =================================

    def test_projection_linearity(self):

        Q_ani = self.pt.dict(self.bd.mynbands)
        self.pt.integrate(self.psit_nG, Q_ani, q=-1) #XXX q???

        P0_ani = dict([(a,Q_ni.copy()) for a,Q_ni in Q_ani.items()])
        self.pt.add(self.psit_nG, Q_ani, q=-1) #XXX q???
        self.pt.integrate(self.psit_nG, P0_ani, q=-1) #XXX q???

        #spos_ac = self.atoms.get_scaled_positions() % 1.0
        #rank_a = self.gd.get_ranks_from_positions(spos_ac)
        #my_atom_indices = np.argwhere(self.gd.comm.rank == rank_a).ravel()

        #                                              ~ a   ~ a'
        #TODO XXX must fix PairOverlap-ish stuff for < p  | phi  > overlaps
        #                                               i      i'

        spos_ac = self.pt.spos_ac
        # spos_ac = self.atoms.get_scaled_positions() % 1.0
        #my_atom_indices = np.argwhere(self.rank_a == self.gd.comm.rank).ravel()

        po = PairOverlap(self.gd, self.setups)
        dB_aa = po.calculate_overlaps(spos_ac, self.pt)
        #dX_aa = po.calculate_overlaps(spos_ac, self.pt, self.phit)
        #P_ani = self.pt.dict(self.bd.mynbands, zero=True)
        P_ani = dict([(a,Q_ni.copy()) for a,Q_ni in Q_ani.items()])
        for a1 in range(len(self.atoms)):
            if a1 in P_ani.keys():
                P_ni = P_ani[a1]
            else:
                # Atom a1 is not in domain so allocate a temporary buffer
                P_ni = np.zeros((self.bd.mynbands,self.setups[a1].ni,), dtype=self.pt.dtype) #TODO
            for a2, Q_ni in Q_ani.items():
                # dX_aa are the overlap extrapolators across atomic pairs
                dB_ii = po.extract_atomic_pair_matrix(dB_aa, a1, a2)
                P_ni += np.dot(Q_ni, dB_ii.T) #sum over a2 and last i in dB_ii
            self.gd.comm.sum(P_ni)

        self.check_and_plot(P_ani, P0_ani, 6, 'projection,linearity')

    def dont_test_pair_overlap_creation(self): #XXX
        # WaveFunctions
        wfs = EmptyWaveFunctions()
        wfs.gd = self.gd
        #wfs.nspins = self.nspins
        wfs.bd = self.bd
        wfs.nbands = self.bd.nbands
        wfs.mynbands = self.bd.mynbands
        #wfs.dtype = self.dtype
        #wfs.world = world
        wfs.kpt_comm = self.kpt_comm
        wfs.band_comm = self.bd.comm
        #wfs.gamma = ?
        #wfs.bzk_kc = ?
        #wfs.ibzk_kc = ?
        #wfs.weight_k = ?
        #wfs.symmetry = ?
        wfs.timer = NullTimer()
        #wfs.rank_a = ?
        #wfs.nibzkpts = ?
        #wfs.kpt_u = ?
        #wfs.ibzk_qc = ?
        #wfs.eigensolver = ?
        #wfs.positions_set = ?

        # GridWaveFunctions
        #wfs.kin = ?
        #wfs.orthonormalized = ?
        wfs.setups = self.setups
        wfs.pt = self.pt

        overlap = PairOverlap(wfs, self.atoms)

# -------------------------------------------------------------------

def UTGaussianWavefunctionSetupFactory(boundaries, dtype):
    sep = '_'
    classname = 'UTGaussianWavefunctionSetup' \
    + sep + {'zero':'Zero', 'periodic':'Periodic', 'mixed':'Mixed'}[boundaries] \
    + sep + {float:'Float', complex:'Complex'}[dtype]
    class MetaPrototype(UTGaussianWavefunctionSetup, object):
        __doc__ = UTGaussianWavefunctionSetup.__doc__
        boundaries = boundaries
        dtype = dtype
    MetaPrototype.__name__ = classname
    return MetaPrototype


# -------------------------------------------------------------------

if False: #XXX DEBUG DIRECT
    class UTC(UTGaussianWavefunctionSetup):
        boundaries='periodic'
        dtype=float

    testcase = UTC('test_projection_linearity')
    testcase.setUp()
    testcase.test_projection_linearity()
    testcase.tearDown()

elif __name__ in ['__main__', '__builtin__']:
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_invops.log', verbosity=2)
    else:
        from gpaw.utilities import devnull
        stream = (world.rank == 0) and sys.stdout or devnull
        testrunner = TextTestRunner(stream=stream, verbosity=2)

    parinfo = []
    for test in [UTDomainParallelSetup_Zero, UTDomainParallelSetup_Periodic, \
                 UTDomainParallelSetup_Mixed]:
        info = ['', test.__name__, test.__doc__.strip('\n'), '']
        testsuite = initialTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        assert testresult.wasSuccessful(), 'Initial verification failed!'
        parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
                        tci in testsuite._tests if hasattr(tci, '_parinfo')])
    parinfo = np.unique(np.sort(parinfo)).tolist()

    testcases = []
    for boundaries in ['periodic']: #['zero', 'periodic', 'mixed']:
         for dtype in [float]: #[float, complex]:
             testcases.append(UTGaussianWavefunctionSetupFactory(boundaries, \
                 dtype))

    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_invops.log for details.')

