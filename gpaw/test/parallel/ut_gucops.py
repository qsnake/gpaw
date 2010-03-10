#!/usr/bin/env python

import sys
import numpy as np

#from ase import Atoms
from ase.units import Bohr
from ase.dft.kpoints import kpoint_convert
from gpaw import debug
from gpaw.mpi import world, distribute_cpus
from gpaw.utilities import gcd
#from gpaw.utilities.tools import md5_array
from gpaw.utilities.gauss import gaussian_wave
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
#from gpaw.test.ut_common import create_parsize_maxbands
from gpaw.paw import kpts2ndarray
#from gpaw.brillouin import reduce_kpoints
from gpaw.parameters import InputParameters
from gpaw.xc_functional import XCFunctional
from gpaw.setup import Setups
from gpaw.utilities.gauss import gaussian_wave
from gpaw.fd_operators import Laplace

# -------------------------------------------------------------------

from gpaw.test.ut_common import ase_svnversion, shapeopt, TestCase, \
    TextTestRunner, CustomTextTestRunner, defaultTestLoader, \
    initialTestLoader, create_random_atoms, create_parsize_minbands

# -------------------------------------------------------------------

class UTDomainParallelSetup(TestCase):
    """
    Setup a simple domain parallel calculation."""

    # Number of bands
    nbands = 1

    # Spin-paired
    nspins = 1

    # Mean spacing and number of grid points per axis
    h = 0.2 / Bohr

    # Generic lattice constant for unit cell
    a = 5.0 / Bohr

    # Type of boundary conditions employed
    boundaries = None

    # Type of unit cell employed
    celltype = None

    def setUp(self):
        for virtvar in ['boundaries', 'celltype']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        # Basic unit cell information:
        pbc_c = {'zero'    : (False,False,False), \
                 'periodic': (True,True,True), \
                 'mixed'   : (True, False, True)}[self.boundaries]
        a, b = self.a, 2**0.5*self.a
        cell_cv = {'general'   : np.array([[0,a,a],[a/2,0,a/2],[a/2,a/2,0]]),
                   'rotated'   : np.array([[0,0,b],[b/2,0,0],[0,b/2,0]]),
                   'inverted'   : np.array([[0,0,b],[0,b/2,0],[b/2,0,0]]),
                   'orthogonal': np.diag([b, b/2, b/2])}[self.celltype]
        cell_cv = np.array([(4-3*pbc)*c_v for pbc,c_v in zip(pbc_c, cell_cv)])

        # Decide how many kpoints to sample from the 1st Brillouin Zone
        kpts_c = np.ceil((10/Bohr)/np.sum(cell_cv**2,axis=1)**0.5).astype(int)
        kpts_c = tuple(kpts_c*pbc_c + 1-pbc_c)
        bzk_kc = kpts2ndarray(kpts_c)
        self.gamma = len(bzk_kc) == 1 and not bzk_kc[0].any()

        #p = InputParameters()
        #Z_a = self.atoms.get_atomic_numbers()
        #xcfunc = XCFunctional(p.xc, self.nspins)
        #setups = Setups(Z_a, p.setups, p.basis, self.nspins, p.lmax, xcfunc)
        #symmetry, weight_k, self.ibzk_kc = reduce_kpoints(self.atoms, bzk_kc,
        #                                                  setups, p.usesymm)

        self.ibzk_kc = bzk_kc.copy() # don't use symmetry reduction of kpoints
        self.nibzkpts = len(self.ibzk_kc)
        self.ibzk_kv = kpoint_convert(cell_cv, skpts_kc=self.ibzk_kc)

        # Parse parallelization parameters and create suitable communicators.
        #parsize, parsize_bands = create_parsize_minbands(self.nbands, world.size)
        parsize, parsize_bands = world.size//gcd(world.size, self.nibzkpts), 1
        assert self.nbands % np.prod(parsize_bands) == 0
        domain_comm, kpt_comm, band_comm = distribute_cpus(parsize,
            parsize_bands, self.nspins, self.nibzkpts)

        # Set up band descriptor:
        self.bd = BandDescriptor(self.nbands, band_comm)

        # Set up grid descriptor:
        N_c = np.round(np.sum(cell_cv**2, axis=1)**0.5 / self.h)
        N_c += 4-N_c % 4 # makes domain decomposition easier
        self.gd = GridDescriptor(N_c, cell_cv, pbc_c, domain_comm, parsize)
        self.assertEqual(self.gamma, np.all(~self.gd.pbc_c))

        # What to do about kpoints?
        self.kpt_comm = kpt_comm

        if debug and world.rank == 0:
            comm_sizes = tuple([comm.size for comm in [world, self.bd.comm, \
                                                   self.gd.comm, self.kpt_comm]])
            print '%d world, %d band, %d domain, %d kpt' % comm_sizes

    def tearDown(self):
        del self.ibzk_kc, self.ibzk_kv, self.bd, self.gd, self.kpt_comm

    # =================================

    def verify_comm_sizes(self):
        if world.size == 1:
            return
        comm_sizes = tuple([comm.size for comm in [world, self.bd.comm, \
                                                   self.gd.comm, self.kpt_comm]])
        self._parinfo =  '%d world, %d band, %d domain, %d kpt' % comm_sizes
        self.assertEqual(self.nbands % self.bd.comm.size, 0)
        self.assertEqual((self.nspins*self.nibzkpts) % self.kpt_comm.size, 0)

    def verify_grid_volume(self):
        gdvol = np.prod(self.gd.get_size_of_global_array())*self.gd.dv
        self.assertAlmostEqual(self.gd.integrate(1+self.gd.zeros()), gdvol, 10)

    def verify_grid_point(self):
        # Volume integral of cartesian coordinates of all available grid points
        gdvol = np.prod(self.gd.get_size_of_global_array())*self.gd.dv
        cmr_v = self.gd.integrate(self.gd.get_grid_point_coordinates()) / gdvol

        # Theoretical center of cell based on all available grid data
        cm0_v = np.dot((0.5*(self.gd.get_size_of_global_array()-1.0) \
            + 1.0-self.gd.pbc_c) / self.gd.N_c, self.gd.cell_cv)

        self.assertAlmostEqual(np.abs(cmr_v-cm0_v).max(), 0, 10)

    def verify_non_pbc_spacing(self):
        atoms = create_random_atoms(self.gd, 1000, 'NH3', self.a/2)
        pos_ac = atoms.get_positions()
        cellvol = np.linalg.det(self.gd.cell_cv)
        if debug: print 'cell volume:', np.abs(cellvol)*Bohr**3, 'Ang^3', cellvol>0 and '(right handed)' or '(left handed)'

        # Loop over non-periodic axes and check minimum distance requirement
        for c in np.argwhere(~self.gd.pbc_c).ravel():
            a_v = self.gd.cell_cv[(c+1)%3]
            b_v = self.gd.cell_cv[(c+2)%3]
            c_v = np.cross(a_v, b_v)
            for d in range(2):
                # Inwards unit normal vector of d'th cell face of c'th axis
                # and point intersected by this plane (d=0,1 / bottom,top).
                n_v = np.sign(cellvol) * (1-2*d) * c_v / np.linalg.norm(c_v)
                if debug: print {0:'x',1:'y',2:'z'}[c]+'-'+{0:'bottom',1:'top'}[d]+':', n_v, 'Bohr'
                if debug: print 'gd.iucell_cv[%d]~' % c, self.gd.iucell_cv[c] / np.linalg.norm(self.gd.iucell_cv[c]), 'Bohr'
                origin_v = np.dot(d * np.eye(3)[c], self.gd.cell_cv)
                d_a = np.dot(pos_ac/Bohr - origin_v[np.newaxis,:], n_v)
                if debug: print 'a:', self.a/2*Bohr, 'min:', np.min(d_a)*Bohr, 'max:', np.max(d_a)*Bohr
                self.assertAlmostEqual(d_a.min(), self.a/2, 0) #XXX digits!


class UTDomainParallelSetup_GUC(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'mixed'
    celltype = 'general'

class UTDomainParallelSetup_Rot(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'mixed'
    celltype = 'rotated'

class UTDomainParallelSetup_Inv(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'mixed'
    celltype = 'inverted'

class UTDomainParallelSetup_Ortho(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__
    boundaries = 'mixed'
    celltype = 'orthogonal'

# -------------------------------------------------------------------

class UTGaussianWavefunctionSetup(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__ + """
    The pseudo wavefunctions are moving gaussians centered around each atom."""

    allocated = False
    dtype = None

    def setUp(self):
        UTDomainParallelSetup.setUp(self)

        for virtvar in ['dtype']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        # Set up kpoint descriptor:
        self.kd = KPointDescriptor(self.nspins, self.nibzkpts, self.kpt_comm, \
            self.gamma, self.dtype)

        # Choose a sufficiently small width of gaussian test functions
        self.sigma = np.min((0.1+0.4*self.gd.pbc_c)*self.gd.cell_c)

        if debug and world.rank == 0:
            print 'sigma=%8.5f Ang' % (self.sigma*Bohr), 'cell_c:', self.gd.cell_c*Bohr, 'Ang', 'N_c:', self.gd.N_c
        self.atoms = create_random_atoms(self.gd, 4, 'H', 4*self.sigma)
        self.r_vG = None
        self.wf_uG = None
        self.laplace0_uG = None

        self.allocate()

    def tearDown(self):
        UTDomainParallelSetup.tearDown(self)
        del self.phase_ucd, self.atoms, self.r_vG, self.wf_uG, self.laplace0_uG
        self.allocated = False

    def allocate(self):
        if self.allocated:
            raise RuntimeError('Already allocated!')

        # Calculate complex phase factors:
        self.phase_ucd = np.ones((self.kd.mynks, 3, 2), complex)
        if not self.gamma:
            for myu, phase_cd in enumerate(self.phase_ucd):
                u = self.kd.global_index(myu)
                s, k = self.kd.what_is(u)
                phase_cd[:] = np.exp(2j * np.pi * self.gd.sdisp_cd * \
                                     self.ibzk_kc[k,:,np.newaxis])
            assert self.dtype == complex, 'Complex wavefunctions are required.'

        self.r_vG = self.gd.get_grid_point_coordinates()
        self.wf_uG = self.gd.zeros(self.kd.mynks, dtype=self.dtype)
        self.laplace0_uG = self.gd.zeros(self.kd.mynks, dtype=self.dtype)
        buf_G = self.gd.empty(dtype=self.dtype)

        sdisp_Ac = []
        for a,spos_c in enumerate(self.atoms.get_scaled_positions() % 1.0):
            for sdisp_x in range(-1*self.gd.pbc_c[0],self.gd.pbc_c[0]+1):
                for sdisp_y in range(-1*self.gd.pbc_c[1],self.gd.pbc_c[1]+1):
                    for sdisp_z in range(-1*self.gd.pbc_c[2],self.gd.pbc_c[2]+1):
                        sdisp_c = np.array([sdisp_x, sdisp_y, sdisp_z])
                        if debug and world.rank == 0:
                            print 'a=%d, spos=%s, sdisp_c=%s' % (a,spos_c,sdisp_c)
                        sdisp_Ac.append((a,spos_c,sdisp_c))

        for a,spos_c,sdisp_c in sdisp_Ac:
            if debug and world.rank == 0:
                print 'Adding gaussian at a=%d, spos=%s, sigma=%8.5f Ang' % (a,spos_c+sdisp_c,self.sigma*Bohr)

            r0_v = np.dot(spos_c+sdisp_c, self.gd.cell_cv)

            for myu in range(self.kd.mynks):
                u = self.kd.global_index(myu)
                s, k = self.kd.what_is(u)
                ibzk_v = self.ibzk_kv[k]

                # f(r) = sum_a A exp(-|r-R^a|^2 / 2sigma^2) exp(i k.r)
                gaussian_wave(self.r_vG, r0_v, self.sigma, ibzk_v, A=1.0,
                              dtype=self.dtype, out_G=buf_G)
                self.wf_uG[myu] += buf_G

                # d^2/dx^2 exp(ikx-(x-x0)^2/2sigma^2)
                # ((ik-(x-x0)/sigma^2)^2 - 1/sigma^2) exp(ikx-(x-x0)^2/2sigma^2)
                dr2_G = np.sum((1j*ibzk_v[:,np.newaxis,np.newaxis,np.newaxis] \
                    - (self.r_vG-r0_v[:,np.newaxis,np.newaxis,np.newaxis]) \
                    / self.sigma**2)**2, axis=0)
                self.laplace0_uG[myu] += (dr2_G - 3/self.sigma**2) * buf_G

        self.allocated = True

    # =================================

    def test_something(self):
        laplace_uG = np.empty_like(self.laplace0_uG)
        op = Laplace(self.gd, dtype=self.dtype)
        for myu, laplace_G in enumerate(laplace_uG):
            phase_cd = {float:None, complex:self.phase_ucd[myu]}[self.dtype]
            op.apply(self.wf_uG[myu], laplace_G, phase_cd)
            print 'myu:', myu, 'diff:', np.std(laplace_G-self.laplace0_uG[myu]), '/', np.abs(laplace_G-self.laplace0_uG[myu]).max()

# -------------------------------------------------------------------

def UTGaussianWavefunctionFactory(boundaries, celltype, dtype):
    sep = '_'
    classname = 'UTGaussianWavefunctionSetup' \
    + sep + {'zero':'Zero', 'periodic':'Periodic', 'mixed':'Mixed'}[boundaries] \
    + sep + {'general':'GUC', 'rotated':'Rot', 'inverted':'Inv',
             'orthogonal':'Ortho'}[celltype] \
    + sep + {float:'Float', complex:'Complex'}[dtype]
    class MetaPrototype(UTGaussianWavefunctionSetup, object):
        __doc__ = UTGaussianWavefunctionSetup.__doc__
        boundaries = boundaries
        celltype = celltype
        dtype = dtype
    MetaPrototype.__name__ = classname
    return MetaPrototype

# -------------------------------------------------------------------

if __name__ in ['__main__', '__builtin__']:
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_gucops.log', verbosity=2)
    else:
        from gpaw.utilities import devnull
        stream = (world.rank == 0) and sys.stdout or devnull
        testrunner = TextTestRunner(stream=stream, verbosity=2)

    parinfo = []
    for test in [UTDomainParallelSetup_GUC, UTDomainParallelSetup_Rot, \
                 UTDomainParallelSetup_Inv, UTDomainParallelSetup_Ortho]:
        info = ['', test.__name__, test.__doc__.strip('\n'), '']
        testsuite = initialTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        assert testresult.wasSuccessful(), 'Initial verification failed!'
        parinfo.extend(['    Parallelization options: %s' % tci._parinfo for \
                        tci in testsuite._tests if hasattr(tci, '_parinfo')])
    parinfo = np.unique(np.sort(parinfo)).tolist()

    testcases = []
    for boundaries in ['zero', 'periodic', 'mixed']:
        for celltype in ['general', 'rotated', 'inverted', 'orthogonal']:
            for dtype in (boundaries=='zero' and [float, complex] or [complex]):
                testcases.append(UTGaussianWavefunctionFactory(boundaries, \
                    celltype, dtype))

    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_gucops.log for details.')

