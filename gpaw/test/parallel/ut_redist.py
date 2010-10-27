#!/usr/bin/env python

import sys, time
import numpy as np

from ase.units import Bohr
from gpaw import debug
from gpaw.mpi import world, distribute_cpus
from gpaw.paw import kpts2ndarray
from gpaw.parameters import InputParameters
from gpaw.xc import XC
from gpaw.brillouin import reduce_kpoints
from gpaw.setup import SetupData, Setups
from gpaw.grid_descriptor import GridDescriptor
from gpaw.band_descriptor import BandDescriptor
from gpaw.kpt_descriptor import KPointDescriptorOld
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.fd import FDWaveFunctions
from gpaw.density import Density
from gpaw.hamiltonian import Hamiltonian
from gpaw.blacs import get_kohn_sham_layouts
from gpaw.utilities.tools import md5_array
from gpaw.utilities.timing import nulltimer

# -------------------------------------------------------------------

from gpaw.test.ut_common import ase_svnversion, shapeopt, TestCase, \
    TextTestRunner, CustomTextTestRunner, defaultTestLoader, \
    initialTestLoader, create_random_atoms

# -------------------------------------------------------------------

p = InputParameters(spinpol=True)
xc = XC(p.xc)
p.setups = dict([(symbol, SetupData(symbol, xc.name)) for symbol in 'HO'])

class UTDomainParallelSetup(TestCase):
    """
    Setup a simple domain parallel calculation."""

    # Number of bands
    nbands = 12

    # Spin-polarized
    nspins = 1

    # Mean spacing and number of grid points per axis (G x G x G)
    h = 0.25 / Bohr
    G = 48

    # Type of boundary conditions employed (determines nibzkpts and dtype)
    boundaries = None
    nibzkpts = None
    dtype = None

    timer = nulltimer

    def setUp(self):
        for virtvar in ['boundaries']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        # Basic unit cell information:
        res, N_c = shapeopt(100, self.G**3, 3, 0.2)
        #N_c = 4*np.round(np.array(N_c)/4) # makes domain decomposition easier
        cell_cv = self.h * np.diag(N_c)
        pbc_c = {'zero'    : (False,False,False), \
                 'periodic': (True,True,True), \
                 'mixed'   : (True, False, True)}[self.boundaries]

        # Create randomized gas-like atomic configuration on interim grid
        tmpgd = GridDescriptor(N_c, cell_cv, pbc_c)
        self.atoms = create_random_atoms(tmpgd)

        # Create setups
        Z_a = self.atoms.get_atomic_numbers()
        assert 1 == self.nspins
        self.setups = Setups(Z_a, p.setups, p.basis, p.lmax, xc)
        self.natoms = len(self.setups)

        # Decide how many kpoints to sample from the 1st Brillouin Zone
        kpts_c = np.ceil((10/Bohr)/np.sum(cell_cv**2,axis=1)**0.5).astype(int)
        kpts_c = tuple(kpts_c * pbc_c + 1 - pbc_c)
        self.bzk_kc = kpts2ndarray(kpts_c)

        # Set up k-point descriptor
        self.kd = KPointDescriptor(self.bzk_kc, self.nspins)
        self.kd.set_symmetry(self.atoms, self.setups, p.usesymm)

        # Set the dtype
        if self.kd.gamma:
            self.dtype = float
        else:
            self.dtype = complex
            
        # Create communicators
        parsize, parsize_bands = self.get_parsizes()
        assert self.nbands % np.prod(parsize_bands) == 0
        domain_comm, kpt_comm, band_comm = distribute_cpus(parsize,
            parsize_bands, self.nspins, self.kd.nibzkpts)

        self.kd.set_communicator(kpt_comm)
        
        # Set up band descriptor:
        self.bd = BandDescriptor(self.nbands, band_comm)

        # Set up grid descriptor:
        self.gd = GridDescriptor(N_c, cell_cv, pbc_c, domain_comm, parsize)

        # Set up kpoint/spin descriptor (to be removed):
        self.kd_old = KPointDescriptorOld(self.nspins, self.kd.nibzkpts,
                                          kpt_comm, self.kd.gamma, self.dtype)

       
    def tearDown(self):
        del self.atoms, self.bd, self.gd, self.kd, self.kd_old

    def get_parsizes(self):
        # Careful, overwriting imported GPAW params may cause amnesia in Python.
        from gpaw import parsize, parsize_bands

        # D: number of domains
        # B: number of band groups
        if parsize is None:
            D = min(world.size, 2)
        else:
            D = parsize
        assert world.size % D == 0
        if parsize_bands is None:
            B = world.size // D
        else:
            B = parsize_bands
        return D, B

    # =================================

    def verify_comm_sizes(self):
        if world.size == 1:
            return
        comm_sizes = tuple([comm.size for comm in [world, self.bd.comm, \
                                                   self.gd.comm, self.kd_old.comm]])
        self._parinfo =  '%d world, %d band, %d domain, %d kpt' % comm_sizes
        self.assertEqual(self.nbands % self.bd.comm.size, 0)
        self.assertEqual((self.nspins * self.kd.nibzkpts) % self.kd_old.comm.size, 0)


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

class UTLocalizedFunctionSetup(UTDomainParallelSetup):
    __doc__ = UTDomainParallelSetup.__doc__

    distribution = None
    allocated = False

    def setUp(self):
        for virtvar in ['distribution']:
            assert getattr(self,virtvar) is not None, 'Virtual "%s"!' % virtvar

        UTDomainParallelSetup.setUp(self)

        # Initial layout
        random_a = np.random.uniform(0, self.gd.comm.size,
                                     size=self.natoms).astype(int)
        world.broadcast(random_a, 0)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        self.rank0_a = {'master'  : np.zeros(self.natoms, dtype=int),
                        'domains' : self.gd.get_ranks_from_positions(spos_ac),
                        'balanced': random_a}[self.distribution]

    def tearDown(self):
        del self.rank0_a
        UTDomainParallelSetup.tearDown(self)
        self.allocated = False

    def allocate(self, M_asp, rank_a):
        if self.allocated:
            raise RuntimeError('Already allocated!')

        my_atom_indices = np.argwhere(rank_a == self.gd.comm.rank).ravel()
        self.holm_nielsen_check(my_atom_indices)
        magmom_a = self.atoms.get_initial_magnetic_moments()
        for a in my_atom_indices:
            ni = self.setups[a].ni
            M_asp[a] = np.empty((self.nspins, ni * (ni + 1) // 2), dtype=float)
        self.chk_sa = np.zeros((self.kd_old.nspins, self.natoms), dtype=np.int64)

        self.allocated = True

    def holm_nielsen_check(self, my_atom_indices):
        if self.gd.comm.sum(np.sum(my_atom_indices, dtype=int).item()) \
            != self.natoms * (self.natoms - 1) // 2:
            raise RuntimeError('Holm-Nielsen check failed on domain %d.' \
                               % self.gd.comm.rank)

    def update_references(self, M_asp, rank_a):
        requests = []
        domain_comm = self.gd.comm
        for s in range(self.kd_old.nspins):
            for a in range(self.natoms):
                domain_rank = rank_a[a]
                if domain_comm.rank == domain_rank:
                    chk = md5_array(M_asp[a][s], numeric=True)
                    if domain_comm.rank == 0:
                        self.chk_sa[s,a] = chk
                    else:
                        requests.append(domain_comm.send(np.array([chk], \
                            dtype=np.int64), 0, 2606+a, block=False))
                elif domain_comm.rank == 0:
                    chk = self.chk_sa[s,a:a+1] #XXX hack to get pointer
                    requests.append(domain_comm.receive(chk, domain_rank, \
                        2606+a, block=False))
        domain_comm.waitall(requests)
        domain_comm.broadcast(self.chk_sa, 0)

    def check_values(self, M_asp, rank_a):
        self.holm_nielsen_check(M_asp.keys())

        # Compare to reference checksums
        ret = True
        domain_comm = self.gd.comm
        for s in range(self.kd_old.nspins):
            for a in range(self.natoms):
                domain_rank = rank_a[a]
                if domain_comm.rank == domain_rank:
                    chk = md5_array(M_asp[a][s], numeric=True)
                    ret &= (chk == self.chk_sa[s,a])
        self.assertTrue(ret)

# -------------------------------------------------------------------

class UTProjectorFunctionSetup(UTLocalizedFunctionSetup):
    __doc__ = UTDomainParallelSetup.__doc__ + """
    Projection expansion coefficients are distributed over domains."""

    def setUp(self):
        UTLocalizedFunctionSetup.setUp(self)

        fdksl = get_kohn_sham_layouts(None, 'fd', self.gd, self.bd)
        lcaoksl = get_kohn_sham_layouts(None, 'lcao', self.gd, self.bd,
                                        nao=self.setups.nao)
        args = (self.gd, self.setups.nvalence, self.setups,
                self.bd, self.dtype, world, self.kd)
        self.wfs = FDWaveFunctions(p.stencils[0], fdksl, fdksl, lcaoksl, *args)
        self.wfs.rank_a = self.rank0_a
        self.allocate(self.wfs.kpt_u, self.wfs.rank_a)
        assert self.allocated

        for kpt in self.wfs.kpt_u:
            for a,P_ni in kpt.P_ani.items():
                for myn,P_i in enumerate(P_ni):
                    n = self.bd.global_index(myn)
                    P_i[:] = 1e12 * kpt.s + 1e9 * kpt.k + 1e6 * a + 1e3 * n \
                        + np.arange(self.setups[a].ni, dtype=self.dtype)

    def tearDown(self):
        del self.wfs
        UTLocalizedFunctionSetup.tearDown(self)

    def allocate(self, kpt_u, rank_a):
        if self.allocated:
            raise RuntimeError('Already allocated!')

        my_atom_indices = np.argwhere(rank_a == self.gd.comm.rank).ravel()
        self.holm_nielsen_check(my_atom_indices)
        self.wfs.allocate_arrays_for_projections(my_atom_indices) #XXX
        self.chk_una = np.zeros((self.kd_old.nks, self.bd.nbands,
                                 self.natoms), dtype=np.int64)
        self.allocated = True

    def update_references(self, kpt_u, rank_a):
        requests = []
        kpt_comm, band_comm, domain_comm = self.kd_old.comm, self.bd.comm, self.gd.comm
        for u in range(self.kd_old.nks):
            kpt_rank, myu = self.kd_old.who_has(u)
            for n in range(self.bd.nbands):
                band_rank, myn = self.bd.who_has(n)
                for a in range(self.natoms):
                    domain_rank = rank_a[a]
                    if kpt_comm.rank == kpt_rank and \
                       band_comm.rank == band_rank and \
                       domain_comm.rank == domain_rank:
                        kpt = kpt_u[myu]
                        chk = md5_array(kpt.P_ani[a][myn], numeric=True)
                        if world.rank == 0:
                            self.chk_una[u,n,a] = chk
                        else:
                            requests.append(world.send(np.array([chk], \
                                dtype=np.int64), 0, 1303+a, block=False))
                    elif world.rank == 0:
                        world_rank = rank_a[a] + \
                            band_rank * domain_comm.size + \
                            kpt_rank * domain_comm.size * band_comm.size
                        chk = self.chk_una[u,n,a:a+1] #XXX hack to get pointer
                        requests.append(world.receive(chk, world_rank, \
                            1303+a, block=False))
        world.waitall(requests)
        world.broadcast(self.chk_una, 0)

    def check_values(self, kpt_u, rank_a):
        for kpt in kpt_u:
           self.holm_nielsen_check(kpt.P_ani.keys())

        # Compare to reference checksums
        ret = True
        kpt_comm, band_comm, domain_comm = self.kd_old.comm, self.bd.comm, self.gd.comm
        for u in range(self.kd_old.nks):
            kpt_rank, myu = self.kd_old.who_has(u)
            for n in range(self.bd.nbands):
                band_rank, myn = self.bd.who_has(n)
                for a in range(self.natoms):
                    domain_rank = rank_a[a]
                    if kpt_comm.rank == kpt_rank and \
                       band_comm.rank == band_rank and \
                       domain_comm.rank == domain_rank:
                        kpt = kpt_u[myu]
                        chk = md5_array(kpt.P_ani[a][myn], numeric=True)
                        ret &= (chk == self.chk_una[u,n,a])
        self.assertTrue(ret)

    # =================================

    def test_initial_consistency(self):
        self.update_references(self.wfs.kpt_u, self.wfs.rank_a)
        self.check_values(self.wfs.kpt_u, self.wfs.rank_a)

    def test_redistribution_to_domains(self):
        self.update_references(self.wfs.kpt_u, self.wfs.rank_a)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        self.wfs.set_positions(spos_ac)
        self.check_values(self.wfs.kpt_u, self.wfs.rank_a)

    def test_redistribution_to_same(self):
        self.update_references(self.wfs.kpt_u, self.wfs.rank_a)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        self.wfs.set_positions(spos_ac)
        self.wfs.set_positions(spos_ac)
        self.check_values(self.wfs.kpt_u, self.wfs.rank_a)

# -------------------------------------------------------------------

def UTProjectorFunctionFactory(boundaries, distribution):
    sep = '_'
    classname = 'UTProjectorFunctionSetup' \
    + sep + {'zero':'Zero', 'periodic':'Periodic', 'mixed':'Mixed'}[boundaries] \
    + sep + {'balanced':'Balanced', 'domains':'Domains', 'master':'Master'}[distribution]
    class MetaPrototype(UTProjectorFunctionSetup, object):
        __doc__ = UTProjectorFunctionSetup.__doc__
        boundaries = boundaries
        distribution = distribution
    MetaPrototype.__name__ = classname
    return MetaPrototype

# -------------------------------------------------------------------

class UTDensityFunctionSetup(UTLocalizedFunctionSetup):
    __doc__ = UTLocalizedFunctionSetup.__doc__ + """
    Atomic density matrices are distributed over domains."""

    def setUp(self):
        UTLocalizedFunctionSetup.setUp(self)

        self.finegd = self.gd.refine()
        self.density = Density(self.gd, self.finegd, self.nspins, p.charge)
        self.density.initialize(self.setups, p.stencils[1], self.timer, \
            self.atoms.get_initial_magnetic_moments(), p.hund)
        self.density.D_asp = {}
        self.density.rank_a = self.rank0_a
        self.allocate(self.density.D_asp, self.density.rank_a)
        assert self.allocated

        for a, D_sp in self.density.D_asp.items():
            ni = self.setups[a].ni
            for s, D_p in enumerate(D_sp):
                D_p[:] = 1e9 * self.kd_old.comm.rank + 1e6 * self.bd.comm.rank \
                    + 1e3 * s + np.arange(ni * (ni + 1) // 2, dtype=float)

    def tearDown(self):
        del self.density
        UTLocalizedFunctionSetup.tearDown(self)

    # =================================

    def test_initial_consistency(self):
        self.update_references(self.density.D_asp, self.density.rank_a)
        self.check_values(self.density.D_asp, self.density.rank_a)

    def test_redistribution_to_domains(self):
        self.update_references(self.density.D_asp, self.density.rank_a)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.density.set_positions(spos_ac, rank_a)
        self.check_values(self.density.D_asp, self.density.rank_a)

    def test_redistribution_to_same(self):
        self.update_references(self.density.D_asp, self.density.rank_a)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.density.set_positions(spos_ac, rank_a)
        self.density.set_positions(spos_ac, rank_a)
        self.check_values(self.density.D_asp, self.density.rank_a)

# -------------------------------------------------------------------

def UTDensityFunctionFactory(boundaries, distribution):
    sep = '_'
    classname = 'UTDensityFunctionSetup' \
    + sep + {'zero':'Zero', 'periodic':'Periodic', 'mixed':'Mixed'}[boundaries] \
    + sep + {'balanced':'Balanced', 'domains':'Domains', 'master':'Master'}[distribution]
    class MetaPrototype(UTDensityFunctionSetup, object):
        __doc__ = UTDensityFunctionSetup.__doc__
        boundaries = boundaries
        distribution = distribution
    MetaPrototype.__name__ = classname
    return MetaPrototype

# -------------------------------------------------------------------

class UTHamiltonianFunctionSetup(UTLocalizedFunctionSetup):
    __doc__ = UTLocalizedFunctionSetup.__doc__ + """
    Atomic hamiltonian matrices are distributed over domains."""

    def setUp(self):
        UTLocalizedFunctionSetup.setUp(self)

        self.finegd = self.gd.refine()
        self.hamiltonian = Hamiltonian(self.gd, self.finegd, self.nspins,
                                       self.setups, p.stencils[1], self.timer,
                                       xc, p.poissonsolver, p.external)
        self.hamiltonian.dH_asp = {}
        self.hamiltonian.rank_a = self.rank0_a
        self.allocate(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)
        assert self.allocated

        for a, dH_sp in self.hamiltonian.dH_asp.items():
            ni = self.setups[a].ni
            for s, dH_p in enumerate(dH_sp):
                dH_p[:] = 1e9 * self.kd_old.comm.rank + 1e6 * self.bd.comm.rank \
                    + 1e3 * s + np.arange(ni * (ni + 1) // 2, dtype=float)    

    def tearDown(self):
        del self.hamiltonian
        UTLocalizedFunctionSetup.tearDown(self)

    # =================================

    def test_initial_consistency(self):
        self.update_references(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)
        self.check_values(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)

    def test_redistribution_to_domains(self):
        self.update_references(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.hamiltonian.set_positions(spos_ac, rank_a)
        self.check_values(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)

    def test_redistribution_to_same(self):
        self.update_references(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        rank_a = self.gd.get_ranks_from_positions(spos_ac)
        self.hamiltonian.set_positions(spos_ac, rank_a)
        self.hamiltonian.set_positions(spos_ac, rank_a)
        self.check_values(self.hamiltonian.dH_asp, self.hamiltonian.rank_a)

# -------------------------------------------------------------------

def UTHamiltonianFunctionFactory(boundaries, distribution):
    sep = '_'
    classname = 'UTHamiltonianFunctionSetup' \
    + sep + {'zero':'Zero', 'periodic':'Periodic', 'mixed':'Mixed'}[boundaries] \
    + sep + {'balanced':'Balanced', 'domains':'Domains', 'master':'Master'}[distribution]
    class MetaPrototype(UTHamiltonianFunctionSetup, object):
        __doc__ = UTHamiltonianFunctionSetup.__doc__
        boundaries = boundaries
        distribution = distribution
    MetaPrototype.__name__ = classname
    return MetaPrototype

# -------------------------------------------------------------------

if __name__ in ['__main__', '__builtin__']:
    # We may have been imported by test.py, if so we should redirect to logfile
    if __name__ == '__builtin__':
        testrunner = CustomTextTestRunner('ut_redist.log', verbosity=2)
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
    for boundaries in ['zero', 'periodic', 'mixed']:
        for distribution in ['balanced', 'domains', 'master']:
            testcases.append(UTProjectorFunctionFactory(boundaries, distribution))
            testcases.append(UTDensityFunctionFactory(boundaries, distribution))
            testcases.append(UTHamiltonianFunctionFactory(boundaries, distribution))

    for test in testcases:
        info = ['', test.__name__, test.__doc__.strip('\n')] + parinfo + ['']
        testsuite = defaultTestLoader.loadTestsFromTestCase(test)
        map(testrunner.stream.writeln, info)
        testresult = testrunner.run(testsuite)
        # Provide feedback on failed tests if imported by test.py
        if __name__ == '__builtin__' and not testresult.wasSuccessful():
            raise SystemExit('Test failed. Check ut_redist.log for details.')
