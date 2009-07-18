"""Common code base for maintaining backwards compatibility in ut_xxx tests."""

__all__ = ['ase_svnrevision', 'shapeopt', 'TestCase', 'TextTestRunner', \
    'CustomTextTestRunner', 'defaultTestLoader', 'initialTestLoader', \
    'create_random_atoms', 'create_parsize_maxbands']

partest = False

# -------------------------------------------------------------------

# Maintain backwards compatibility with ASE 3.1.0 svn. rev. 846 or later
try:
    from ase.svnrevision import svnrevision as ase_svnrevision
except ImportError:
    # Fall back on minimum required ASE svn.rev.
    ase_svnrevision = 846
else:
    # From test/ase3k_version.py.
    full_ase_svnrevision = ase_svnrevision
    if ase_svnrevision[-1] == 'M':
        ase_svnrevision = ase_svnrevision[:-1]
    if ase_svnrevision.rfind(':') != -1:
        ase_svnrevision = ase_svnrevision[:ase_svnrevision.rfind(':')]
    ase_svnrevision = int(ase_svnrevision)

# Hack to use a feature from ASE 3.1.0 svn. rev. 1001 or later.
if ase_svnrevision >= 1001: # wasn't bug-free between rev. 893 and 1000
    from ase.utils.memory import shapeopt
else:
    shapeopt = None

if partest:
    from gpaw.testing.parunittest import ParallelTestCase as TestCase, \
        ParallelTextTestRunner as TextTestRunner, ParallelTextTestRunner as \
        CustomTextTestRunner, defaultParallelTestLoader as defaultTestLoader
    def CustomTextTestRunner(logname, verbosity=1):
        return TextTestRunner(stream=logname, verbosity=verbosity)
elif ase_svnrevision >= 929:
    from ase.test import CustomTestCase as TestCase, CustomTextTestRunner
    from unittest import TextTestRunner, defaultTestLoader
else:
    # Hack to use features from ASE 3.1.0 svn. rev. 929 or later.
    import sys
    from ase.parallel import paropen
    from unittest import TextTestRunner, defaultTestLoader, TestCase as _UTC

    if sys.version_info < (2, 4, 0, 'final', 0):
        class TestCase(_UTC): 
            assertTrue = _UTC.failUnless 
            assertFalse = _UTC.failIf 
    else: 
        TestCase = _UTC

    class CustomTextTestRunner(TextTestRunner): 
        def __init__(self, logname, descriptions=1, verbosity=1): 
            self.f = paropen(logname, 'w') 
            TextTestRunner.__init__(self, self.f, descriptions, verbosity) 
 
        def run(self, test): 
            stderr_old = sys.stderr 
            try: 
                sys.stderr = self.f 
                testresult = TextTestRunner.run(self, test) 
            finally: 
                sys.stderr = stderr_old 
            return testresult 


from copy import copy
initialTestLoader = copy(defaultTestLoader)
assert hasattr(initialTestLoader, 'testMethodPrefix')
initialTestLoader.testMethodPrefix = 'verify'

# -------------------------------------------------------------------

import numpy as np

from math import sin, cos
from ase import Atoms, molecule
from ase.units import Bohr
from gpaw.mpi import compare_atoms
from gpaw.utilities.tools import md5_array

def create_random_atoms(gd, nmolecules=10, name='H2O', mindist=4.5):
    """Create gas-like collection of atoms from randomly placed molecules.
    Applies rigid motions to molecules, translating the COM and/or rotating
    by a given angle around an axis of rotation through the new COM.

    Warning: This is only intended for testing parallel grid/LFC consistency.
    """
    assert not gd.is_non_orthogonal(), 'Orthogonal grid required.'
    cell_c = gd.cell_cv.diagonal() * Bohr
    atoms = Atoms(cell=cell_c, pbc=gd.pbc_c)

    # Store the original state of the random number generator
    randstate = np.random.get_state()
    np.random.seed(np.array([md5_array(data, numeric=True) for data
        in [nmolecules, gd.cell_cv, gd.pbc_c, gd.N_c]]).astype(int))

    for m in range(nmolecules):
        amol = molecule(name)

        # Rotate the molecule around COM according to three random angles
        # The rotation axis is given by spherical angles phi and theta
        v,phi,theta = np.random.uniform(0.0, 2*np.pi, 3) # theta [0,pi[ really
        axis = np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
        amol.rotate(axis, v)

        # Dimensions of the smallest possible box centered on the COM
        dpos_ac = amol.get_positions()-amol.get_center_of_mass()[np.newaxis,:]
        combox_c = np.abs(dpos_ac).max(axis=0)
        delta_c = (1-np.array(gd.pbc_c)) * (combox_c + mindist)
        assert (delta_c < cell_c-delta_c).all(), 'Box is too tight to fit atoms.'
        center_c = [np.random.uniform(d,w-d) for d,w in zip(delta_c, cell_c)]

        # Translate the molecule such that COM is located at random center
        offset_ac = (center_c-amol.get_center_of_mass())[np.newaxis,:]
        amol.set_positions(amol.get_positions()+offset_ac)
        assert np.linalg.norm(amol.get_center_of_mass()-center_c) < 1e-12
        atoms.extend(amol)

    # Restore the original state of the random number generator
    np.random.set_state(randstate)
    assert compare_atoms(atoms)
    return atoms


# -------------------------------------------------------------------

from gpaw.utilities import gcd
from gpaw import parsize, parsize_bands

def create_parsize_maxbands(nbands, world_size):
    """Safely parse command line parallel arguments for band parallel case."""
    # D: number of domains
    # B: number of band groups   
    if parsize_bands is None:
        if parsize is None:
            B = gcd(nbands, world_size) # largest possible
            D = world_size // B
        else:
            D = parsize
            B = world_size // np.prod(D)
    else:
        B = parsize_bands
        D = parsize or world_size // B
    return D, B

