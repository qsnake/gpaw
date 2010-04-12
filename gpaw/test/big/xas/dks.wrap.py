from gpaw import setup_paths
setup_paths.insert(0, '.')
from gpaw.test import equal

execfile('../../../../doc/tutorials/xas/dks.py')

equal(532.774045723, e2 - e1, 1e-5)
