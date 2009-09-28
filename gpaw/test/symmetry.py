import numpy as np
from gpaw.symmetry import Symmetry
from ase.dft.kpoints import monkhorst_pack

# Primitive diamond lattice, with Si lattice parameter
a = 5.475
cell_cv = .5 * a * np.array([(1, 1, 0), (1, 0, 1), (0, 1, 1)])
spos_ac = np.array([(.00, .00, .00),
                    (.25, .25, .25)])
id_a = [1, 1] # Two identical atoms
pbc_c = np.ones(3, bool)
bzk_kc = monkhorst_pack((4, 4, 4))

# Do check
symm = Symmetry(id_a, cell_cv, pbc_c)
symm.analyze(spos_ac)
ibzk_kc, w_k = symm.reduce(bzk_kc)
assert len(symm.operations) == 24
assert len(symm.symmetries) == 6
assert len(w_k) == 10
a = 3 / 32.; b = 1 / 32.; c = 6 / 32.
assert np.all(w_k == [a, b, a, c, c, a, a, a, a, b])


# Linear chain of four atoms, with H lattice parameter
cell_cv = np.diag((8., 5., 5.))
spos_ac = np.array([[ 0.125,  0.5  ,  0.5  ],
                    [ 0.375,  0.5  ,  0.5  ],
                    [ 0.625,  0.5  ,  0.5  ],
                    [ 0.875,  0.5  ,  0.5  ]])
id_a = [1, 1, 1, 1] # Four identical atoms
pbc_c = np.array([1, 0, 0], bool)
bzk_kc = monkhorst_pack((3, 1, 1))

# Do check
symm = Symmetry(id_a, cell_cv, pbc_c)
symm.analyze(spos_ac)
ibzk_kc, w_k = symm.reduce(bzk_kc)
assert len(symm.operations) == 2
assert len(symm.symmetries) == 2
assert len(w_k) == 2
assert np.all(w_k == [1 / 3., 2 / 3.])
