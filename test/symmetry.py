import numpy as np
from gpaw.symmetry import Symmetry
from ase.dft.kpoints import monkhorst_pack

# Diamond lattice, with Si lattice parameter
a = 5.475
cell_cv = .5 * a * np.array([(1, 1, 0),
                             (1, 0, 1),
                             (0, 1, 1)])
spos_ac = np.array([(.00, .00, .00),
                    (.25, .25, .25)])
id_a = [1, 1] # Two idetical atoms
pbc_c = np.ones(3)

symm = Symmetry(id_a, cell_cv, pbc_c,)

symm.analyze(spos_ac)

assert len(symm.operations) == 24

bzk_kc = monkhorst_pack((4, 4, 4))
ibzk_kc, w_k = symm.reduce(bzk_kc)
assert len(symm.symmetries) == 6
assert len(w_k) == 10

a = 3 / 32.
b = 1 / 32.
c = 6 / 32.
assert np.all(w_k == [a, b, a, c, c, a, a, a, a, b])
