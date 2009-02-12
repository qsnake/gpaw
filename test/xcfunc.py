import numpy as np
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import equal

nspins = 2
for xc in [XCFunctional('PBE', nspins),
           XCFunctional('LDA', nspins),
           XCFunctional('XC-2-1.0', nspins)]:
    naa = 0.1 * np.ones(1)
    nb = 0.12 * np.ones(1)
    e = np.zeros(1)
    va = np.zeros(1)
    vb = np.zeros(1)
    a2 = 1.2 * np.ones(1)
    aa2 = 0.2 * np.ones(1)
    ab2 = 0.4 * np.ones(1)
    deda2 = np.zeros(1)
    dedaa2 = np.zeros(1)
    dedab2 = np.zeros(1)
    xc.calculate_spinpolarized(e, naa, va, nb, vb,
                              a2, aa2, ab2,
                              deda2, dedaa2, dedab2)
    E0 = e[0]
    x = 0.000001
    dna, dnb, da2, daa2, dab2 = 2.4, -1.3, 5.7, 2.0, -1.7
    #dna, dnb, da2, daa2, dab2 = 0, 0, 0, 0, 2.9
    dE = (va[0] * dna + vb[0] * dnb +
          deda2[0] * da2 + dedaa2[0] * daa2 * 2 + dedab2[0] * dab2 * 2)
    naa += x * dna
    nb += x * dnb
    a2 += x * da2
    aa2 += x * daa2
    ab2 += x * dab2
    xc.calculate_spinpolarized(e, naa, va, nb, vb,
                              a2, aa2, ab2,
                              deda2, dedaa2, dedab2)
    equal((e[0] - E0) / x, dE, 6)
    
