import Numeric as num
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import equal


for xc in [XCFunctional('PBE'),
           XCFunctional('LDA'),
           XCFunctional('XC-2-1.0')]:
    naa = 0.1 * num.ones(1, num.Float)
    nb = 0.12 * num.ones(1, num.Float)
    e = num.zeros(1, num.Float)
    va = num.zeros(1, num.Float)
    vb = num.zeros(1, num.Float)
    a2 = 1.2 * num.ones(1, num.Float)
    aa2 = 0.2 * num.ones(1, num.Float)
    ab2 = 0.4 * num.ones(1, num.Float)
    deda2 = num.zeros(1, num.Float)
    dedaa2 = num.zeros(1, num.Float)
    dedab2 = num.zeros(1, num.Float)
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
    
