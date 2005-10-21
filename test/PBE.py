from gridpaw.xc_functional import XCFunctional
from gridpaw.utilities import equal


xc = XCFunctional('PBE')
rs = 2.0
zeta = -0.6
a2 = 0.01

gga = True
spinpol = True

d = 0.000001

ex, dexdrs, dexda2 = xc.exchange(rs, a2)
dexdrsn = (xc.exchange(rs + d, a2)[0] - ex) / d
dexda2n = (xc.exchange(rs, a2 + d)[0] - ex) / d

ec, decdrs, decdzeta, decda2 = xc.correlation(rs, zeta, a2)
decdrsn = (xc.correlation(rs + d, zeta, a2)[0] - ec) / d
decdzetan = (xc.correlation(rs, zeta + d, a2)[0] - ec) / d
decda2n = (xc.correlation(rs, zeta, a2 + d)[0] - ec) / d

equal(dexdrs, dexdrsn, 5e-8)
equal(dexda2, dexda2n, 3e-4)
equal(decdrs, decdrsn, 6e-8)
equal(decdzeta, decdzetan, 5e-9)
equal(decda2, decda2n, 7e-5)
