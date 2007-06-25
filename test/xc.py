# This test takes approximately 0.0 seconds
from gpaw.xc_functional import XCFunctional

for name in ['PBE', 'LDA', 'RPBE', 'revPBE', 'LDAc', 'LDAx', 'RPBEx', 'revPBEx',
             'PW91']:
    xc = XCFunctional(name)
    x = xc.exchange
    c = xc.correlation
    c0 = xc.xc.correlation0
    rs = 3.0
    zeta = -0.6
    a2 = 0.05

    d = 0.000001

    ex, dexdrs, dexda2 = x(rs, a2)
    dexdrsn = (x(rs + d, a2)[0] - x(rs - d, a2)[0]) / d / 2
    dexda2n = (x(rs, a2 + d)[0] - x(rs, a2 - d)[0]) / d / 2

    ec, decdrs, decdzeta, decda2 = c(rs, zeta, a2)
    decdrsn = (c(rs + d, zeta, a2)[0] - c(rs - d, zeta, a2)[0]) / d / 2
    decdzetan = (c(rs, zeta + d, a2)[0] - c(rs, zeta - d, a2)[0]) / d / 2
    decda2n = (c(rs, zeta, a2 + d)[0] - c(rs, zeta, a2 - d)[0]) / d / 2

    ec0, dec0drs, dec0dzeta, dec0da2 = c(rs, 0.0, a2)
    dec0drsn = (c(rs + d, 0.0, a2)[0] - c(rs - d, 0.0, a2)[0]) / d / 2
    dec0dzetan = (c(rs, 0.0 + d, a2)[0] - c(rs, 0.0 - d, a2)[0]) / d / 2
    dec0da2n = (c(rs, 0.0, a2 + d)[0] - c(rs, 0.0, a2 - d)[0]) / d / 2

    ec00, dec00drs, dec00dzeta, dec00da2 = c0(rs, a2)
    dec00drsn = (c0(rs + d, a2)[0] - c0(rs - d, a2)[0]) / d / 2
    dec00da2n = (c0(rs, a2 + d)[0] - c0(rs, a2 - d)[0]) / d / 2

    error = 0.0
    for E in [(dexdrs, dexdrsn),
              (dexda2, dexda2n),
              (decdrs, decdrsn),
              (decdzeta, decdzetan),
              (decda2, decda2n),
              (ec0, ec00),
              (dec0drs, dec00drs, dec0drsn, dec00drsn),
              (dec0da2, dec00da2, dec0da2n, dec00da2n),
              (0.0, dec0dzetan, dec0dzeta, dec00dzeta)]:
        for e in E[1:]:
            de = abs(e - E[0])
            if de > error:
                error = de
    assert error < 3.5e-9
    print name, error


