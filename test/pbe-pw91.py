# This test takes approximately 0.0 seconds
from math import pi, sqrt
from gpaw.xc_functional import XCFunctional

x0 = XCFunctional('LDAx')

def f0(xc, rs, s):
    n = 3 / (4 * pi * rs**3)
    third = 1.0 / 3.0
    kF = (3 * pi**2 * n)**third
    a2 = (2 * kF * n * s)**2
    exc = (n * xc.exchange(rs, a2)[0] +
           n * xc.correlation(rs, 0.0, a2)[0])
    ex0 = n * x0.exchange(rs)[0]
    return exc / ex0

def f1(xc, rs, s):
    n = 3 / (4 * pi * rs**3)
    na = 2 * n
    third = 1.0 / 3.0
    kF = (3 * pi**2 * n)**third
    rsa = (3 / pi / 4 / na)**third
    a2 = (2 * kF * n * s)**2
    exc = (n * xc.exchange(rsa, 4 * a2)[0] +
           n * xc.correlation(rs, 1.0, a2)[0])
    ex0 = n * x0.exchange(rs)[0]
    return exc / ex0

pbe = XCFunctional('PBE')
pw91 = XCFunctional('PW91')
assert abs(f0(pbe, 2, 3) - 1.58) < 0.01
assert abs(f1(pbe, 2, 3) - 1.88) < 0.01
assert abs(f0(pw91, 2, 3) - 1.60) < 0.01
assert abs(f1(pw91, 2, 3) - 1.90) < 0.01

if 0:
    from pylab import *

    f = f0
    #f= f1

    s = linspace(0, 3, 16)
    t = '-'
    for xc in [pbe, pw91]:
        for rs in [0.0001, 2, 10]:
            plot(s, [f(xc, rs, x) for x in s], t)
        t = 'o'
    show()
