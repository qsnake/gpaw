from math import pi, sqrt
from gpaw.xc_functional import XCFunctional

nspin_1 = 1
nspin_2 = 2

x0 = XCFunctional('LDAx', nspin_1)

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

pbe_1 = XCFunctional('PBE', nspin_1)
pbe_2 = XCFunctional('PBE', nspin_2)
pw91_1 = XCFunctional('PW91', nspin_1)
pw91_2 = XCFunctional('PW91', nspin_2)
assert abs(f0(pbe_1, 2, 3) - 1.58) < 0.01
assert abs(f1(pbe_2, 2, 3) - 1.88) < 0.01
assert abs(f0(pw91_1, 2, 3) - 1.60) < 0.01
assert abs(f1(pw91_2, 2, 3) - 1.90) < 0.01

if 0:
    from pylab import *

    f = f0
    #f= f1

    s = linspace(0, 3, 16)
    t = '-'
    for xc in [pbe_1, pw91_1]:
        for rs in [0.0001, 2, 10]:
            plot(s, [f(xc, rs, x) for x in s], t)
        t = 'o'
    show()
