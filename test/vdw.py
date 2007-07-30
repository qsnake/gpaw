from math import sqrt, exp, pi
from gpaw.vdw import VanDerWaals
import Numeric as num
n = 48
d = num.ones((2 * n, n, n), num.Float)
a = 4.0
c = a / 2
h = a / n
for x in range(2 * n):
    for z in range(n):
        for y in range(n):
            r = sqrt((x * h - c)**2 + (y * h - c)**2 + (z * h - c)**2)
            d[x, y, z] = exp(-2 * r) / pi

print num.sum(d.flat) * h**3
uc = num.array([(2 * a,0,0),(0,a,0),(0,0,a)])
e1 = VanDerWaals(d, unitcell=uc,xcname='revPBE').GetEnergy(n=4)
d += d[::-1].copy()
e2 = VanDerWaals(d, unitcell=uc,xcname='revPBE').GetEnergy(n=4)
print e1, e2, 2 * e1 - e2
