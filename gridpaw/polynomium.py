import sys
from math import pi

import Numeric as num


a_i = num.array((1.0, 0.0, -3.0, 2.0))
i_i = num.arange(4)
c_l = num.zeros(3, num.Float)
b_lj = num.zeros((3, 6), num.Float)
for l in range(3):
    c_l[l] = 1.0 / num.sum(a_i / (3 + 2 * l + i_i))
    b_lj[l, 2:6] = 4 * pi * c_l[l] * a_i / \
                   (l * (l + 1) - (i_i + 2 + l) * (i_i + 3 + l))
    b_lj[l, 0] = 4 * pi / (2 * l + 1) - num.sum(b_lj[l])


I_l = num.zeros(3, num.Float)
for l in range(3):
    for i in range(4):
        for j in range(6):
            I_l[l] += c_l[l] * a_i[i] * b_lj[l, j] / (i + j + 2 * l + 3)


if __name__ == '__main__':
    rc = 2.1
    x = num.arange(150) / 100.0
    g_lg = num.zeros((3, 150), num.Float) 
    v_lg = num.zeros((3, 150), num.Float) 
    s_g = num.zeros(150, num.Float)
    for i in range(4):
        s_g += a_i[i] * x**i
    s_g[100:] = 0.0

    for l in range(3):
        g_lg[l] = c_l[l] * x**l / rc**(3 + l) * s_g
        for j in range(6):
            v_lg[l] += b_lj[l, j] * x**j
        v_lg[l] *= x**l / rc**(l + 1)
        v_lg[l, 100:] = 4 * pi / (2 * l + 1) / (rc * x[100:])**(l + 1)
        print >> sys.stderr, l, I_l[l] / rc**(2 * l + 1) - \
              num.sum(v_lg[l] * g_lg[l] * x**2) * rc**3 / 100

    for g in range(150):
        r = x[g] * rc
        print r,
        for l in range(3):
            print g_lg[l, g], v_lg[l, g],
        print
