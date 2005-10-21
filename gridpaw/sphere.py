# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import pickle
import Numeric as num
from math import pi


## N = 169
N = 49

if __name__ == '__main__':
    from spherical_harmonics import Y
    print 'Constructing "%d.pickle" ...' % N
    weights = num.zeros(N, num.Float)
    YY = num.zeros((N, 45), num.Float)
    Y_nL = num.zeros((N, 9), num.Float)
    points = num.zeros((N, 3), num.Float)
    n = 0
    for line in open('../FliegeMaier/%d.txt' % N, 'r'):
        x, y, z, w = [float(s) for s in line.split()]
        weights[n] = w
        Y_L = [Y(L, x, y, z) for L in range(9)]
        Y_nL[n] = Y_L
        points[n] = (x, y, z)
        LL = 0
        for L1 in range(9):
            for L2 in range(L1, 9):
                YY[n, LL] = Y_L[L1] * Y_L[L2]
                LL += 1
        n += 1
    weights /= 4.0 * pi
    pickle.dump((weights, YY, Y_nL, points), open('%d.pickle' % N, 'w'))
else:
    from gridpaw import home
    filename = home + '/.gridpaw/setups/%d.pickle' % N
    weights, YY, Y_nL, points = pickle.load(open(filename))
