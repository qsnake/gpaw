# Mostly been tested on BG/P at Argonne National Laboratory
# Longest partition dimensions (P_x, P_y, P_z) for this BG/P
# is Z-axis. We fold the torus dimension into the Z-axis, making
# it even longer. For band parallelization, we "stack" adjacent
# bands along Z-axis from bottom (0,0,0) to top (0,0,P_z). This
# should also work for spin polarized calculation for a judicous
# choice of --state-parallelization=B.

from sys import argv
import numpy as np
nodes = int(argv[1])
ppn = 2
shape = [int(x) for x in argv[2].split(',')]
mode = 'normal'
# print "Shape =", shape
if len(argv) == 4:
    mode = argv[3]
layout = {   64: (4,  4,  4 ),
            128: (4,  4,  8 ),
            256: (8,  4,  8 ), 
            512: (8,  8,  8 ),
           1024: (8,  8,  16),
           2048: (8,  8,  32),
           4096: (8,  16, 32),
           8192: (8,  32, 32),
          16384: (16, 32, 32),
          24576: (24, 32, 32),
          32768: (32, 32, 32),
          40960: (40, 32, 32)}[nodes]
domains = shape[0] * shape[1] * shape[2]
blocks = nodes * ppn // domains
# print "State parallelization = ", blocks
assert blocks * domains == nodes * ppn
A = layout[0] // shape[0]
B = layout[1] // shape[1]
C = layout[2] // shape[2] * ppn # torus dimension folds into z-axes
check = np.zeros(layout + (4,), int)
rank = 0
if mode == 'normal':
    for n in range(blocks):
        x0 = (n // (B*C)) * shape[0]
        y0 = ((n // C) % B) * shape[1]
        z0 = (n % C) * shape[2] // ppn 
        # coordinate of block below
        # print "n, x0,y0,z0=", n, x0,y0,z0
        for a in range(shape[0]):
            for b in range(shape[1]):
                for c in range(shape[2]):
                    x = x0 + a
                    y = y0 + b
                    z = z0 + c // ppn
                    t = c % ppn
                    check[x, y, z, t] = 1
                    print x, y, z, t
                rank += 1
else:
    assert mode == 'shared'
    for n in range(blocks):
        t = n % ppn
        m = n // ppn
        x0 = (m % A) * shape[0]
        y0 = ((m // A) % B) * shape[1]
        z0 = (m // (A * B)) * shape[2]
        for a in range(shape[0]):
            for b in range(shape[1]):
                for c in range(shape[2]):
                    x = x0 + a
                    y = y0 + b
                    z = z0 + c
                    check[x, y, z, t] = 1
                    print x, y, z, t
                rank += 1
assert check.sum() == nodes * ppn
