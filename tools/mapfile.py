from sys import argv
import numpy as np
nodes = int(argv[1])
ppn = 4
shape = [int(x) for x in argv[2].split(',')]
mode = 'normal'
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
assert blocks * domains == nodes * ppn
A = layout[0] // shape[0]
B = layout[1] // shape[1]
check = np.zeros(layout + (4,), int)
rank = 0
if mode == 'normal':
    for n in range(blocks):
        x0 = (n % A) * shape[0]
        y0 = ((n // A) % B) * shape[1]
        z0 = (n // (A * B)) * shape[2] // ppn
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
