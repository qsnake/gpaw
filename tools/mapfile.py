# Mostly been tested on BG/P at Argonne National Laboratory
# Longest partition dimensions (P_x, P_y, P_z) for this BG/P
# is Z-axis. We fold the torus dimension into the Z-axis, making
# it even longer. For band parallelization, we "stack" adjacent
# bands along Z-axis from bottom (0,0,0) to top (0,0,P_z). This
# should also work for spin polarized calculation for a judicous
# choice of --state-parallelization=B.
from sys import argv
mode = argv[1]
nodes = int(argv[2])
ppn = 4
assert mode == 'domain' or 'band'
if mode == 'domain':
    shape = [int(d) for d in argv[3].split(',')]
    assert len(shape) == 4

if mode == 'band':
    strides = [str(d) for d in argv[3].split(',')]
    assert len(strides) == 4
    for i in range(4):
        print strides
        assert strides[i] in ['X','x','Y','y','Z','z','T','t']
        stride2dim = { 'X':0, 'x': 0,
                       'Y':1, 'y': 1,
                       'Z':2, 'z': 2,
                       'T':3, 't': 3}
        
layout = {   64: (4,  4,  4, ppn ),
            128: (4,  4,  8, ppn ),
            256: (8,  4,  8, ppn ), 
            512: (8,  8,  8, ppn ),
           1024: (8,  8,  16, ppn),
           2048: (8,  8,  32, ppn),
           4096: (8,  16, 32, ppn),
           8192: (8,  32, 32, ppn),
          16384: (16, 32, 32, ppn),
          24576: (24, 32, 32, ppn),
          32768: (32, 32, 32, ppn),
          40960: (40, 32, 32, ppn)}[nodes]
cores = nodes * ppn
if mode == 'domain':
    domains = shape[0] * shape[1] * shape[2]
    blocks = cores // domains
    assert blocks * domains == nodes * ppn
    assert blocks == shape[3]
    A = layout[0] // shape[0]
    B = layout[1] // shape[1]
    C = layout[2] // shape[2] * ppn # torus dimension folds into z-axes
    rank = 0
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
                    if (rank > (cores // A - 1)) or (rank > (cores // B - 1)):
                        reflections = (-1)**(cores // A - 1)
                        if reflections == -1:
                            z = layout[2] - 1 - z
                    # For dual mode on BG/P, we *may* need
                    # to use core 2 (not core 1) as the
                    # 2nd core
                    # if (ppn == 2) and (t == 1):
                    #     t = 2
                    print x, y, z, t
                    rank += 1

if mode == 'band':
    var0 = strides[0]
    var1 = strides[1]
    var2 = strides[2]
    var3 = strides[3]
    stride0 = layout[stride2dim[strides[0]]]
    stride1 = layout[stride2dim[strides[1]]]
    stride2 = layout[stride2dim[strides[2]]]
    stride3 = layout[stride2dim[strides[3]]]                 
    for var3 in range(stride3): # bands
        for var0 in range(stride0): # x
            for var1 in range(stride1): # y 
                for var2 in range(stride2): # z domains are the fastest
                    print var0, var1, var2, var3
