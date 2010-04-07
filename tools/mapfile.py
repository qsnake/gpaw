# Mostly been tested on BG/P at Argonne National Laboratory
# There are two modes for generating map files:
#
# Domain - used for band parallelization in TDDFT; 
#          keep domains for same group of bands 
#          on adjacent nodes
#          
# Band   - used for band parallelization in DFT; 
#          keep domains for nearest-neighbor bands
#          on adjacent nodes
#
# usage:
#
# Domain
# ------
# Four integers belows must match the integers used for
# domain-decomposition (first three) and
# state-parallelization (last):
#
# python mapfile.py domain <number of nodes> 4,4,4,8 > <filename>
#
# Band
# ----
# Four characters below must match the dimensions used for
# domain-decomposition (first three) and 
# state-parallelzation (last):
#
# python mapfile.py band <number of nodes> X,Y,T,Z > <filanme>
#
# Further reading notes available here:
# https://wiki.fysik.dtu.dk/gpaw/install/BGP/performance.html
#
# Some results can be found in doc/devel/Au_cluster
#
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
    stride2dim = [0, 0, 0, 0]
    for i in range(4):
        assert strides[i] in ['X','x','Y','y','Z','z','T','t']
        str2num = { 'X':0, 'x': 0,
                    'Y':1, 'y': 1,
                    'Z':2, 'z': 2,
                    'T':3, 't': 3}[strides[i]]
        stride2dim[i] = str2num
    assert sum(stride2dim) == 6
        
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
    # Longest partition dimensions (P_x, P_y, P_z) for this BG/P
    # is Z-axis. We fold the torus dimension into the Z-axis, making
    # it even longer. For band parallelization, we "stack" adjacent
    # bands along Z-axis from bottom (0,0,0) to top (0,0,P_z). This
    # should also work for spin polarized and k-points for a judicous
    # of the fourth parameter.
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
                    # The reflection term causes ranks to reflect of the 
                    # boundary instead of wrapping to the other side.
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
    # The fourth comma seperated parameters will assign dimensions
    # in order of domain_x, domain_y, domain_z, blocks of bands.
    # Here we assume that GPAW ranks are assigned in the following order
    # Z, Y, X, bands, spins and k-points
    input_strU = argv[3].upper()
    input_strL = argv[3].lower()
    ordinary = 'XYZT'
    input_strU2 = input_strU.replace(',','') # input_strU without commas
    perm=[]
    for d in range(4):
        fx=ordinary[d]
        p=input_strU2.index(fx)
        perm.append(p)

    X, Y, Z, T = [layout[perm[i]] for i in range(4)]

    for t in range(T):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    a, b, c, d = eval(input_strL)
                    print a, b, c, d
                    
            
