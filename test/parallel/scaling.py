from time import time
try:
    import numpy as npy
    npy_float = npy.float
except ImportError:
    try:
        import Numeric as npy
        npy_float = npy.Float
    except ImportError:
        raise SystemExit('numpy nor Numeric not installed!')
from gpaw import mpi
from gpaw.operators import Laplace
from gpaw.transformers import Transformer
from gpaw.grid_descriptor import GridDescriptor
from gpaw.preconditioner import Preconditioner

def run(ngpts, repeat, narrays, prec=False):
    if mpi.rank == 0:
        out = open('timings-%d.dat' % ngpts, 'w')
    else:
        out = None
    p = 1
    while p <= mpi.size and ngpts**3 / p > 4**3:
        if mpi.rank == 0:
            out.write('%4d' % p)
        comm = mpi.world.new_communicator(npy.arange(p))
        if comm is not None:
            go(comm, ngpts, repeat, narrays, out, prec)
        mpi.world.barrier()
        p *= 2
    if mpi.rank == 0:
        out.close()

def go(comm, ngpts, repeat, narrays, out, prec):
    N_c = npy.array((ngpts, ngpts, ngpts))
    a = 10.0
    gd = GridDescriptor(N_c, (a, a, a), comm=comm))
    gdcoarse = gd.coarsen()
    gdfine = gd.refine()
    kin1 = Laplace(gd, -0.5, 1).apply
    laplace = Laplace(gd, -0.5, 2)
    kin2 = laplace.apply
    restrict = Transformer(gd, gdcoarse, 1).apply
    interpolate = Transformer(gd, gdfine, 1).apply
    precondition = Preconditioner(gd, laplace, npy_float)
    a1 = gd.empty(narrays)
    a1[:] = 1.0
    a2 = gd.empty(narrays)
    c = gdcoarse.empty(narrays)
    f = gdfine.empty(narrays)

    T = [0, 0, 0, 0, 0]
    for i in range(repeat):
        comm.barrier()
        kin1(a1, a2)
        comm.barrier()
        t0a = time()
        kin1(a1, a2)
        t0b = time()
        comm.barrier()
        t1a = time()
        kin2(a1, a2)
        t1b = time()
        comm.barrier()
        t2a = time()
        for A, C in zip(a1,c):
            restrict(A, C)
        t2b = time()
        comm.barrier()
        t3a = time()
        for A, F in zip(a1,f):
            interpolate(A, F)
        t3b = time()
        comm.barrier()
        if prec:
            t4a = time()
            for A in a1:
                precondition(A, None, None, None)
            t4b = time()
            comm.barrier()

        T[0] += t0b - t0a
        T[1] += t1b - t1a
        T[2] += t2b - t2a
        T[3] += t3b - t3a
        if prec:
            T[4] += t4b - t4a

    if mpi.rank == 0:
        out.write(' %2d %2d %2d' % tuple(gd.parsize_c))
        out.write(' %12.6f %12.6f %12.6f %12.6f %12.6f\n' %
                  tuple([t / repeat / narrays for t in T]))
        out.flush()

if __name__ == '__main__':
    run(128, 50, 2, prec=True)
    run(32, 50, 20)
    #run(32, 1, 1, prec=1)
