"""Test of BLACS Redistributor.

Requires at least 8 MPI tasks.
"""

import sys

import numpy as np

from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world, distribute_cpus
from gpaw.utilities import scalapack
from gpaw.utilities.blacs import scalapack_set, scalapack_zero 
from gpaw.blacs import BlacsGrid, Redistributor, parallelprint, \
    BlacsBandLayouts

G = 120  # number of grid points (G x G x G)
N = 10  # number of bands

# B: number of band groups
# D: number of domains
B = 2
D = 2

n = N // B     # number of bands per group
assert n * B == N, 'n=%d, B=%d, N=%d' % (n, B, N)

h = 0.2        # grid spacing
a = h * G      # side length of box

# Set up communicators:
domain_comm, kpt_comm, band_comm = distribute_cpus(parsize=D, parsize_bands=B, \
                                                   nspins=1, nibzkpts=2)
assert world.size == D*B*kpt_comm.size

if world.rank == 0:
    print 'MPI: %d domains, %d band groups, %d kpts' % (domain_comm.size, band_comm.size, kpt_comm.size)

# Set up band and grid descriptors:
bd = BandDescriptor(N, band_comm, False)
gd = GridDescriptor((G, G, G), (a, a, a), True, domain_comm, parsize=D)

mcpus, ncpus, blocksize = 2, 2, 6

def main(seed=42, dtype=float):
    ksl = BlacsBandLayouts(gd, bd, mcpus, ncpus, blocksize)
    nbands = bd.nbands
    mynbands = bd.mynbands

    # Diagonalize
    # We would *not* create H_Nn in the real-space code this way.
    # This is just for testing purposes.
    # Note after MPI_Reduce, only meaningful information on gd masters
    H_Nn = np.zeros((nbands, mynbands), dtype=dtype)
    U_nN = np.empty((mynbands, nbands), dtype=dtype)

    if ksl.Nndescriptor: # hack
        scalapack_set(ksl.Nndescriptor, H_Nn, 0.1, 75.0, 'L')
    else:
        gd.comm.rank != 0

    print "H_Nn"
    parallelprint(world, H_Nn)
    
    diagonalizer = ksl.get_diagonalizer()
    eps_n = np.zeros(bd.mynbands)
    diagonalizer.diagonalize(H_Nn, U_nN, eps_n)
    print "U_nN"
    parallelprint(world, U_nN)
    print "eps_n"
    parallelprint(world, eps_n)
    
    # Inverse Cholesky
    S_Nn = np.zeros((nbands, mynbands), dtype=dtype)
    C_nN = np.empty((mynbands, nbands), dtype=dtype) 

    if ksl.Nndescriptor: # hack
        scalapack_set(ksl.Nndescriptor, S_Nn, 0.1, 75.0, 'L')
    else:
        gd.comm.rank != 0

    print "S_Nn"
    parallelprint(world, S_Nn)
    diagonalizer.inverse_cholesky(S_Nn, C_nN)
    print "C_nN"
    parallelprint(world, C_nN)

if __name__ in ['__main__', '__builtin__']:
    if not scalapack(True):
        print('Not built with ScaLAPACK. Test does not apply.')
    else:
        main(dtype=float)
        main(dtype=complex)


