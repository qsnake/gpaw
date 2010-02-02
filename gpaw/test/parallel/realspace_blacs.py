"""Test of BLACS Redistributor.

Requires at least 8 MPI tasks.
"""

import sys

import numpy as np

from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world, distribute_cpus
from gpaw.utilities.blacs import scalapack_set 
from gpaw.blacs import BlacsGrid, Redistributor, parallelprint, \
    BlacsBandDescriptor

G = 120  # number of grid points (G x G x G)
N = 10  # number of bands

# B: number of band groups
# D: number of domains
B = 2
D = 2

M = N // B     # number of bands per group
assert M * B == N, 'M=%d, B=%d, N=%d' % (M,B,N)

h = 0.2        # grid spacing
a = h * G      # side length of box

# Set up communicators:
domain_comm, kpt_comm, band_comm = distribute_cpus(parsize=D, parsize_bands=B, \
                                                   nspins=1, nibzkpts=2)
assert world.size >= D*B*kpt_comm.size

if world.rank == 0:
    print 'MPI: %d domains, %d band groups, %d kpts' % (domain_comm.size, band_comm.size, kpt_comm.size)

# Set up band and grid descriptors:
bd = BandDescriptor(N, band_comm, False)
gd = GridDescriptor((G, G, G), (a, a, a), True, domain_comm, parsize=D)

mcpus, ncpus, blocksize = 2, 2, 6

# horrible acronym
def main(seed=42, dtype=float):
    bbd = BlacsBandDescriptor(world, gd, bd, kpt_comm, mcpus, ncpus, blocksize)
    nbands = bd.nbands
    mynbands = bd.mynbands

    # Diagonalize
    # We would *not* create H_Nn in the real-space code this way.
    # This is just for testing purposes.
    # Note after MPI_Reduce, only meaningful information on gd masters
    H_Nn = bbd.Nndescriptor.zeros(dtype=dtype)
    scalapack_set(bbd.Nndescriptor, H_Nn, 0.1, 75.0, 'L')
    # We would create U_nN in the real-space code this way.
    U_nN = np.empty((mynbands, nbands), dtype=dtype)
    diagonalizer = bbd.get_diagonalizer()
    eps_n = np.zeros(bd.mynbands)
    diagonalizer.diagonalize(H_Nn, U_nN, eps_n)
    print 'after broadcast'
    parallelprint(world, U_nN)
    parallelprint(world, eps_n)
    
    # Inverse Cholesky
    S_Nn = bbd.Nndescriptor.zeros(dtype=dtype)
    scalapack_set(bbd.Nndescriptor, S_Nn, 0.1, 75.0, 'L')
    C_nN = np.empty((mynbands, nbands), dtype=dtype)
    diagonalizer.inverse_cholesky(S_Nn, C_nN)
    print 'after cholesky'
    parallelprint(world, C_nN)

if __name__ == '__main__':
    main(dtype=float)
    # main(dtype=complex)


