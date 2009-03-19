from time import time
import sys
import numpy as np
from gpaw import parsize, parsize_bands
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.operators import Laplace
from gpaw.mpi import world
from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import inverse_cholesky
import _gpaw

B = parsize_bands   # number of blocks
    
G = 120  # number of grid points (G x G x G)
N = 16  # number of bands

h = 0.2        # grid spacing
a = h * G      # side length of box
M = N // B     # number of bands per block
assert M * B == N

D = world.size // B  # number of domains
assert D * B == world.size

# Set up communicators:
r = world.rank // D * D
domain_comm = world.new_communicator(np.arange(r, r + D))
band_comm = world.new_communicator(np.arange(world.rank % D, world.size, D))
scalapack0_comm = world.new_communicator(np.arange(0, world.size, D))
scalapack1_comm = world.new_communicator(np.arange(0, world.size, D // B))
    
# Set up domain and grid descriptors:
domain = Domain((a, a, a))
domain.set_decomposition(domain_comm, parsize, N_c=(G, G, G))
gd = GridDescriptor(domain, (G, G, G))

# Random wave functions:
np.random.seed(world.rank)
psit_mG = np.random.uniform(-0.5, 0.5, size=(M,) + tuple(gd.n_c))
if world.rank == 0:
    print 'Size of wave function array:', psit_mG.shape

# Send and receive buffers:
send_mG = gd.empty(M)
recv_mG = gd.empty(M)

def run():
    S_nn = overlap(psit_mG, send_mG, recv_mG)

    # t1 = time()
    # if world.rank == 0:
    #     inverse_cholesky(S_nn)
    #     C_nn = S_nn
    # else:
    #     C_nn = np.empty((N, N))
    # t2 = time()

    # if world.rank == 0:
    #     print 'Cholesky Time %f' % (t2-t1)
        
    # Distribute matrix:
    # world.broadcast(C_nn, 0)

    # psit_mG[:] = matrix_multiply(C_nn, psit_mG, send_mG, recv_mG)

    if world.rank == 0:
        print 'Made it past matrix multiply'

    # Check:
    # S_nn = overlap(psit_mG, send_mG, recv_mG)

    # Assert below requires more memory.
    # if world.rank == 0:
    #   # Fill in upper part:
    #   for n in range(N - 1):
    #      S_nn[n, n + 1:] = S_nn[n + 1:, n]
    #      assert (S_nn.round(7) == np.eye(N)).all()

def overlap(psit_mG, send_mG, recv_mG):
    """Calculate overlap matrix.

    Compute the entire overlap matrix and put the columns
    in the correct order."""

    rank = band_comm.rank
    S_imm = np.empty((B, M, M))
    send_mG[:] = psit_mG

    # Shift wave functions:
    for i in range(B - 1):
        rrequest = band_comm.receive(recv_mG, (rank + 1) % B, 42, False)
        srequest = band_comm.send(send_mG, (rank - 1) % B, 42, False)
        # Index for correct order in S_imm
        j = (rank - i) % B
        gemm(gd.dv, psit_mG, send_mG, 0.0, S_imm[j], 'c')
        band_comm.wait(rrequest)
        band_comm.wait(srequest)
        send_mG, recv_mG = recv_mG, send_mG

    j = (rank - (B - 1)) % B
    gemm(gd.dv, psit_mG, send_mG, 0.0, S_imm[j], 'c')

    # This will put S_imm on every rank
    domain_comm.sum(S_imm)

    # Blocks of matrix on each processor become one matrix.
    # We can think of this as a 1D block matrix
    if (scalapack0_comm):
        S_nm = S_imm.reshape(N,M)
    else:
        S_nm = None
    del S_imm

    # Test, N = 16
    # Step 0 - Initialize distributed Matrix to None;
    # Otherwise, scalapack_redist will complain of UnboundLocalError
    if world.rank == 0: print "before redist"
    H_nm = None 
    S_nm = None
    B_mm = None
     
    # Step 1 - Fill matrices with values and convert to Fortran order 
    # Create a simple matrix H: diagonal elements + 0.1 off-diagonal
    # Create a simple overlap matrix: unity + 0.2 off-diagonal
    if (scalapack0_comm):
        H_nm = scalapack0_comm.rank*np.eye(N,M,-M*scalapack0_comm.rank)
        H_nm = H_nm[:,0:4] + 0.1*np.eye(N,M,-M*scalapack0_comm.rank+1)
        H_nm = H_nm.copy("Fortran") # Fortran order required for ScaLAPACK
        S_nm = np.eye(N,M,-M*scalapack0_comm.rank)
        S_nm = S_nm[:,0:4] + 0.2*np.eye(N,M,-M*scalapack0_comm.rank+1)
        S_nm = S_nm.copy("Fortran") # Fortran order required for ScaLAPACK
        print scalapack0_comm.rank, "H_nm =", H_nm
        print scalapack0_comm.rank, "S_nm =", S_nm

    # Step 2 - Create descriptor for distributed arrays.
    # Desc for H_nm : 1D grid
    desc0 = _gpaw.blacs_create(scalapack0_comm,N,N,1,B,N,M)
    # Desc for H_mm : 2D grid
    desc1 = _gpaw.blacs_create(scalapack1_comm,N,N,B,B,4,4)

    # Step 3 - Redistribute from 1D -> 2D grid
    # This is mostly for performance reasons
    H_mm = _gpaw.scalapack_redist(H_nm,desc0,desc1)
    S_mm = _gpaw.scalapack_redist(S_nm,desc0,desc1)

    # Make backup copy for inverse Cholesky test later
    if S_mm is not None:
        B_mm = S_mm.copy("Fortran")
    
    # Debug
    # if world.rank == 0: print 'redistributed array'
    # if H_mm is not None:
    #     print "H_mm Fortran order =", H_mm.flags.f_contiguous
    #     print H_mm
    # if S_mm is not None:
    #     print "S_mm Fortran order =", S_mm.flags.f_contiguous
    #     print S_mm
                
    # Step 4 - Call ScaLAPACK diagonalize
    # Call scalapack diagonalize D&C
    # W, Z_mm  = _gpaw.scalapack_diagonalize_dc(H_mm, desc1)

    # Call ScaLAPACK general diagonalize
    W, Z_mm = _gpaw.scalapack_general_diagonalize(H_mm, S_mm, desc1)

    # Debug
    # Eigenvectors
    # if Z_mm is not None:
    #     print "Z_mm Fortran order =", Z_mm.flags.f_contiguous

    # Step 5 - Redistribute from 2D -> 1D grid
    # Copy from Z_mm -> H_nm, we re-use arrays whenever possible
    # Note that W is still only on scalapack1_comm, i.e. desc1, and
    # contains *all* eigenvalues
    H_nm = _gpaw.scalapack_redist(Z_mm,desc1,desc0)

    # Debug
    # if world.rank == 0: print 'result in original distribution'
    # if H_nm is not None:
    #      print "H_nm Fortran order =", H_nm.flags.f_contiguous
        
    # Step 6 - Convert to C array for general use in GPAW.
    if H_nm is not None:
        H_nm = H_nm.copy("C")
        print scalapack0_comm.rank, "W =", W 
        print scalapack0_comm.rank, "H_nm =", H_nm

    # Step 7 - Inverse cholesky test
    _gpaw.scalapack_inverse_cholesky(B_mm,desc1)

    # Step 8 - Redistribute from 2D -> 1D grid
    # Copy from B_mm -> B_nm
    B_nm = _gpaw.scalapack_redist(B_mm,desc1,desc0)

    # Step 9 - Convert to C array for general use in GPAW
    if B_nm is not None:
        B_nm = B_nm.copy("C")
        print scalapack0_comm.rank, "B_nm =", B_nm
        
    _gpaw.blacs_destroy(desc0)
    _gpaw.blacs_destroy(desc1)

def matrix_multiply(C_nn, psit_mG, send_mG, recv_mG):
    """Calculate new linear compination of wave functions."""
    rank = band_comm.rank
    C_imim = C_nn.reshape((B, M, B, M))
    send_mG[:] = psit_mG
    psit_mG[:] = 0.0
    beta = 0.0
    for i in range(B - 1):
        rrequest = band_comm.receive(recv_mG, (rank + 1) % B, 117, False)
        srequest = band_comm.send(send_mG, (rank - 1) % B, 117, False)
        gemm(1.0, send_mG, C_imim[rank, :, (rank + i) % B], beta, psit_mG)
        beta = 1.0
        band_comm.wait(rrequest)
        band_comm.wait(srequest)
        send_mG, recv_mG = recv_mG, send_mG
    gemm(1.0, send_mG, C_imim[rank, :, rank - 1], beta, psit_mG)

    return psit_mG

ta = time()

# Do twenty iterations
for x in range(1):
    run()

tb = time()

if world.rank == 0:
    print 'Total Time %f' % (tb -ta)
    
