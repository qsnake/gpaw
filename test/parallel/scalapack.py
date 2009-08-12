# Note that GPAW does not do transpose for calls
# to LAPACK involving operations on symmetric matrices. 
# Hence, UPLO = 'U' in Fortran equals UPLO = 'L' in
# NumPy C-style. For simplicity, we will
# convert everything here to Fortran style since
# this is the default for ScaLAPACK
#
# Here we compare ScaLAPACK results for a 16-by-16 matrix
# with those obtain with serial LAPACK
import numpy as np

from gpaw import GPAW
from gpaw import debug
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize, inverse_cholesky
from gpaw.utilities.blacs import *

# We could possibly have a stricter criteria, but these are all
# the printed digits at least
w_tol = 1.e-8
z_tol = 1.e-8
c_tol = 1.e-8


N = 16
B = 4
M = N/B

# blacs grid dimensions DxD and non-even blocking factors just
# to make things more non-trivial.
D = 2 
nb = 3
mb = 3

assert world.size == B
assert world.size >= D*D

A = np.empty((N,N))
A[:,0:M] = 0.0*np.eye(N,M,0)
A[:,0:M] = A[:,0:M]+ 0.1*np.eye(N,M,-M*0+1) # shift off of diag +1
A[:,M:2*M] = 1.0*np.eye(N,M,-M)
A[:,M:2*M] = A[:,M:2*M] + 0.1*np.eye(N,M,-M*1+1) # shift off of diag +1
A[:,2*M:3*M] = 2.0*np.eye(N,M,-M*2)
A[:,2*M:3*M] = A[:,2*M:3*M] + 0.1*np.eye(N,M,-M*2+1) # shift off of diag +1 
A[:,3*M:4*M] = 3.0*np.eye(N,M,-M*3)
A[:,3*M:4*M] = A[:,3*M:4*M]+ 0.1*np.eye(N,M,-M*3+1) # shift off of diag +1
if world.rank == 0:
    print "A = ", A
# We should really use Fortran ordered array but this gives
# a false positive in LAPACK's debug mode
# A = A.copy("Fortran")
A = A.transpose().copy()

S = np.empty((N,N))
S[:,0:M] = 1.0*np.eye(N,M,0)
S[:,0:M] = S[:,0:M]+ 0.2*np.eye(N,M,-M*0+1) # shift off of diag +1
S[:,M:2*M] = 1.0*np.eye(N,M,-M)
S[:,M:2*M] = S[:,M:2*M] + 0.2*np.eye(N,M,-M*1+1) # shift off of diag +1
S[:,2*M:3*M] = 1.0*np.eye(N,M,-M*2)
S[:,2*M:3*M] = S[:,8:12] + 0.2*np.eye(N,4,-M*2+1) # shift off of diag +1 
S[:,3*M:4*M] = 1.0*np.eye(N,M,-M*3)
S[:,3*M:4*M] = S[:,3*M:4*M] + 0.2*np.eye(N,4,-M*3+1) # shift off of diag +1
if world.rank == 0:
    print "S = ", S
# We should really use Fortran ordered array but this gives
# a false positive in LAPACK's debug mode
# S = S.copy("Fortran")
S = S.transpose().copy()
w = np.empty(N)

# We need to make a backup of A since LAPACK diagonalize will destroy it
A2 = A.copy()
info = diagonalize(A2, w)

if world.rank == 0:
    if info != 0:
        print "WARNING: diagonalize info = ", info
    print "diagonalize: lambda = ", w
    print "diagonalize: eigenvectors = ", A2

wg = np.empty(N)

# We need to make a backup of S since LAPACK diagonalize will destroy it
S2 = S.copy()
info = diagonalize(A, wg, S2)

if world.rank == 0:
    if info != 0:
        print "WARNING: general diagonalize info = ", info
    print "general diagonalize: lambda = ", wg
    print "general diagonalize: eigenvectors = ", A

# For consistency, also make a backup of S since LAPACK will destroy it
C = S.copy()
info = inverse_cholesky(C)

if world.rank == 0:
    if info != 0:
        print "WARNING: general diagonalize info = ", info
    print "cholesky = ", C

# Initialize distributed matrices to None;
# Otherwise, scalapack_redist will complain of UnboundLocalError
# This happens because scalapack local array does not exist on
# both the source and destination communications. As this is a
# simple example, we should be OK here without it. 
A_nm = None
S_nm = None
C_nm = None
C_mm = None

# Create arrays in parallel
A_nm = world.rank*np.eye(N,M,-M*world.rank)
A_nm = A_nm[:,0:N/B] + 0.1*np.eye(N,M,-M*world.rank+1)
A_nm = A_nm.copy("Fortran") # Fortran order required for ScaLAPACK
S_nm = np.eye(N,M,-M*world.rank)
S_nm = S_nm[:,0:N/B] + 0.2*np.eye(N,M,-M*world.rank+1)
S_nm = S_nm.copy("Fortran") # Fortran order required for ScaLAPACK
C_nm = S_nm.copy("Fortran")

# Create descriptors
# Desc for serial : 0-D grid
desc0 = blacs_create(world,N,N,1,1,N,N)
# Desc for A_nm : 1-D grid
desc1 = blacs_create(world,N,N,1,B,N,M)
# Desc for H_mm : 2-D grid
desc2 = blacs_create(world,N,N,D,D,mb,nb)

# Redistribute from 1-D -> 2-D grid
# in practice we must do this for performance
# reasons so this is not hypothetical
A_mm = scalapack_redist(A_nm,desc1,desc2)
S_mm = scalapack_redist(S_nm,desc1,desc2)
C_mm = scalapack_redist(C_nm,desc1,desc2)

W, Z_mm = scalapack_diagonalize_dc(A_mm, desc2)

Wg, Zg_mm = scalapack_general_diagonalize(A_mm, S_mm, desc2)

scalapack_inverse_cholesky(C_mm, desc2)

if  world.rank == 0:
    print "W =", w
    print "Wg =", wg
assert len(W) == len(w)

for i in range(len(W)):
    if abs(W[i]-w[i]) > w_tol:
        raise NameError('scalapack_diagonalize_dc: incorrect eigenvalues!')
        
assert len(Wg) == len(wg)

for i in range(len(W)):
    if abs(Wg[i]-wg[i]) > w_tol:
        raise NameError('scalapack_general_diagonalize: incorrect eigenvalues!')

# Check eigenvectors
# Easier to do this if everything if everything is collect on one node
Z_0 = scalapack_redist(Z_mm,desc2,desc0)
Zg_0 = scalapack_redist(Zg_mm,desc2,desc0)
C_0 = scalapack_redist(C_mm, desc2, desc0)

if world.rank == 0:
    Z_0 = Z_0.copy("C")
    Zg_0 = Zg_0.copy("C")
    C_0 = C_0.copy("C")
else:            
    Z_0 = np.zeros((N,N))
    Zg_0 = np.zeros((N,N))
    C_0 = np.zeros((N,N))
    
assert Z_0.shape == A2.shape == Zg_0.shape == A.shape == C_0.shape == C.shape

world.broadcast(Z_0, 0)
world.broadcast(Zg_0, 0)
world.broadcast(C_0, 0)

# Note that in general the an entire set of eigenvectors can differ by -1.
# For degenerate eigenvectors, this is even worse as they can be rotated
# by any angle as long as they space the Hilber space.
# 
# The indices on the matrices are swapped due to different orderings
# between C and Fortran arrays.
for i in range(Z_0.shape[0]):
    for j in range(Z_0.shape[1]):
        if abs(abs(Z_0[i,j])-abs(A2[j,i])) > z_tol:
            raise NameError('scalapack_diagonalize_dc: incorrect eigenvectors!')
        if abs(abs(Zg_0[i,j])-abs(A[j,i])) > z_tol:
            raise NameError('scalapack_general_diagonalize: incorrect eigenvectors!')        
        if abs(abs(C_0[i,j])-abs(C[j,i])) > c_tol:
            raise NameError('scalapack_inverse_cholesky: failed!')
