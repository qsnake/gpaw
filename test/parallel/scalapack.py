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

w_tol = 1.e-12
z_tol = 1.e-7
c_tol = 1.e-7


N = 16
B = 4
M = N/B
# blacs grid dimensions
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
print "Hamiltonian =", A
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
print "Overlap = ", S
# We should really use Fortran ordered array but this gives
# a false positive in LAPACK's debug mode
# S = S.copy("Fortran")
S = S.transpose().copy()
w = np.empty(N)

# We need to make a backup of A since LAPACK diagonalize will destroy it
A2 = A.copy()
info = diagonalize(A2, w)

if info != 0:
    print "WARNING: diagonalize info = ", info
print "lambda", w
print "eigenvectors", A2

wg = np.empty(N)

# We need to make a backup of S since LAPACK diagonalize will destroy it
S2 = S.copy()
info = diagonalize(A, wg, S2)

if info != 0:
    print "WARNING: general diagonalize info = ", info
print "lambda", wg
print "eigenvectors", A

# For consistency, also make a backup of S since LAPACK will destroy it
C = S.copy()
info = inverse_cholesky(C)

if info != 0:
    print "WARNING: general diagonalize info = ", info
print "overlap", C

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

W, Z_mm = scalapack_diagonalize_dc(A_mm, desc2)

Wg, Zg_mm = scalapack_general_diagonalize(A_mm, S_mm, desc2)

scalapack_inverse_cholesky(C_mm, desc2)

assert len(W) == len(w)

for i in range(len(W)):
    if abs(W[i]-w[i]) > w_tol:
        raise NameError('scalapack_diagonalize_dc eigenvalues wrong!')
        


assert len(Wg) == len(wg)

for i in range(len(W)):
    if abs(Wg[i]-wg[i]) > w_tol:
        raise NameError('scalapack_general_diagonalize eigenvalues wrong!')

# Check eigenvectors
# Easier to do this if everything if everything is collect on one node
Z_0 = scalapack_redist(Z_mm,desc2,desc0)
Zg_0 = scalapack_redist(Z_mm,desc2,desc0)
C_0 = scalapack_redist(C_00, desc2, desc0)

assert Z_0.shape == A2.shape == Zg_0.shape == A.shape == C_0.shape == C.shape

# We compare Fortran and C NumPy arrays, but this is not a problem here
# because NumPy does all the hardwork for us.
for i in Z_0.shape[0]:
    for j in Z_0.shape[0]:
        if abs(Z_0[i,j]-A2[i,j]) > z_tol:
            raise NameError('scalapack_diagonalize_dc eigenvectors failed!')
        if abs(Zg_0[i,j]-A[i,j]) > z_tol:
            raise NameError('scalapack_diagonalize_dc eigenvectors failed!')        
        if abs(C_0[i,j]-C[i,j]) > c_tol:
            raise NameError('scalapack_inverse_cholesky failed!')
