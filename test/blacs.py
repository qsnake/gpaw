# Simple test case for new BLACS/ScaLAPACK
# We compare the results here to those obtained
# from
# test/parallel/bandpar3.py
# with analogous parameters.
#
# Note that GPAW does not do transpose for calls
# to LAPACK involving operations on symmetric matrices. 
# Hence, UPLO = 'U' in Fortran equals UPLO = 'L' in
# NumPy C-style. To do a simple comparison we will make
# convert everything here to Fortran style.
import numpy as np
from gpaw.utilities.lapack import diagonalize, inverse_cholesky

N = 16;
A = np.empty((N,N))
A[:,0:4] = 0.0*np.eye(N,4,0)
A[:,0:4] = A[:,0:4]+ 0.1*np.eye(N,4,+1) # shift off of diag +1
A[:,4:8] = 1.0*np.eye(N,4,-4)
A[:,4:8] = A[:,4:8] + 0.1*np.eye(N,4,-3) # shift off of diag +1
A[:,8:12] = 2.0*np.eye(N,4,-8)
A[:,8:12] = A[:,8:12] + 0.1*np.eye(N,4,-7) # shift off of diag +1 
A[:,12:16] = 3.0*np.eye(N,4,-12)
A[:,12:16] = A[:,12:16]+ 0.1*np.eye(N,4,-11) # shift off of diag +1
A = A.copy("Fortran")
print "Hamiltonian =", A

B = np.empty((N,N))
B[:,0:4] = 1.0*np.eye(N,4,0)
B[:,0:4] = B[:,0:4]+ 0.2*np.eye(N,4,+1) # shift off of diag +1
B[:,4:8] = 1.0*np.eye(N,4,-4)
B[:,4:8] = B[:,4:8] + 0.2*np.eye(N,4,-3) # shift off of diag +1
B[:,8:12] = 1.0*np.eye(N,4,-8)
B[:,8:12] = B[:,8:12] + 0.2*np.eye(N,4,-7) # shift off of diag +1 
B[:,12:16] = 1.0*np.eye(N,4,-12)
B[:,12:16] = B[:,12:16] + 0.2*np.eye(N,4,-11) # shift off of diag +1
B = B.copy("Fortran")
print "Overlap = ", B

w = np.empty(N)

# We need to make a backup of A since LAPACK diagonalize will destroy it
A2 = A.copy("Fortran")
info = diagonalize(A2, w)

if info != 0:
    print "WARNING: diagonalize info=", info
print "lambda", w
print "eigenvectors", A2

# We need to make a backup of B since LAPACK diagonalize will destroy it
B2 = B.copy("Fortran")
info = diagonalize(A, w, B2)

if info != 0:
    print "WARNING: general diagonalize info = print", info
print "lambda", w
print "eigenvectors", A

info = inverse_cholesky(B)

if info != 0:
    print "WARNING: general diagonalize info = print", info
print "overlap", B
