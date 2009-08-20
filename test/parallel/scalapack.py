# Note that GPAW does not do transpose for calls
# to LAPACK involving operations on symmetric matrices. 
# Hence, UPLO = 'U' in Fortran equals UPLO = 'L' in
# NumPy C-style. For simplicity, we will
# convert everything here to Fortran style since
# this is the default for ScaLAPACK
#
# Here we compare ScaLAPACK results for a N-by-N matrix
# with those obtain with serial LAPACK
from time import time
import numpy as np

from gpaw import GPAW
from gpaw import debug
from gpaw.mpi import world
from gpaw.utilities.lapack import diagonalize, inverse_cholesky

from gpaw.utilities import scalapack
from gpaw.utilities.blacs import *

# We could possibly have a stricter criteria, but these are all
# the printed digits at least
w_tol = 1.e-8
z_tol = 1.e-8
c_tol = 1.e-8


N = 512
B = 4
M = N/B

# blacs grid dimensions DxD and non-even blocking factors just
# to make things more non-trivial.
D = 2 
nb = 64
mb = 64

assert world.size == B
assert world.size >= D*D

def test(complex_type):

    if complex_type:
        epsilon = 1.0j
    else:
        epsilon = 1.0

    if debug and world.rank == 0:
        print "epsilon =",  epsilon
        
    if complex_type:
        A = np.zeros((N,N),dtype=complex)
    else:
        A = np.zeros((N,N),dtype=float)
        
    A[:,0:M] = 0.0*np.eye(N,M,0)
    A[:,0:M] = A[:,0:M]+ 0.1*np.eye(N,M,-M*0+1)*epsilon # diag +1
    A[:,M:2*M] = 1.0*np.eye(N,M,-M)
    A[:,M:2*M] = A[:,M:2*M] + 0.1*np.eye(N,M,-M*1+1)*epsilon # diag +1
    A[:,2*M:3*M] = 2.0*np.eye(N,M,-M*2)
    A[:,2*M:3*M] = A[:,2*M:3*M] + 0.1*np.eye(N,M,-M*2+1)*epsilon # diag +1
    A[:,3*M:4*M] = 3.0*np.eye(N,M,-M*3)
    A[:,3*M:4*M] = A[:,3*M:4*M]+ 0.1*np.eye(N,M,-M*3+1)*epsilon # diag +1
    if debug and world.rank == 0:
        print "A = ", A
        
    # We should really use Fortran ordered array but this gives
    # a false positive in LAPACK's debug mode
    # A = A.copy("Fortran")
    A = A.transpose().copy()

    S = np.zeros_like(A)
        
    S[:,0:M] = 1.0*np.eye(N,M,0)
    S[:,0:M] = S[:,0:M]+ 0.2*np.eye(N,M,-M*0+1)*epsilon # diag +1
    S[:,M:2*M] = 1.0*np.eye(N,M,-M)
    S[:,M:2*M] = S[:,M:2*M] + 0.2*np.eye(N,M,-M*1+1)*epsilon # diag +1
    S[:,2*M:3*M] = 1.0*np.eye(N,M,-M*2)
    S[:,2*M:3*M] = S[:,2*M:3*M] + 0.2*np.eye(N,M,-M*2+1)*epsilon # diag +1 
    S[:,3*M:4*M] = 1.0*np.eye(N,M,-M*3)
    S[:,3*M:4*M] = S[:,3*M:4*M] + 0.2*np.eye(N,M,-M*3+1)*epsilon # diag +1
    if debug and world.rank == 0:
        print "S = ", S
        
    # We should really use Fortran ordered array but this gives
    # a false positive in LAPACK's debug mode
    # S = S.copy("Fortran")
    S = S.transpose().copy()

    w = np.zeros(N,dtype=float)
    # We need to make a backup of A since LAPACK diagonalize will overwrite
    Ag = A.copy()
    info = diagonalize(A, w)

    if world.rank == 0:
        if info != 0:
            print "WARNING: diag info = ", info

    wg = np.zeros_like(w)

    # We need to make a backup of S since LAPACK diagonalize will overwrite
    C = S.copy()
    info = diagonalize(Ag, wg, S)

    if world.rank == 0:
        if info != 0:
            print "WARNING: general diag info = ", info

    # We are done, so it's ok to overwrite C
    info = inverse_cholesky(C)

    if world.rank == 0:
        if info != 0:
            print "WARNING: cholesky info = ", info

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
    A_nm = A_nm[:,0:M] + 0.1*np.eye(N,M,-M*world.rank+1)*epsilon
    A_nm = A_nm.copy("Fortran") # Fortran order required for ScaLAPACK
    S_nm = np.eye(N,M,-M*world.rank)
    S_nm = S_nm[:,0:M] + 0.2*np.eye(N,M,-M*world.rank+1)*epsilon
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
    A_mm = scalapack_redist(A_nm, desc1, desc2, world, 0, 0)
    Ag_mm = A_mm.copy("Fortran") # A_mm will be destroy upon call to
                                  # scalapack_diagonalize_dc 
    S_mm = scalapack_redist(S_nm, desc1, desc2, world, 0, 0)
    C_mm = scalapack_redist(C_nm, desc1, desc2, world, 0, 0)

    if debug:
        print "A_mm = ", A_mm
        print "S_mm = ", S_mm
    
    W, Z_mm = scalapack_diagonalize_dc(A_mm, desc2)
    Wg, Zg_mm = scalapack_general_diagonalize(Ag_mm, S_mm, desc2)
    scalapack_inverse_cholesky(C_mm, desc2)

    # Check eigenvalues and eigenvectors
    # Easier to do this if everything if everything is collected on one node
    Z_0 = scalapack_redist(Z_mm, desc2, desc0, world, 0, 0)
    Zg_0 = scalapack_redist(Zg_mm, desc2, desc0, world, 0, 0)
    C_0 = scalapack_redist(C_mm, desc2, desc0, world, 0, 0)

    if world.rank == 0:
        Z_0 = Z_0.copy("C")
        Zg_0 = Zg_0.copy("C")
        C_0 = C_0.copy("C")
    else:            
        Z_0 = np.zeros_like(A)
        Zg_0 = np.zeros_like(A)
        C_0 = np.zeros_like(A)

    assert Z_0.shape == Zg_0.shape == C_0.shape
    assert Z_0.shape == A.shape
    assert Zg_0.shape == Ag.shape
    assert C_0.shape == C.shape

    world.broadcast(Z_0, 0)
    world.broadcast(Zg_0, 0)
    world.broadcast(C_0, 0)

    if debug:
        print "W = ", W
        print "diag: lambda = ", w

        print "Z_0 ", Z_0
        print "diag: eigenvectors = ", A

        print "Wg = ", Wg
        print "general diag =" , wg

        print "Zg_0", Zg_0
        print "general diag: eigenvectors = ", Ag
    
    for i in range(len(W)):
        if abs(W[i]-w[i]) > w_tol:
            raise NameError('sca_diag_dc: incorrect eigenvalues!')
        
    assert len(Wg) == len(wg)

    for i in range(len(W)):
        if abs(Wg[i]-wg[i]) > w_tol:
            print "i=", i
            raise NameError('sca_general_diag: incorrect eigenvalues!')

    # Note that in general the an entire set of eigenvectors can
    # differ by -1. For degenerate eigenvectors, this is even
    # worse as they can be rotated by any angle as long as they
    # span the Hilbert space.
    #
    # The indices on the matrices are swapped due to different orderings
    # between C and Fortran arrays.
    for i in range(Z_0.shape[0]):
        for j in range(Z_0.shape[1]):
            if abs(abs(Z_0[i,j])-abs(A[j,i])) > z_tol:
                print "i, j, Z_0, A", i, j, Z_0[i,j], A[j,i]
                raise NameError('sca_diag_dc: incorrect eigenvectors!')
            if abs(abs(Zg_0[i,j])-abs(Ag[j,i])) > z_tol:
                print "i, j, Zg_0, Ag", i, j, Zg_0[i,j], Ag[j,i]
                raise NameError('sca_general_diag: incorrect eigenvectors!')   
            if abs(abs(C_0[i,j])-abs(C[j,i])) > c_tol:
                print "i, j, C_0, C", i, j, C_0[i,j], C[j,i]
                raise NameError('sca_inverse_cholesky: failed!')

    if world.rank == 0:
        if complex_type:
            print "complex type verified!"
        else:
            print "double type verified!"

if not scalapack():
    print('Not built with ScaLAPACK. Test does not apply.')
else:
    ta = time()
    for x in range(20):
        # Test real scalapack
        test(False)
        # Test complex scalapack
        test(True)
    tb = time()

    if world.rank == 0:
        print 'Total Time %f' % (tb - ta)
    
