import numpy as np
import scipy.sparse.linalg as sla

class ScipyLinearSolver:
    """Wrapper class for the linear solvers in scipy.sparse.linalg.

    Solve the linear system of equations

            A * x = b

    where A is a linear operator and b is the known rhs. The linear operator
    provided as argument in the ``solve`` method must have a ``shape``
    attribute (a tuble (M,N) where M and N give the size of the corresponding
    matrix) and a ``dtype`` attribute giving datatype of the matrix entries.
    
    """
    
    # Supported solvers
    solvers = {'cg': sla.cg,             # symmetric positive definite 
               'minres': sla.minres,     # symmetric indefinite
               'gmres': sla.gmres,       # non-symmetric
               'bicg': sla.bicg,         # non-symmetric
               'cgs': sla.cgs,           # similar to bicg
               'bicgstab': sla.bicgstab, # 
               'qmr': sla.qmr
               }
    # 'lgmres': sla.lgmres # scipy v. 0.8.0
    
    def __init__(self, method='cg', preconditioner=None, tolerance=1e-5,
                 max_iter=1000):
        """Initialize the linear solver.

        method: str
            One of the supported linear solvers in scipy.
        preconditioner: LinearOperator
            Instance of class ``LinearOperator`` from the
            ``scipy.sparse.linalg`` package.

        """

        if method not in ScipyLinearSolver.solvers:
            raise RuntimeError("Unsupported solver %s" % method)
                                   
        self.solver = ScipyLinearSolver.solvers[method]
        self.pc = preconditioner

        self.tolerance = tolerance
        self.max_iter = max_iter

        # Iteration counter
        self.i = None
        
    def solve(self, A, x_G, b_G):
        """Solve linear system Ax = b."""

        # Initialize iteration counter
        self.i = 0

        size = x_G.size
        shape = x_G.shape
        assert size == np.prod(shape)

        # Reshape arrays for scipy
        x_0 = x_G.ravel()
        b = b_G.ravel()
        print "We segfault here 1.", A.dtype, x_G.dtype, b_G.dtype                
        x, info = self.solver(A, b, x0=x_0, maxiter=self.max_iter, M=self.pc,
                              tol=self.tolerance, callback=self.iteration)
        print "We segfault here 2."        
        x_G[:] = x.reshape(shape)
        
        return self.i, info
            
    def iteration(self, x_i):
        """Passed as callback function to the scipy-routine."""

        self.i += 1
