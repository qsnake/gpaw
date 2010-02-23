import numpy as np

class SternheimerOperator:
    """Class implementing the linear operator in the Sternheimer equation.

    Sternheimer equation:
    
         (H - eps_n) P_c |\delta\Psi_n> = - P_c * \deltaV_KS * |\Psi_n>

    where P_c is the projection operator onto the unoccupied states.
    
    The main purpose of this class is to provide a method ("dot") that can
    evaluate the multiplication with a an arbitrary vector. This functionality
    is required by iterative Krylov solvers.
    
    """
    
    def __init__(self, hamiltonian, wfs, gd):
        """Init method."""

        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.gd = gd

        # Variables for k-point and band index
        self.k = None
        self.n = None

        # For scipy's linear solver
        N = np.prod(gd.n_c)
        self.shape = (N,N)
        self.dtype = float
        # LinearOperator.__init__(self, shape, self.matvec, dtype=float)
        
    def set_blochstate(self, k, n):
        """Set k-vector and band index for the Bloch-state in consideration.

        Parameters
        ----------
        k: int
           k-point index
        n: int
           Band index

        """
        
        self.k = k
        self.n = n
        
    def dot(self, x_G, y_G):
        """Multiplication with a vector.

        Evaluates the matrix-vector multiplication

                y_G = L * x_G

        where L is the linear operator appearing in the Sternheimer equation.
                
        Parameters
        ----------
        x_G: ndarray
            Representation of a vector defined on a 3d-grid
        y_G: ndarray
            Representation of a vector defined on a 3d-grid
            
        """
        
        assert self.k is not None
        assert self.n is not None

        kpt = self.wfs.kpt_u[self.k]
        
        self.hamiltonian.apply(x_G, y_G, self.wfs, kpt,
                               calculate_P_ani=False)

        y_G -= kpt.eps_n[self.n] * x_G


    def matvec(self, x):
        """Matrix vector multiplication for scipy.sparse.linalg cgs solver.

        Parameters
        ----------
        x: ndarry
            1-dimensional array holding the grid representation of the vector

        """

        assert self.k is not None
        assert self.n is not None

        size = x.size
        
        # Output array
        y_G = self.gd.zeros()
        shape = y_G.shape

        assert size == np.prod(shape)
        
        x_G = x.reshape(shape)
        
        kpt = self.wfs.kpt_u[self.k]
        
        self.hamiltonian.apply(x_G, y_G, self.wfs, kpt,
                               calculate_P_ani=False)

        y_G -= kpt.eps_n[self.n] * x_G

        y = y_G.ravel()
        
        return y
    
    def project(self, x_G):
        """Project the vector onto the unoccupied states at k+q.

        Implementation for q != 0 to be done !


               --                    --             
          P  = >  |Psi ><Psi | = 1 - >  |Psi ><Psi |
           c   --     c     c        --     v     v 
                c                     v             
        """

        nvalence = self.wfs.nvalence
        # k+q-vector
        kpt = self.wfs.kpt_u[self.k]
        psit_nG = kpt.psit_nG
            
        for n in range(nvalence/2):

            psit_G = psit_nG[n]
            
            proj_n = self.gd.integrate(x_G * psit_G)
            x_G -= proj_n * psit_G
        
    def norm(self, x_G):
        """L2-norm."""

        return self.gd.integrate(x_G**2)
