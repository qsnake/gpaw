# Copyright (c) 2007 Lauri Lehtovaara

"""This module defines BiCGStab-class, which implements biconjugate
gradient stabilized method. Requires Numeric and BLAS."""

import Numeric as num
import BasicLinearAlgebra

class BiCGStab:
    """ Biconjugate gradient stabilized method
    
    This class solves a set of linear equations A.x = b using biconjugate 
    gradient stabilized method (BiCGStab). The matrix A is a general, 
    non-singular matrix, e.g., it can be nonsymmetric, complex, and 
    indefinite. The method requires only access to matrix-vector product 
    A.x = b, which is called A.dot(x). Thus A must provide the member 
    function dot(self,x,b), where x and b are complex arrays 
    (Numeric.array([],Numeric.Complex), and x is the known vector, and 
    b is the result.
    """ 
    
    def __init__(self, tolerance = 1e-15, max_iterations = 100, eps=1e-15):
        """Create the BiCGStab-object.
        
        Tolerance should not be smaller than attainable accuracy, which is 
        order of kappa(A) * eps, where kappa(A) is the (spectral) condition 
        number of the matrix. The maximum number of iterations should be 
        significantly less than matrix size, approximately 
        .5 sqrt(kappa) ln(2/tolerance). A small number is treated as zero
        if it's magnitude is smaller than argument eps.
        
        ================ =====================================================
        Parameters:
        ================ =====================================================
        tolerance        tolerance for the norm of the residual ||b - A.x||^2
        max_iterations   maximum number of iterations
        eps              if abs(rho) or (omega) < eps, it's regarded as zero 
                         and the method breaks down
        ================ =====================================================

        """
        
        self.tol = tolerance
        self.max_iter = max_iterations
        if ( eps < tolerance ):
            self.eps = eps
        else:
            self.eps = tolerance
        self.iterations = -1
        
        
    def solve(self, A, x, b, debug=0):
        """Solve a set of linear equations A.x = b.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        A           matrix A
        x           initial guess x_0 (on entry) and the result (on exit)
        b           right-hand side vector
        =========== ==========================================================

        """
        blas = BasicLinearAlgebra.BLAS()
        
        # r_0 = b - A x_0
        r = num.zeros(x.shape, num.Complex)
        A.dot(-x,r)
        r += b
        
        q = num.array(r, num.Complex)
        p = num.zeros(r.shape, num.Complex)
        v = num.zeros(r.shape, num.Complex)
        t = num.zeros(r.shape, num.Complex)
        alpha = 0.
        rhop  = 1.
        omega = 1.
        
        for i in range(self.max_iter):
            if (debug): print '--- iteration ', i+1, ' ---' 
            
            # rho_i-1 = q^H r_i-1
            rho = blas.zdotc(q,r)
            if (debug): print 'rho = ', rho

            # if abs(rho) < eps, then BiCGStab breaks down
            if ( abs(rho) < self.eps ):
                raise Exception("Biconjugate gradient stabilized method failed (abs(rho)=%le < eps = %le)." % (abs(rho),self.eps))
            
            # if i=1, p_i = r_i-1
            # else beta = (rho_i-1 / rho_i-2) (alpha_i-1 / omega_i-1)
            #      p_i = r_i-1 + b_i-1 (p_i-1 - omega_i-1 v_i-1)
            beta = (rho / rhop) * (alpha / omega)
            
            # p = r + beta * (p - omega * v)
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            blas.zaxpy(-omega, v, p)
            p *= beta
            p += r
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            if (debug): print 'beta = ', beta
            if (debug==2): print 'p = ', p
            
            # v_i = A.(M^-1.p), M = 1
            A.dot(p,v)
            # alpha_i = rho_i-1 / (q^H v_i)
            alpha = rho / blas.zdotc(q,v)
            # s = r_i-1 - alpha_i v_i
            blas.zaxpy(-alpha, v, r)
            # s is denoted by r
            
            if (debug): print 'alpha = ', alpha
            if (debug==2): print 'v = ', v
            if (debug==2): print 's = ', r
            
            if (debug): print '|s|^2 = ', blas.zdotc(r,r)
            
            # x_i = x_i-1 + alpha_i (M^-1.p_i) + omega_i (M^-1.s)
            # next line is x_i = x_i-1 + alpha (M^-1.p_i)
            blas.zaxpy(alpha, p, x)
            
            # if ( |s|^2 < tol^2 ) done
            if ( abs(blas.zdotc(r,r)) < self.tol*self.tol ):
                break
            
            # t = A.(M^-1.s), M = 1
            A.dot(r,t)
            # omega_i = t^H s / (t^H t) 
            omega = blas.zdotc(t,r) / blas.zdotc(t,t)
            
            if (debug==2): print 't = ', t
            if (debug): print 'omega = ', omega
            
            # x_i = x_i-1 + alpha_i (M^-1.p_i) + omega_i (M^-1.s)
            # next line is x_i = ... + omega_i (M^-1.s)
            blas.zaxpy( omega, r, x )
            # r_i = s - omega_i * t
            blas.zaxpy( -omega, t, r )
            # s is no longer denoted by r
            
            if (debug==2): print 'x = ', x
            if (debug==2): print 'r = ', r
            
            # if ( |r|^2 < tol^2 ) done
            if ( abs(blas.zdotc(r,r)) < self.tol*self.tol ):
                break
            
            # if abs(omega) < eps, then BiCGStab breaks down
            if ( abs(omega) < self.eps ):
                raise Exception("Biconjugate gradient stabilized method failed (abs(omega)=%le < eps = %le)." % (abs(omega),self.eps))
            
            rhop = rho
            
        # done
        self.iterations = i+1

        return x
        
