# Copyright (c) 2007 Lauri Lehtovaara

"""This module defines BiCGStab-class, which implements biconjugate
gradient stabilized method. Requires Numeric and BLAS."""

import Numeric as num

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc


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
    
    def __init__( self, gd, timer = None,
                  tolerance = 1e-15, max_iterations = 100, eps=1e-15 ):
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
        gd               grid descriptor for coarse (pseudowavefunction) grid
        timer            timer
        tolerance        tolerance for the norm of the residual ||b - A.x||^2
        max_iterations   maximum number of iterations
        eps              if abs(rho) or (omega) < eps, it's regarded as zero 
                         and the method breaks down
        ================ =====================================================

        """
        
        self.tol = tolerance
        self.max_iter = max_iterations
        if ( eps <= tolerance ):
            self.eps = eps
        else:
            raise RuntimeError("BiCGStab method got invalid tolerance (tol = %le < eps = %le)." % (tolerance,eps))

        self.iterations = -1
        
        self.gd = gd
        self.timer = timer
        

    def solve(self, A, x, b):
        """Solve a set of linear equations A.x = b.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        A           matrix A
        x           initial guess x_0 (on entry) and the result (on exit)
        b           right-hand side vector
        =========== ==========================================================

        """
        if self.timer is not None:
            self.timer.start('BiCGStab')

        # r_0 = b - A x_0
        r = self.gd.zeros(typecode=num.Complex)
        A.dot(-x,r)
        r += b
        
        q = self.gd.empty(typecode=num.Complex)
        q[:] = r
        p = self.gd.zeros(typecode=num.Complex)
        v = self.gd.zeros(typecode=num.Complex)
        t = self.gd.zeros(typecode=num.Complex)
        m = self.gd.zeros(typecode=num.Complex)
        alpha = 0.
        rhop  = 1.
        omega = 1.

        # Vector dot product, a^H b, where ^H is conjugate transpose
        def zdotc(x,y):
            return self.gd.comm.sum(dotc(x,y))
        # a x + y => y
        def zaxpy(a,x,y):
            axpy(a*(1+0J), x, y)

        # scale = square of the norm of b
        scale = abs( zdotc(b,b) )
        if scale < self.eps:
            scale = 1.0

        for i in range(self.max_iter):
            # rho_i-1 = q^H r_i-1
            rho = zdotc(q,r)

            # if i=1, p_i = r_i-1
            # else beta = (rho_i-1 / rho_i-2) (alpha_i-1 / omega_i-1)
            #      p_i = r_i-1 + b_i-1 (p_i-1 - omega_i-1 v_i-1)
            beta = (rho / rhop) * (alpha / omega)

            # if abs(beta) / scale < eps, then BiCGStab breaks down
            if ( (i > 0) and
                 ((abs(beta) / scale) < self.eps) ):
                raise RuntimeError("Biconjugate gradient stabilized method failed (abs(beta)=%le < eps = %le)." % (abs(rho),self.eps))
            
            
            # p = r + beta * (p - omega * v)
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            zaxpy(-omega, v, p)
            p *= beta
            p += r
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # v_i = A.(M^-1.p)
            A.solve_preconditioner(p,m)
            A.dot(m,v)
            # alpha_i = rho_i-1 / (q^H v_i)
            alpha = rho / zdotc(q,v)
            # s = r_i-1 - alpha_i v_i
            zaxpy(-alpha, v, r)
            # s is denoted by r
            
            # x_i = x_i-1 + alpha_i (M^-1.p_i) + omega_i (M^-1.s)
            # next line is x_i = x_i-1 + alpha (M^-1.p_i)
            zaxpy(alpha, m, x)
            
            # if ( |s|^2 < tol^2 ) done
            if ( (abs(zdotc(r,r)) / scale) < self.tol*self.tol ):
                break
            
            # t = A.(M^-1.s), M = 1
            A.solve_preconditioner(r,m)
            A.dot(m,t)
            # omega_i = t^H s / (t^H t) 
            omega = zdotc(t,r) / zdotc(t,t)
            
            # x_i = x_i-1 + alpha_i (M^-1.p_i) + omega_i (M^-1.s)
            # next line is x_i = ... + omega_i (M^-1.s)
            zaxpy(omega, m, x)
            # r_i = s - omega_i * t
            zaxpy(-omega, t, r)
            # s is no longer denoted by r
            
            # if ( |r|^2 < tol^2 ) done
            if ( (abs(zdotc(r,r)) / scale) < self.tol*self.tol ):
                break
            
            # if abs(omega) < eps, then BiCGStab breaks down
            if ( (abs(omega) / scale) < self.eps ):
                raise RuntimeError("Biconjugate gradient stabilized method failed (abs(omega)/scale=%le < eps = %le)." % (abs(omega) / scale,self.eps))
            
            rhop = rho
            
        # done
        self.iterations = i+1

        if self.timer is not None:
            self.timer.stop('BiCGStab')

        #print self.iterations
        return x

