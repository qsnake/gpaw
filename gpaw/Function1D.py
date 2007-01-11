from gaunt import gaunt
from math import pi, sqrt
from gpaw.utilities import hartree
import LinearAlgebra as linalg
import Numeric as num

# Small helper function to perform critical division with density
# I'm sure this could be improved, how?
def soft_divide(a,b):
    return a/(b+1e-100)
                  
# A class that encapsules a series of real spherical harmonic.
# Addition and multiplication operators are overloaded, so you
# can add and multiply these series with simple arithmetic operators.

class Function1D:

    # Init Function1D with only one spherical harmonic
    def __init__(self, *args):

        # Create empty dictionary for harmonic data
        self.harmonics = {}
        
        if len(args)==3:
            l,m,u = args
            # Store the function to dictionary
            self.harmonics[l,m] = u * sqrt(4*pi)
        elif len(args)==1:
            self.harmonics = args[0]

    # Adds a constant to the spherical harmonics
    def add_constant(self, c):
       if (self.harmonics.has_key(0,0)):
           self.harmonics[0,0] = self.harmonics[0,0] + c
       else:
           self.harmonics[0,0] = c

    # Divide every radial part with given radial part n
    def divide_all(self, n):
        for lm1, u1 in self.harmonics.iteritems():
            self.harmonics[lm1] = soft_divide(u1, n)

    # This does the same thing as divide_all, but spherically
    # averages n first.
    def __div__(self, x):
        temp = Function1D();
        temp.copyfrom(self);
        
        for lm2, u2 in x.harmonics.iteritems():
            l2,m2 = lm2
            # We don't know how to divide with l,m <> 0
            assert l2 == 0
            assert m2 == 0

        temp.divide_all(1/sqrt(4*pi)*x.harmonics[0,0])
        return temp

    def copyfrom(self, x):
        self.harmonics = {}
        for lm2, u2 in x.harmonics.iteritems():
            l2,m2 = lm2
            self.harmonics[lm2] = u2.copy()
            
    # Multiply every radial part with constant c
    def scalar_mul(self, c):
        for lm1, u1 in self.harmonics.iteritems():
            self.harmonics[lm1] *= c
        return self

    # The negation of function
    def __neg__(self):
        return self.scalar_mul(-1)

    # Integrate over angle
    # Only spherical harmonics of s-type will survive.
    def integrateY(self):
        if (self.harmonics.has_key((0,0))):
            return 1/sqrt(4*pi)*self.harmonics[(0,0)]
        else:
            return 0

    # Integrate both over angle and radial part
    def integrateRY(self, r, dr):
        if (self.harmonics.has_key((0,0))):
            # 4*pi comes from integration over angle
            # and 1/sqrt(4*pi) is the spherical harmonic for s-type functions
            return 1/sqrt(4*pi)*num.dot(r**2 * dr, self.harmonics[0,0]) 
        else:
            return 0

    # Return the poisson solution of this series as charge density
    def solve_poisson(self, r,dr,beta, N):

        temp_harmonics = {}
        
        for lm1, u1 in self.harmonics.iteritems():
            l1,m1 = lm1
            V = u1.copy()
            #print "lm",l1,m1
            #print u1
            #print beta
            #print N
            
            hartree(l1, u1 * r * dr, beta, N, V);   
            temp_harmonics[l1,m1] = V/(4*pi) /r

        return Function1D(temp_harmonics)

    # Add two series together
    def __add__(self, x):
        
        temp_harmonics = {}
        for lm1, u1 in self.harmonics.iteritems():
            l1,m1 = lm1
            temp_harmonics[l1, m1] = u1
        for lm2, u2 in x.harmonics.iteritems():
            l2,m2 = lm2
            if (temp_harmonics.has_key((l2,m2))):
                temp_harmonics[l2,m2] = temp_harmonics[l2,m2] + u2
            else:
                temp_harmonics[l2,m2] = u2.copy()

        return Function1D(temp_harmonics)

    # Multiply two series. Here we need gaunt's coefficients
    def __mul__(self, x):
       
        temp_harmonics = {}
        #print "Kerrotaan sarjat:"
        #print "1: ",self
        #print "2: ",x
        #print "-----------"
        for lm1, u1 in self.harmonics.iteritems():
            l1,m1 = lm1
            for lm2, u2 in x.harmonics.iteritems():
                l2,m2 = lm2
                for l3 in range(abs(l1-l2),l1+l2+1):
                    for m3 in range(-l3,l3+1):
                        #print l1,m1,l2,m2,l3,m3
                        G = gaunt[l1**2+(m1+l1), l2**2+(m2+l2), l3**2+(m3+l3)]
                        if (abs(G) >1e-15 ):
                            u_result = G*u1*u2;
                            if (l3>2):
                                pass
                                #print "Rejecting harmonic ", l3, m3
                            else:
                                if (temp_harmonics.has_key((l3,m3))):
                                    temp_harmonics[l3,m3] = temp_harmonics[l3,m3] + u_result
                                else:
                                    temp_harmonics[l3,m3] = u_result

        tulos = Function1D(temp_harmonics)
        
        return tulos

    def __str__(self):
        return "Function1D:" + self.harmonics.__str__()
