import numpy as npy

class Polynomial:
    """Polynomial p(x,y,z). 
    """
    
    def __init__(self, values, coords, order):
        """Construct a polynomial p(x,y,z)."""
        
        if (order < 0):
            raise "Error in Polynomial: Order of the polynomial is below zero."
        if (order > 2):
            raise "Error in Polynomial: Polynomials higher than quadratic (order =2) are not yet supported."
        
        self.order = order
        if order == 0:
            self.c = npy.zeros([1.0])
            self.c[0] = npy.sum(values) / len(values)
        elif order == 1:
            A = npy.zeros([len(coords), 4])
            b = npy.zeros([len(coords)])
            # c0 + c1 x + c2 y + c3 z = b
            for i in range(len(coords)):
                A[i][0] = 1
                A[i][1] = coords[i][0]
                A[i][2] = coords[i][1]
                A[i][3] = coords[i][2]
                b[i] = values[i]
            c = npy.linalg.lstsq(A, b)
            c = c[0]
            self.c = [c[0]]
            self.c += [[c[1], c[2], c[3]]]
        elif order == 2:
            A = npy.zeros([len(coords), 10])
            b = npy.zeros([len(coords)])
            # c0 + c1 x + c2 y + c3 z 
            #    + c4 x^2 + c5 y^2 + c6 z^2 
            #    + c7 x y + c8 x z + c9 y z = b
            for i in range(len(coords)):
                A[i][0] = 1.
                A[i][1] = coords[i][0]
                A[i][2] = coords[i][1]
                A[i][3] = coords[i][2]
                A[i][4] = coords[i][0] * coords[i][0]
                A[i][5] = coords[i][1] * coords[i][1]
                A[i][6] = coords[i][2] * coords[i][2]
                A[i][7] = coords[i][0] * coords[i][1]
                A[i][8] = coords[i][0] * coords[i][2]
                A[i][9] = coords[i][1] * coords[i][2]
                b[i] = values[i]
            c = npy.linalg.lstsq(A, b)
            #print c
            c = c[0]
            self.c = [c[0]]
            self.c += [[c[1], c[2], c[3]]]
            self.c += [[c[4], c[5], c[6], c[7], c[8], c[9]]]
            
    def coeff(self, i,j,k):
        # if coeff(0,0,0)
        if i+j+k == 0: 
            return self.c[0]
        # if order == 0, other zeros
        elif self.order == 0:
            return 0.0
        
        if i+j+k == 1:
            return self.c[1][i*0 + j*1 + k*2]
        elif self.order == 1:
            return 0.0
        
        if i+j+k == 2:
            # if i,j or k == 2
            if (i % 2) + (j % 2) + (k % 2) == 0:
                return self.c[2][(i/2)*0 + (j/2)*1 + (k/2)*2]
            else:
                return self.c[2][2 + i*0 + j*1 + k*2]
        elif self.order == 2:
            return 0.0
        
        raise "Error in Polynomial: Polynomials higher than quadratic (order =2) are not yet supported."

    def value(self, x, y, z):
        if self.order == 0:
            return self.c[0]
        elif self.order == 1:
            return self.c[0] + \
                self.c[1][0] * x + self.c[1][1] * y + self.c[1][2] * z
        elif self.order == 2:
            return self.c[0] \
                + self.c[1][0] * x \
                + self.c[1][1] * y \
                + self.c[1][2] * z \
                + self.c[2][0] * x**2 \
                + self.c[2][1] * y**2 \
                + self.c[2][2] * z**2 \
                + self.c[2][3] * x*y \
                + self.c[2][4] * x*z \
                + self.c[2][5] * y*z
        else:
            raise "Error in Polynomial: Polynomials higher than quadratic (order =2) are not yet supported."
