#!/usr/bin/env python

import optimizer

#N_MAX = 100
ALPHA = 1.
BETA = .5
GAMMA = 2.

"""
Default test function with one minimum at (1,2,3, ...).
The minimum is exactly 42. Takes a list of coordinates as an argument
and returns a number.
"""
def standardfunction(p):
    y = 42
    for i in range(len(p)):
        y += (p[i]-(i+1))**2
    return y

"""
Performs the 'amoeba'-like downhill simplex method in ndim dimensions.

p: a list of (ndim+1) vectors each with ndim coordinates,
corresponding to the vertices of the simplex.

y: a list of function values evaluated at the vertices, ordered consistently with the vertices in p. y thus must have length (ndim+1) as well

ndim: the dimension count of the space in question. Of course this variable is mostly for show since it's not really necessary in python

tolerance: fractional tolerance used to evaluate convergence criterion

function: the function to be minimized. The function must take exactly
ndim parameters, each parameter being one number

maxIterations: the maximal number of iterations to be performed before
returning, in case convergence is slow

Returns the number of times the function has been evaluated during the
procedure.

After invocation the argument lists p and y will have been modified to contain
the simplex vertices and associated function values at termination of the
procedure.

"""
class Amoeba:

    def __init__(self, simplex, values=None, function=standardfunction,tolerance=0.001):
        p = simplex
        y = values
        self.mpts = len(p)
        self.ndim = self.mpts - 1

        self.p = p

        if y is None:
            y = map(function, p)
        elif len(p) != len(y):
            raise Exception('Vertex count differs from value count')
        self.y = y

        self.evaluationcount = 0
        self.tolerance = tolerance
        self.function = function
        #This is probably the coordinate sum, i.e.
        #it probably has to do with the geometric center of the simplex
        self.psum = [None]*self.ndim
        self.getpsum()
        self.analyzepoints()
        

    def optimize(self):
        while self.step() > self.tolerance:
            pass

    def analyzepoints(self):
        p = self.p
        y = self.y
        mpts = self.mpts
        ndim = self.ndim
        
        #hostObject.callBackFunction(p,y,evaluationcount)
        #print >> out, 'Points:', p
        #print >> out, 'yValues:', y
        #print >> out, 'EvalCount:',evaluationcount
        #print >> out
        #out.flush()

        #Write current points to file for recovery if something goes wrong
        #pickleDump((p,y),dump)



        
        
        iLow = 0 #index of lowest value
        iHigh = None #index of highest value
        i2ndHigh = None #index of second highest value
        if y[0] > y[1]:
            (iHigh, i2ndHigh) = (0, 1)
        else:
            (iHigh, i2ndHigh) = (1, 0)

        #Loop through vertices to find index values for highest/lowest entries
        for i in range(mpts):
            if y[i] < y[iLow]:
                iLow = i
            if y[i] > y[iHigh]:
                i2ndHigh = iHigh
                iHigh = i
            elif y[i] > y[i2ndHigh]:
                if i != iHigh:
                    i2ndHigh = i

        #Things should be floats already, but it's good to be safe
        self.iLow = iLow
        self.iHigh = iHigh
        self.i2ndHigh = i2ndHigh
        self.relativedeviation = float(abs(y[iHigh] - y[iLow]))/abs(y[iHigh]+y[iLow])

        return self.relativedeviation
    
    def getpsum(self):
        """
        Given a list of (ndim+1) vectors each with ndim coordinates,
        returns the list of coordinate sums across vectors,
        i.e. the n'th element is the sum of the n'th coordinates of all vectors in p
        """

        for i in range(self.ndim):
            x = sum([q[i] for q in self.p])
            self.psum[i] = x

        return# psum

    def step(self):
        p = self.p
        y = self.y
        ndim = self.ndim
        mpts = self.mpts
        psum = self.psum
        function = self.function
        iHigh = self.iHigh
        i2ndHigh = self.i2ndHigh
        iLow = self.iLow
        
        yTry = self.amotry(p, y, psum, ndim, function, iHigh, -ALPHA)
        self.evaluationcount += 1

        if yTry <= y[iLow]:
            yTry = self.amotry(p, y, psum, ndim, function, iHigh, GAMMA)
            self.evaluationcount += 1
        elif yTry >= y[i2ndHigh]:
            ySave = y[iHigh]
            yTry = self.amotry(p, y, psum, ndim, function, iHigh, BETA)
            self.evaluationcount += 1
            if yTry >= ySave:
                for i in range(mpts):
                    if i != iLow:
                        for j in range(ndim):
                            psum[j] = .5 * (p[i][j] + p[iLow][j])
                            p[i][j] = psum[j]
                        y[i] = function(psum)
                self.evaluationcount += ndim
                psum = self.getpsum()

        return self.analyzepoints()
        
                            
    """
    Extrapolates through or partway to simplex face, possibly finding a better
    vertex
    """
    def amotry(self, p, y, psum, ndim, function, iHigh, factor):
        #Wonder what these 'factors' do exactly
        factor1 = (1. - factor)/ndim
        factor2 = factor1 - factor

        pTry = [psum[j]*factor1 - p[iHigh][j]*factor2 for j in range(ndim)]

        yTry = function(pTry)

        if yTry < y[iHigh]:
            y[iHigh] = yTry
            for j in range(ndim):
                psum[j] += pTry[j] - p[iHigh][j]
                p[iHigh][j] = pTry[j]

        return yTry

def main():
    p = optimizer.get_random_simplex([4,2,1,5,2,1,5,3,1,3])
    amoeba = Amoeba(p, tolerance=0.000001)
    
    
    amoeba.optimize()
    print amoeba.p
    print amoeba.y

if __name__ == '__main__':
     main()
