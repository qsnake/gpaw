#!/usr/bin/env python
"""
Contains an implementation of the downhill simplex optimization method.
"""
import numpy as npy

reflect = -1.0
halfway = 0.5
extrapolate = 2.0

def standardfunction(x):
    """Default test function with one minimum at (1,2,3, ...).
    
    The minimum is exactly 42. Takes a list of coordinates as an argument
    and returns a number.
    """
    y = 42
    for i in range(len(x)):
        y += (x[i] - (i + 1))**2
    return y

class Logger:
    """
    This class can be used to encapsulate a function which is to be optimized.
    Invocations of that function will then be logged by this object during
    optimization.
    """
    def __init__(self, function, amoeba):
        self.innerfunction = function
        self.amoeba = amoeba
        self.x = []
        self.y = []
        self.dx = []
        self.dev = []
        self.vol = []
        self.center = 0

    def function(self, args):
        y = self.innerfunction(args)
        self.x.append(tuple(args))
        self.y.append(y)
        currentcenter = center(self.amoeba.simplex)
        dcenter = currentcenter - self.center
        self.dx.append(npy.sqrt(npy.dot(dcenter, dcenter)))        
        #self.c.append(center(self.amoeba.simplex))
        self.dev.append(self.amoeba.relativedeviation)
        self.vol.append(volume(self.amoeba.simplex))
        self.center = currentcenter
        return y

    def unzip(self):
        # equivalent to zip(*...)?
        return [[x[i] for x in self.x] for i in range(len(self.x[0]))]

    def numarray(self, transpose = False):
        arr = npy.array(self.x)
        if transpose:
            arr = npy.transpose(arr)
        return arr

class Amoeba:
    """Implementation of the downhill simplex minimization method.

    This works by evaluating function values on the vertices of a simplex,
    moving worse points in the general direction of better ones."""

    def __init__(self, simplex, values=None, function=standardfunction,
                 tolerance=0.001, savedata=False):
        y = values
        self.vertexcount = len(simplex)
        self.dimcount = self.vertexcount - 1

        self.simplex = simplex

        self.relativedeviation=0.

        if savedata:
            self.logger = Logger(function, self)
            function = self.logger.function

        if y is None:
            y = map(function, simplex)
        elif len(simplex) != len(y):
            raise Exception('Vertex count differs from value count')
        self.y = y

        self.evaluationcount = 0
        self.tolerance = tolerance
        self.function = function
        #This is probably the coordinate sum, i.e.
        #it probably has to do with the geometric center of the simplex
        self.coord_sums = [None]*self.dimcount
        self.calc_coord_sums()
        self.analyzepoints()

    def optimize(self):
        while self.step() > self.tolerance:
            pass

    def analyzepoints(self):
        simplex = self.simplex
        y = self.y
        vertexcount = self.vertexcount
        dimcount = self.dimcount
        
        ilow = 0 #index of lowest value
        ihigh = None #index of highest value
        i2ndhigh = None #index of second highest value
        if y[0] > y[1]:
            (ihigh, i2ndhigh) = (0, 1)
        else:
            (ihigh, i2ndhigh) = (1, 0)

        #Loop through vertices to find index values for highest/lowest entries
        for i in range(vertexcount):
            if y[i] < y[ilow]:
                ilow = i
            if y[i] > y[ihigh]:
                i2ndhigh = ihigh
                ihigh = i
            elif y[i] > y[i2ndhigh]:
                if i != ihigh:
                    i2ndhigh = i

        #Things should be floats already, but it's good to be safe
        self.ilow = ilow
        self.ihigh = ihigh
        self.i2ndhigh = i2ndhigh
        self.relativedeviation = float(abs(y[ihigh] - y[ilow]))/abs(y[ihigh]
                                                                    +y[ilow])

        return self.relativedeviation
    
    def calc_coord_sums(self):
        """
        Given a list of (dimcount+1) vectors each with dimcount
        coordinates, returns the list of coordinate sums across
        vectors, i.e. the n'th element is the sum of the n'th
        coordinates of all vectors in p
        """
        for i in range(self.dimcount):
            self.coord_sums[i] = sum([q[i] for q in self.simplex])

    def step(self):
        simplex = self.simplex
        y = self.y
        dimcount = self.dimcount
        vertexcount = self.vertexcount
        coord_sums = self.coord_sums
        function = self.function
        ihigh = self.ihigh
        i2ndhigh = self.i2ndhigh
        ilow = self.ilow
        
        ytest = self.grope(reflect)

        if ytest <= y[ilow]:
            ytest = self.grope(extrapolate)
        elif ytest >= y[i2ndhigh]:
            ysave = y[ihigh]
            ytest = self.grope(halfway)
            if ytest >= ysave:
                for i in range(vertexcount):
                    if i != ilow:
                        for j in range(dimcount):
                            coord_sums[j] = .5 * (simplex[i][j] +
                                                 simplex[ilow][j])
                            simplex[i][j] = coord_sums[j]
                        y[i] = function(coord_sums)
                self.evaluationcount += dimcount
                coord_sums = self.calc_coord_sums()

        return self.analyzepoints()
        
                            
    def grope(self, factor):
        """
        Extrapolates through or partway to simplex face, possibly
        finding a better vertex
        """

        y = self.y
        ihigh = self.ihigh
        dimcount = self.dimcount
        simplex = self.simplex
        coord_sums = self.coord_sums
        factor1 = (1. - factor)/dimcount
        factor2 = factor1 - factor

        ptest = [coord_sums[j]*factor1 - simplex[ihigh][j]*factor2
                  for j in range(dimcount)]

        ytest = self.function(ptest)

        if ytest < y[ihigh]:
            y[ihigh] = ytest
            for j in range(dimcount):
                coord_sums[j] += ptest[j] - simplex[ihigh][j]
                simplex[ihigh][j] = ptest[j]

        self.evaluationcount += 1
        
        return ytest

def center(simplex):
    vertices = [npy.array(point) for point in simplex]
    return sum(vertices)/len(simplex)

def volume(simplex):
    vertices = [npy.array(point) for point in simplex]
    differences = [vertices[i]-vertices[i+1] for i in range(len(vertices)-1)]
    return npy.linalg.det(npy.array(differences))
