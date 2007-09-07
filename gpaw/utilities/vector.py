from math import sqrt
import Numeric as num

class Vector3d(list):
    def __init__(self,vector=None):
        if vector is None or vector == []:
            vector = [0,0,0]
        list.__init__(self)
##        print "init: vector=",vector
        for c in range(3):
            self.append(float(vector[c]))
        self.l = False

    def __mul__(self, x):
##        print "mul: x,self.v=",x,self
        if type(x) == type(self):
##            print "=> res=", num.dot( self, x )
            return num.dot( self, x )
        else:
##            print "=> res=", Vector3d( x*num.array(self) )
            return Vector3d( x*num.array(self) )
        
    def __rmul__(self, x):
        return self.__mul__(x)
        
    def __str__(self):
        return "(%g,%g,%g)" % tuple(self)

    def length(self,value=None):
        if value:
            fac = value / self.length()
            for c in range(3):
                self[c] *= fac
##            print "...self=",self
            self.l = False
##        print "....self,id=",self,id(self)
        if not self.l:
            self.l = sqrt(self.norm())
##        print ".....self=",self,self.l
        return self.l

    def norm(self):
        return num.sum( self*self )

    def vprod(self, a, b=None):
        """vector product"""
        if b is None:
            # [self x a]
            return Vector3d([self[1]*a[2]-self[2]*a[1],
                             self[2]*a[0]-self[0]*a[2],
                             self[0]*a[1]-self[1]*a[0]])
        else:
            # [a x b]
            return Vector3d([a[1]*b[2]-a[2]*b[1],
                             a[2]*b[0]-a[0]*b[2],
                             a[0]*b[1]-a[1]*b[0]])
                         
    def x(self):
        return self[0]

    def y(self):
        return self[1]

    def z(self):
        return self[2]
