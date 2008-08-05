from pylab import *
from numpy import *

V_s = 0.2
b = 3.0
a = 2.0

x = arange(-3, 3, 0.1)
y = arange(-3, 3, 0.1)
z = arange(-2, 2, 0.1)

V = empty((len(x),len(y),len(z)), float)
for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            V[i,j,k] = V_s*(exp(-cos(2*pi*x[i]/b)-cos(2*pi*y[j]/b)-2*a*z[k])
                            - 2*exp(-a*z[k]))

contourf(x, y, V[:,:,1], 10)
show()
