import Numeric as num

from gpaw.utilities import equal
from gpaw.utilities.lapack import sqrt_matrix

# check sqrt of a matrix

A = [[20, 4], [4, 1]]
a = [[4.4, .8], [.8, .6]]
A = num.array(A, num.Float)
print 'A=', A
a = num.array(a, num.Float)
b = sqrt_matrix(A)
print 'sqrt(A)=', b
equal(num.sum((a-b).flat), 0, 1.e-12)
