from math import sin, cos
import Numeric as num

## def RotationCoef(N, dt):
##     v = num.arange(N*N-1, -1, -1)
##     c1 = num.ones((1,N*N))
##     c2 = num.zeros((1,N*N))
##     coefs=num.reshape(num.asarray([c2, c2, c2, c1]), (1,4*N*N)).astype(num.Float)
##     return [coefs, v]

def RotationCoef(N, dt):
    """Calculate coefficients for rotation.
        
    Returns the coefficients for the dt-rotation of an (N x N) matrix,
    employing a linear interpolation over the four nearest neighbours.
    The function returns the object [[c1, c2, c3, c4], v], where all elements
    are N^2-dimensional vectors. Given a matrix M reshaped as a vector,
    a rotation can be achieved as::
    
    M'[i] = c1[i]*M[v[i]] + c2[i]*M[v[i]+1] +
                c3[i]*M[v[i]+N] + c4[i]*M[v[i]+1+N]
    """
    coords=num.indices((N,N)).astype(num.Float)
    coords=num.reshape(coords, (2,N*N))
    trans=0.5*N
    coords-=trans
    
    # The coordinatematrix is rotated -dt, so that for a given point in the
    # rotated matrix, it is possible to determine from where on an unrotated
    # matrix it would have originated. The coordinates are then shifted back.
    row=cos(dt)*coords[0]+sin(dt)*coords[1] + trans
    col=-sin(dt)*coords[0]+cos(dt)*coords[1] + trans
    
    # A correction to ensure the x- and y-edges are included. The
    # edges may otherwise be omitted, due to the finite machine precision.
    # Points on the edges are moved an 'infinitessimal' step inwards.
    # Condition for validity is eps << 1, which should hold for any
    # sane choice of N.
    eps = 1e-14
    
    ##  valids = num.logical_and(
    ## 	    num.logical_and(1-eps < row, row < N-1 + eps), 
    ## 	    num.logical_and(1-eps < col, col < N-1 + eps))
    valids = num.logical_and(
	    num.logical_and(1-eps <= row, row <= N-1+eps), 
	    num.logical_and(1-eps <= col, col <= N-1+eps))
    # Points are truncated. Points originating from outside the
    # matrix will be pushed to the edges, but their values will
    # be zero, due to the interpolation.
    row = num.clip(row, 1, N-1)
    col = num.clip(col, 1, N-1)
    rowindex=num.floor(row).astype(num.Int)
    colindex=num.floor(col).astype(num.Int)
    row-= rowindex
    col-= colindex

    v = N*rowindex+colindex
    
    # See "Ono and Hirose: Timesaving Double-Grid Method for Real-Space
    # Electronic-Structure Calculations".
    c1 = (1-col)*(1-row)*valids # x (row, col)
    c2 = col*(1-row)*valids     # x (row, col+1)
    c3 = (1-col)*row*valids     # x (row+1, col)
    c4 = col*row*valids         # x (row+1, col+1)
    v = v*valids
    coefs = num.reshape(num.asarray([c1, c2, c3, c4]), (1,4*N*N))
    return [coefs, v]


def rotate(diff_c, r_c, angle):
    cs = cos(angle)
    sn = sin(angle)
    R_cc = num.array([(1,   0,  0),
		      (0,  cs, sn),
		      (0, -sn, cs)])
    diff_c -= r_c - num.dot(R_cc, r_c)


if __name__ == '__main__':
    from math import pi
    c, v = RotationCoef(4, -pi)
    c.shape = (4,4,4)
    for x in c:
        print x
    v.shape = (4, 4)
    print v
