from math import sin, cos
import Numeric as num

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

def RotationCoef(N, dt):
    """Calculate coefficients for rotation.
        
    Returns the coefficients for the dt-rotation of an (N x N) matrix,
    employing a linear interpolation over the four nearest neighbours.
    RotationCoef guarantees symmetry between +dt and -dt rotations.
    The function returns the quantities c, val1, from1, to1, val2,
    from2, to2. c gives the length of the other vectors, and the 1
    and 2 means that the vectors correspond to a positive, respectively
    negative rotation.
    """

    coords=num.indices((N,N)).astype(num.Float)
    coords=num.reshape(coords, (2,N*N))
    trans=0.5*N
    coords-=trans
    
    # The coordinatematrix is rotated -dt, so that for a given point in the
    # rotated matrix, it is possible to determine from where on an unrotated
    # matrix it would have originated. The coordinates are then shifted back.
    row1=cos(dt)*coords[0]+sin(dt)*coords[1] + trans
    col1=-sin(dt)*coords[0]+cos(dt)*coords[1] + trans
    row2=cos(dt)*coords[0]-sin(dt)*coords[1] + trans
    col2=sin(dt)*coords[0]+cos(dt)*coords[1] + trans

    # Points that are judged to have originated from outside the unrotated
    # matrix are registered. There is a small correction to allow for
    # machine precision problems at the edges
    eps = 1e-14
    valids1 = num.logical_and(
	    num.logical_and(1-eps < row1, row1 < N-1+eps), 
	    num.logical_and(1-eps < col1, col1 < N-1+eps))
    valids2 = num.logical_and(
	    num.logical_and(1-eps < row2, row2 < N-1+eps), 
	    num.logical_and(1-eps < col2, col2 < N-1+eps))
    
    # Points are truncated. Points originating from outside the
    # matrix will be pushed to the edges, but their values will
    # be zero, due to the interpolation.
    row1 = num.clip(row1, 1, N-1-eps)
    col1 = num.clip(col1, 1, N-1-eps)
    row2 = num.clip(row2, 1, N-1-eps)
    col2 = num.clip(col2, 1, N-1-eps)

    rowindex1=num.floor(row1).astype(num.Int)
    colindex1=num.floor(col1).astype(num.Int)
    rowindex2=num.floor(row2).astype(num.Int)
    colindex2=num.floor(col2).astype(num.Int)
    
    row1-= rowindex1
    col1-= colindex1
    row2-= rowindex2
    col2-= colindex2

    # See "Ono and Hirose: Timesaving Double-Grid Method for Real-Space
    # Electronic-Structure Calculations".
    c11 = (1-col1)*(1-row1)*valids1 # x (row, col)
    c21 = col1*(1-row1)*valids1     # x (row, col+1)
    c31 = (1-col1)*row1*valids1     # x (row+1, col)
    c41 = col1*row1*valids1         # x (row+1, col+1)

    c12 = (1-col2)*(1-row2)*valids2 # x (row, col)
    c22 = col2*(1-row2)*valids2     # x (row, col+1)
    c32 = (1-col2)*row2*valids2     # x (row+1, col)
    c42 = col2*row2*valids2         # x (row+1, col+1)

    v1 = N*rowindex1+colindex1
    v2 = N*rowindex2+colindex2

    val1 = num.zeros(8*N*N).astype(num.Float)
    from1 = num.zeros(8*N*N)
    to1 = num.zeros(8*N*N)
    val2 = num.zeros(8*N*N).astype(num.Float)
    from2 = num.zeros(8*N*N)
    to2 = num.zeros(8*N*N)

    c = 0
    # The positive and negative rotation-matrices are determined.
    for p in range(N*N):
        v = v1[p]
        to1[c] = from2[c] = p
        from1[c] = to2[c] = v
        val1[c] = c11[p]
        c+=1
        to1[c] = from2[c] = p
        from1[c] = to2[c] = v+1
        val1[c] = c21[p]
        c+=1
        to1[c] = from2[c] = p
        from1[c] = to2[c] = v+N
        val1[c] = c31[p]
        c+=1
        to1[c] = from2[c] = p
        from1[c] = to2[c] = v+N+1
        val1[c] = c41[p]
        c+=1

        v = v2[p]
        to1[c] = from2[c] = v
        from1[c] = to2[c] = p
        val1[c] = c12[p]
        c+=1
        to1[c] = from2[c] = v+1
        from1[c] = to2[c] = p
        val1[c] = c22[p]
        c+=1
        to1[c] = from2[c] = v+N
        from1[c] = to2[c] = p
        val1[c] = c32[p]
        c+=1
        to1[c] = from2[c] = v+N+1
        from1[c] = to2[c] = p
        val1[c] = c42[p]
        c+=1
        
    val1 = 0.5*val1
    return c, val1, from1, to1, val1, from2, to2
