def RotationCoef(N, dt):
	"""Calculate coefficients for rotation.
	
	Returns the coefficients for the dt-rotation of an (N x N) matrix,
	employing a linear interpolation over the four nearest neighbours.
	The function returns the object [c1, c2, c3, c4, v], where all elements
	are N^2-dimensional vectors. Given a matrix M reshaped as a vector,
	a rotation can be achieved as::

	  M'[i] = c1[i]*M[v[i]] + c2[i]*M[v[i]+1] +
	          c3[i]*M[v[i]+N] + c4[i]*M[v[i]+1+N]
	"""
	import Numeric as num
	from math import sin,cos

        coords=num.indices((N,N)).astype(num.Float)
	coords=num.reshape(coords, (2,N*N))
	trans=0.5*(N-1)
	coords-=trans

	# The coordinatematrix is rotated -dt, so that for a given point in the
	# rotated matrix, it is possible to determine from where on an unrotated
	# matrix it would have originated.
        row=cos(dt)*coords[0]+sin(dt)*coords[1]
        col=-sin(dt)*coords[0]+cos(dt)*coords[1]

	#The point of origin is shifted to the top left corner of the matrix.
        row+=trans
        col+=trans

	# A correction to ensure the x- and y-edges are included. The
	# edges may be omitted, due to the finite machine precision.
	# Points on the edges are moved an 'infinitessimal' step inwards.
	# Condition for validity is eps << 1, which should hold for any
	# sane choice of N.
	eps = N*1e-14
	
	valids = num.logical_and(
		num.logical_and(-eps <= row, row < N-1 + eps), 
	        num.logical_and(-eps <= col, col < N-1 + eps))

	# Points are truncated. Points originating from outside the
	# matrix will be pushed to the edges, but their values will
	# be zero, due to the interpolation.
	row = num.clip(row, 0.0, N-1-eps)
	col = num.clip(col, 0.0, N-1-eps)
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

        return [c1, c2, c3, c4, v]
