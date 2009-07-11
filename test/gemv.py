#!/usr/bin/env python

import time
import numpy as np
from gpaw.utilities.blas import gemmdot, dotu, gemv

def getrand(shape, dtype):
    if type(shape) is int:
        shape = (shape,)
    nelements = np.prod(shape)
    if dtype == float:
        return np.random.normal(size=nelements).reshape(shape)
    elif dtype == complex:
        return (np.random.uniform(size=nelements) * np.exp(1j \
            * np.random.uniform(0,2*np.pi,size=nelements))).reshape(shape)
    else:
        raise ValueError('Unsupported dtype "%s".' % dtype)

Q = 1456
G = 5013
dtype = float

mem = 0
itemsize = np.nbytes[np.dtype(dtype)]
mem += Q*G*itemsize #n_qg
mem += G*itemsize #x_g
mem += Q*itemsize #nx_q

print 'Estimated memory: %8.5f MB' % (mem/1024**2.,)

n_qg = getrand((Q,G), dtype)
x_g = getrand(G, dtype)

# -------------------------------------------------------------------

print '\n%s\nnx_q calculations\n%s\n' % ('='*40, '='*40)
numflop = (dtype==float and 2 or 8)*Q*G
numreps = 100

# Reference value
nx0_q = np.dot(n_qg, x_g)

t = time.time()
for n in range(numreps):
    nx1_q = np.dot(n_qg, x_g)
t = time.time()-t
performance = numflop*numreps/t
print 'dot    : %8.5f s, %8.5f Mflops' % (t,performance/1024**2.)
assert np.abs(nx0_q-nx1_q).max()<5e-12
del nx1_q

t = time.time()
nx2_q = np.zeros(Q, dtype)
for n in range(numreps):
    gemmdot(n_qg, x_g, 1.0, 0.0, nx2_q)
t = time.time()-t
performance = numflop*numreps/t
print 'gemmdot: %8.5f s, %8.5f Mflops' % (t,performance/1024**2.)
assert np.abs(nx0_q-nx2_q).max()<5e-12
del nx2_q

t = time.time()
nx3_q = np.empty(Q, dtype)
for n in range(numreps):
    nx3_q.fill(0.0)
    gemv(1.0, n_qg, x_g, 0.0, nx3_q, 't')
t = time.time()-t
performance = numflop*numreps/t
print 'gemv   : %8.5f s, %8.5f Mflops' % (t,performance/1024**2.)
assert np.abs(nx0_q-nx3_q).max()<5e-12
del nx3_q

t = time.time()
nT_gq = n_qg.T.copy()
nx4_q = np.empty(Q, dtype)
for n in range(numreps):
    nx4_q.fill(0.0)
    gemv(1.0, nT_gq, x_g, 0.0, nx4_q, 'n')
t = time.time()-t
performance = numflop*numreps/t
print 'gemvT  : %8.5f s, %8.5f Mflops' % (t,performance/1024**2.)
assert np.abs(nx0_q-nx4_q).max()<5e-12
del nT_gq, nx4_q

