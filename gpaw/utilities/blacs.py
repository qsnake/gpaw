# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the C and Fortran packages:
Basic Linear Algebra Communication Subprogramcs (BLACS)
ScaLAPACK

See also:
http://www.netlib.org/blacs
and
http://www.netlib.org/scalapack
"""

from sys import stderr

import numpy as np

from gpaw.utilities import warning
from gpaw import debug
import gpaw.mpi as mpi
import _gpaw

def blacs_create(comm_obj, m, n, nprow, npcol, mb, nb, order='R'):
    sdf
    assert m > 0
    assert n > 0
    assert nprow > 0
    assert npcol > 0
    assert len(order) == 1
    assert order in 'CcRr'
    if comm_obj is not None:
        assert nprow * npcol <= comm_obj.size
    assert 0 < mb <= m
    assert 0 < nb <= n
    return _gpaw.blacs_create(comm_obj.get_c_object(),
                              m, n, nprow, npcol, mb, nb, order)

def blacs_destroy(adesc):
    assert len(adesc) == 9
    _gpaw.blacs_destroy(adesc)


def scalapack_redist1(a_obj, adesc, bdesc, isreal, comm_obj=mpi.world, m=0,
                      n=0):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert a_obj.dtype in [float, complex]
        if a_obj.dtype == float:
            assert isreal == True
        else:
            assert isreal == False
        if a_obj.flags.c_contiguous:
            print >> stderr, warning ('scalapack_redist: local matrices are '
                                      'not Fortran contiguous\n')
    assert len(adesc) == 9
    assert len(bdesc) == 9
    assert 0 <= m <= adesc[2]
    assert 0 <= n <= adesc[3]
    assert (bdesc[2] == m) or (bdesc[2] == adesc[2])
    assert (bdesc[3] == n) or (bdesc[3] == adesc[3])
    # There is no simple may to check if adesc and bdesc are disjoint to
    #comm_obj
    return _gpaw.scalapack_redist1(a_obj, adesc, bdesc, isreal,
                                   comm_obj.get_c_object(), m, n)


def scalapack_diagonalize_dc(a_obj, adesc, uplo):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    assert len(adesc) == 9
    assert uplo in ['U','L']
    return _gpaw.scalapack_diagonalize_dc(a_obj, adesc, uplo)


def scalapack_diagonalize_ex(adescriptor, a_obj, b_obj, c_obj, eps_obj, uplo):
    adesc = adescriptor.asarray()
    
    if adescriptor:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    #if b_obj is not None:
        assert b_obj.ndim == 2
        assert (b_obj.dtype == float) or (b_obj.dtype == complex)
        assert b_obj.flags.f_contiguous
    #if a_obj is None:
        #assert b_obj is None
    assert len(adesc) == 9
    assert uplo in ['U','L']
    if adescriptor.blacsgrid.is_active():
        _gpaw.scalapack_diagonalize_ex(a_obj, adesc, uplo, b_obj, c_obj,
                                       eps_obj)


def scalapack_inverse_cholesky(a_obj, adesc, uplo):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    assert len(adesc) == 9
    assert uplo in ['U','L']
    _gpaw.scalapack_inverse_cholesky(a_obj, adesc, uplo)


def pblas_pdgemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
                 transa='N', transb='N'):

    
    assert transa in ['N', 'T'] and transb in ['N', 'T']
    M, K = desca.gshape
    K, N = descb.gshape
    if transb == 'T':
        N, K = K, N
    assert transa == 'N' # XXX remember to implement 'T'

    _gpaw.pblas_pdgemm(N, M, K, alpha, b_KN.T, a_MK.T, beta, c_MN.T,
                       descb.asarray(), desca.asarray(), descc.asarray(),
                       transb, transa)


def pblas_simple_gemm(desca, descb, descc, a_MK, b_KN, c_MN, transa='N',
                      transb='N'):
    if transb == 'N':
        assert desca.check(a_MK)
        assert descb.check(b_KN)
        assert descc.check(c_MN)
        assert desca.gshape[1] == descb.gshape[0]
        assert desca.gshape[0] == descc.gshape[0]
        assert descb.gshape[1] == descc.gshape[1]
    # XXX also check for 'T'
    
    alpha = 1.0
    beta = 0.0
        
    pblas_pdgemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
                 transa, transb)


def pblas_pdgemv(alpha, a, adesc, x, xdesc, beta, y, ydesc):
    M, N = adesc.gshape
    assert M == ydesc.gshape[0]
    assert N == xdesc.gshape[0]
    assert adesc.check(a)
    assert xdesc.check(x)
    assert ydesc.check(y)
    assert xdesc.gshape[1] == ydesc.gshape[1]
    _gpaw.pblas_pdgemv(N, M, alpha,
                       a, adesc.asarray(),
                       x, xdesc.asarray(),
                       beta,
                       y, ydesc.asarray())


def pblas_simple_gemv(adesc, xdesc, ydesc, a, x, y):
    alpha = 1.0
    beta = 0.0
    pblas_pdgemv(alpha, a, adesc, x, xdesc, beta, y, ydesc)

    
#if not debug:
#    blacs_create = _gpaw.blacs_create
#    blacs_destroy = _gpaw.blacs_destroy
#    scalapack_redist1 = _gpaw.scalapack_redist1
#    scalapack_diagonalize_dc = _gpaw.scalapack_diagonalize_dc
#    scalapack_diagonalize_ex = _gpaw.scalapack_diagonalize_ex
#    scalapack_inverse_cholesky = _gpaw.scalapack_inverse_cholesky
