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
    assert m > 0
    assert n > 0
    assert nprow > 0 
    assert npcol > 0
    assert len(order) == 1
    assert order in ['C','c','R','r']
    if comm_obj is not None:
        assert nprow*npcol <= comm_obj.size
    assert 0 < mb <= m
    assert 0 < nb <= n
    return _gpaw.blacs_create(comm_obj, m, n, nprow, npcol, mb, nb, order)

def blacs_destroy(adesc):
    assert len(adesc) == 9
    _gpaw.blacs_destroy(adesc)

def scalapack_redist(a_obj, adesc, bdesc, isreal=True, comm_obj=mpi.world, m=0, n=0):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert a_obj.dtype in [float, complex]
        if a_obj.dtype == float:
            assert isreal == True
        else:
            assert isreal == False
        if a_obj.flags.c_contiguous:
            print >> stderr, warning ('scalapack_redist: local matrices are not Fortran contiguous\n')
    assert len(adesc) == 9
    assert len(bdesc) == 9
    assert 0 <= m <= adesc[2]
    assert 0 <= n <= adesc[3]
    assert (bdesc[2] == m) or (bdesc[2] == adesc[2])
    assert (bdesc[3] == n) or (bdesc[3] == adesc[3])
    # There is no simple may to check if adesc and bdesc are disjoint to comm_obj
    return _gpaw.scalapack_redist(a_obj, adesc, bdesc, isreal, comm_obj, m, n)

def scalapack_diagonalize_dc(a_obj, adesc):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    assert len(adesc) == 9
    return _gpaw.scalapack_diagonalize_dc(a_obj, adesc)

def scalapack_general_diagonalize(a_obj, b_obj, adesc):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    if b_obj is not None:
        assert b_obj.ndim == 2
        assert (b_obj.dtype == float) or (b_obj.dtype == complex)
        assert b_obj.flags.f_contiguous
    if a_obj is None:
        assert b_obj is None
    assert len(adesc) == 9
    return _gpaw.scalapack_general_diagonalize(a_obj, b_obj, adesc)

def scalapack_inverse_cholesky(a_obj, adesc):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    assert len(adesc) == 9
    _gpaw.scalapack_inverse_cholesky(a_obj, adesc)
    
if not debug:
    blacs_create = _gpaw.blacs_create
    blacs_destroy = _gpaw.blacs_destroy
    scalapack_redist = _gpaw.scalapack_redist
    scalapack_diagonalize = _gpaw.scalapack_diagonalize_dc
    scalapack_general_diagonalize = _gpaw.scalapack_general_diagonalize
    scalapack_inverse_cholesky = _gpaw.scalapack_inverse_cholesky
