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

from gpaw.utilities import is_contiguous, warning
from gpaw import debug
import gpaw.mpi as mpi
import _gpaw

def blacs_create(comm_obj=None, m, n, nprow, npcol, mb, nb, row_order='R'):
    assert m > 0
    assert n > 0
    assert nprow > 0
    assert npcol > 0
    assert row_order[0] in ['C','c','R','r']
    if comm_obj is not None:
        assert nprow*npcol <= comm_obj.size
    assert m >= mb > 0
    assert n >= nb > 0
    _gpaw.blacs_create(comb_obj, m, n, nprow, npcol, mb, nb, row_order)

def blacs_destroy(adesc):
    assert len(adesc) == 9
    _gpaw.blacs_destroy(adesc)

def scalapack_redist(a_obj, adesc, bdesc, comm_obj=mpi.world, m=0, n=0):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        if a_obj.flags.c_contiguous:
            print >> stderr, warning ('scalapack_redist: local matrices are not Fortran contiguous\n')
    assert len(adesc) == 9
    assert len(bdesc) == 9
    assert bdesc[2] == m
    assert bdesc[3] == n
    assert m <= adesc[2]
    assert n <= adesc[3]
    # There is no simple may to check if adesc and bdesc are disjoint to comm_obj
    _gpaw.scalapack_redist(a_obj, adesc, bdesc, comm_obj, m, n)

def scalapack_diagonalize_dc(a_obj, adesc):
    if a_obj is not None:
        assert a_obj.ndim == 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    assert len(adesc) == 9
    _gpaw.scalapack_diagonalize_dc(a_obj, adesc)

def scalapack_general_diagonalize(a_obj, adesc):
    if a_obj is not None:
        assert a_obj.ndim = 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    if b_obj is not None:
        assert a_obj.ndim = 2
        assert (a_obj.dtype == float) or (a_obj.dtype == complex)
        assert a_obj.flags.f_contiguous
    if a_obj is None:
        assert b_obj is None
    assert len(adesc) == 9
    _gpaw.scalapack_diagonalize_dc(a_obj, adesc)

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
