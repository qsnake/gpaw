"""Module defining  ``Eigensolver`` classes."""

from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.lcao.eigensolver import LCAO


def get_eigensolver(name, mode, convergence=None):
    """Create eigensolver object."""
    if name is None:
        if mode == 'lcao':
            name = 'lcao'
        else:
            name = 'rmm-diis'
    if isinstance(name, str):
        eigensolver = {'rmm-diis':  RMM_DIIS,
                       'cg':        CG,
                       'dav':       Davidson,
                       'lcao':      LCAO
                       }[name]()
    else:
        eigensolver = name
    
    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence['eigenstates']

    assert isinstance(eigensolver, LCAO) == (mode == 'lcao')

    return eigensolver
